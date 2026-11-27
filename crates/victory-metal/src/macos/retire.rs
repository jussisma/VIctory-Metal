//! Deferred retirement of GPU-pinned host allocations (ADR-017, ADR-025b).
//!
//! A `MTLCommandBuffer` that is committed asynchronously may still be reading
//! staging slices and shared `xmem::share` leases after `commit` returns. Those
//! resources must stay alive until the GPU signals completion, so the host
//! retires them against the shared fence value the command buffer publishes in
//! its `addCompletedHandler` block and drains them afterwards.
//!
//! This module deliberately holds no Metal type: it is plain, testable host
//! bookkeeping. The transport owns one [`RetireQueue`], pushes every resource a
//! submission pins against that submission's fence value, and calls
//! [`RetireQueue::drain_upto`] with the contiguous completed fence value, both
//! from its `addCompletedHandler:` block and from every frame-boundary drain.
//!
//! # Ordering contract
//!
//! * A retire entry carries the fence value that must be *observed complete*
//!   before the resource may be released. A monotonic fence therefore makes
//!   `drain_upto(completed)` a single pass over the front of the queue.
//! * Pushes are monotonic in normal operation, but an out-of-order push (a
//!   late-retired staging slice behind a newer one) is accepted and released
//!   only when the watermark reaches it. Entries are kept in push order so a
//!   stalled entry never releases a newer one early.
//!
//! # Bounded depth
//!
//! The queue has a soft capacity ([`RetireQueue::DEFAULT_CAPACITY`] by default,
//! or [`RetireQueue::with_capacity`]). It never drops a live entry to stay under
//! it: the only safe eviction is a front entry whose fence the last observed
//! completion already covers, which [`RetireQueue::drain_upto`] and
//! [`RetireQueue::drain_completed`] perform. If the front is stalled, the depth
//! may exceed the capacity, but it is never silent: [`RetireQueue::push`]
//! retains the entry and counts the refusal, [`RetireQueue::high_watermark`]
//! reports the peak depth, and [`RetireQueue::is_over_capacity`] reports the
//! stall. Callers that can apply backpressure should use
//! [`RetireQueue::try_push`], which returns [`RetireQueueFull`] instead of
//! growing the queue.

use std::collections::VecDeque;

/// One deferred release: `id` names the host resource, `fence_value` is the
/// completion value that must be observed before releasing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetiredResource {
    pub fence_value: u64,
    pub id: u64,
}

/// Why a bounded queue refused a new pin.
///
/// The queue is at capacity and its front entry's fence has not completed, so
/// dropping anything would release a live resource. The caller must wait for
/// the oldest fence and drain before retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetireQueueFull {
    /// The configured capacity that was reached.
    pub capacity: usize,
    /// Fence value of the oldest entry; draining it makes room.
    pub oldest: u64,
    /// The completion value the queue has observed so far.
    pub completed: u64,
}

impl std::fmt::Display for RetireQueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "retire queue at capacity {} with oldest fence {} (completed {})",
            self.capacity, self.oldest, self.completed
        )
    }
}

impl std::error::Error for RetireQueueFull {}

/// FIFO of resources awaiting GPU completion.
#[derive(Debug)]
pub struct RetireQueue {
    pending: VecDeque<RetiredResource>,
    capacity: usize,
    /// Highest depth ever observed, so a stalled completion is measurable even
    /// though the queue never silently drops a live entry.
    high_watermark: usize,
    /// Highest completion value any drain has observed.
    completed: u64,
    /// Pins accepted over capacity because their front was still live.
    refusals: u64,
}

impl Default for RetireQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RetireQueue {
    /// Default soft capacity: a floor that keeps a normal frame's pins
    /// comfortable while making a stalled completion visible long before the
    /// queue could be mistaken for unbounded.
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// Creates an empty queue with [`Self::DEFAULT_CAPACITY`].
    pub const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            capacity: Self::DEFAULT_CAPACITY,
            high_watermark: 0,
            completed: 0,
            refusals: 0,
        }
    }

    /// Creates an empty queue with an explicit soft capacity. Zero is clamped
    /// to one so the queue always has a meaningful bound.
    pub const fn with_capacity(capacity: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            capacity: if capacity == 0 { 1 } else { capacity },
            high_watermark: 0,
            completed: 0,
            refusals: 0,
        }
    }

    /// The configured soft capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Highest depth ever reached. Exceeding [`Self::capacity`] means a
    /// completion stalled and pins were retained rather than dropped.
    pub const fn high_watermark(&self) -> usize {
        self.high_watermark
    }

    /// The highest completion value observed by a drain.
    pub const fn completed(&self) -> u64 {
        self.completed
    }

    /// How many pins were accepted over capacity because the front had not
    /// completed (see [`Self::push`]).
    pub const fn refusals(&self) -> u64 {
        self.refusals
    }

    /// True when a stalled completion has pushed the depth above capacity.
    pub fn is_over_capacity(&self) -> bool {
        self.pending.len() > self.capacity
    }

    /// Number of resources still pinned.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True when nothing is pinned.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Pins `id` until `fence_value` completes, bounded by the soft capacity.
    ///
    /// If the queue is at capacity and its front is still live, the pin is
    /// **retained anyway**: a stalled GPU completion cannot be fixed by
    /// discarding bookkeeping, and dropping either the front or the incoming
    /// resource would free a buffer the GPU may still read. The refusal is
    /// counted and visible through [`Self::refusals`], [`Self::high_watermark`]
    /// and [`Self::is_over_capacity`]; callers that can apply backpressure
    /// should drain and use [`Self::try_push`] instead.
    pub fn push(&mut self, fence_value: u64, id: u64) {
        if let Err(full) = self.try_push(fence_value, id) {
            debug_assert_eq!(full.capacity, self.capacity);
            self.refusals += 1;
            self.pending.push_back(RetiredResource { fence_value, id });
            self.high_watermark = self.high_watermark.max(self.pending.len());
        }
    }

    /// Pins `id` until `fence_value` completes, refusing to exceed capacity.
    ///
    /// Unlike [`Self::push`], this never grows the queue past
    /// [`Self::capacity`]. Drain completed entries first
    /// ([`Self::drain_completed`] or [`Self::drain_upto`]); if the front is
    /// still live, [`RetireQueueFull`] is returned and nothing is pinned.
    pub fn try_push(&mut self, fence_value: u64, id: u64) -> Result<(), RetireQueueFull> {
        if self.pending.len() >= self.capacity {
            return Err(RetireQueueFull {
                capacity: self.capacity,
                oldest: self.pending.front().map_or(0, |entry| entry.fence_value),
                completed: self.completed,
            });
        }
        self.pending.push_back(RetiredResource { fence_value, id });
        self.high_watermark = self.high_watermark.max(self.pending.len());
        Ok(())
    }

    /// Oldest pinned fence value, if any. A frame slot may be reused only when
    /// this is `None` or already complete.
    pub fn oldest(&self) -> Option<u64> {
        self.pending.front().map(|entry| entry.fence_value)
    }

    /// Releases every resource whose fence value is `<= completed`, in FIFO
    /// order, and returns their ids.
    ///
    /// A stalled (out-of-order) front entry blocks later entries even if their
    /// own fence values are already complete, because release order is what the
    /// caller's allocator observes.
    pub fn drain_upto(&mut self, completed: u64) -> Vec<u64> {
        let completed = self.completed.max(completed);
        self.completed = completed;
        let mut released = Vec::new();
        while let Some(front) = self.pending.front() {
            if front.fence_value > completed {
                break;
            }
            let entry = self.pending.pop_front().expect("front exists");
            released.push(entry.id);
        }
        released
    }

    /// Releases every resource covered by the last observed completion.
    ///
    /// This is the bounded eviction point when a caller wants room before
    /// pushing: it may only remove entries whose fences already completed.
    pub fn drain_completed(&mut self) -> Vec<u64> {
        let completed = self.completed;
        self.drain_upto(completed)
    }

    /// Drops all bookkeeping without releasing anything. Only for teardown
    /// after the device has been drained and every frame is known complete.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_queue_drains_nothing() {
        let mut queue = RetireQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.oldest(), None);
        assert!(queue.drain_upto(u64::MAX).is_empty());
    }

    #[test]
    fn drain_is_monotonic_and_inclusive() {
        let mut queue = RetireQueue::new();
        queue.push(3, 30);
        queue.push(4, 40);
        queue.push(5, 50);
        assert_eq!(queue.oldest(), Some(3));
        assert!(queue.drain_upto(2).is_empty());
        assert_eq!(queue.drain_upto(3), vec![30]);
        assert_eq!(queue.drain_upto(4), vec![40]);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.drain_upto(u64::MAX), vec![50]);
        assert!(queue.is_empty());
    }

    #[test]
    fn stalled_front_entry_blocks_later_ones() {
        let mut queue = RetireQueue::new();
        queue.push(9, 90);
        queue.push(5, 50);
        assert_eq!(queue.drain_upto(8), Vec::<u64>::new());
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.drain_upto(9), vec![90, 50]);
    }

    #[test]
    fn clear_drops_bookkeeping() {
        let mut queue = RetireQueue::new();
        queue.push(1, 10);
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn capacity_is_explicit_and_zero_is_clamped() {
        assert_eq!(RetireQueue::new().capacity(), RetireQueue::DEFAULT_CAPACITY);
        assert_eq!(RetireQueue::with_capacity(0).capacity(), 1);
        assert_eq!(RetireQueue::with_capacity(7).capacity(), 7);
        assert_eq!(RetireQueue::with_capacity(7).high_watermark(), 0);
        assert_eq!(
            RetireQueue::default().capacity(),
            RetireQueue::DEFAULT_CAPACITY
        );
    }

    #[test]
    fn try_push_refuses_at_capacity_and_drain_makes_room() {
        let mut queue = RetireQueue::with_capacity(2);
        assert_eq!(queue.try_push(1, 10), Ok(()));
        assert_eq!(queue.try_push(2, 20), Ok(()));
        assert_eq!(queue.high_watermark(), 2);
        assert_eq!(
            queue.try_push(3, 30),
            Err(RetireQueueFull {
                capacity: 2,
                oldest: 1,
                completed: 0,
            }),
            "a live front must refuse rather than drop a resource"
        );
        assert_eq!(queue.len(), 2);
        assert!(!queue.is_over_capacity());
        assert_eq!(queue.refusals(), 0);
        assert_eq!(queue.oldest(), Some(1));

        // Draining the live front makes room; nothing was dropped early.
        assert_eq!(queue.drain_upto(1), vec![10]);
        assert_eq!(queue.completed(), 1);
        assert_eq!(queue.try_push(3, 30), Ok(()));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.high_watermark(), 2);
    }

    #[test]
    fn push_retains_a_live_entry_over_capacity_and_reports_it() {
        let mut queue = RetireQueue::with_capacity(2);
        queue.push(100, 0);
        queue.push(101, 1);
        // The front fence 100 never completes: later completed pins must not
        // make the queue drop either the live front or the incoming resource.
        queue.push(102, 2);
        queue.push(103, 3);
        assert_eq!(queue.len(), 4);
        assert_eq!(queue.high_watermark(), 4);
        assert_eq!(queue.refusals(), 2);
        assert!(queue.is_over_capacity());
        assert_eq!(queue.oldest(), Some(100));

        // Nothing was released while the front was live.
        assert_eq!(queue.drain_upto(99), Vec::<u64>::new());
        assert_eq!(queue.len(), 4);

        // Once the front completes, FIFO releases everything in push order.
        assert_eq!(queue.drain_upto(u64::MAX), vec![0, 1, 2, 3]);
        assert!(queue.is_empty());
        assert!(!queue.is_over_capacity());
        // Depth and stall counters stay observable after the queue drains.
        assert_eq!(queue.high_watermark(), 4);
        assert_eq!(queue.refusals(), 2);
    }

    #[test]
    fn drain_completed_uses_the_last_observed_completion() {
        let mut queue = RetireQueue::with_capacity(4);
        queue.push(5, 50);
        queue.push(6, 60);
        queue.push(8, 80);

        assert_eq!(queue.drain_completed(), Vec::<u64>::new());
        assert_eq!(queue.drain_upto(6), vec![50, 60]);
        assert_eq!(queue.completed(), 6);
        // The last observed completion is 6, so the out-of-order 8 stays.
        assert_eq!(queue.drain_completed(), Vec::<u64>::new());
        assert_eq!(queue.len(), 1);

        // A later observation releases it; `completed` never moves backwards.
        assert_eq!(queue.drain_upto(9), vec![80]);
        assert_eq!(queue.completed(), 9);
        assert_eq!(queue.drain_upto(4), Vec::<u64>::new());
        assert_eq!(queue.completed(), 9);
    }
}
