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
//! # Bounded depth (hard)
//!
//! The capacity ([`RetireQueue::DEFAULT_CAPACITY`] by default, or
//! [`RetireQueue::with_capacity`]) is a **hard bound** for every caller that
//! pins through [`RetireQueue::push_backpressure`] or
//! [`RetireQueue::try_push`]; both never grow the queue past it and never drop a
//! live entry to stay under it. The only safe eviction is a front entry whose
//! fence the last observed completion already covers, which
//! [`RetireQueue::drain_upto`] and [`RetireQueue::drain_completed`] perform.
//!
//! A caller that cannot wait (or that must not block under its own lock) pins
//! with [`RetireQueue::try_push`] and handles [`RetireQueueFull`] itself. A
//! caller that can wait pins with [`RetireQueue::push_backpressure`], which
//! retries a bounded number of times ([`RetireQueue::MAX_BACKPRESSURE_WAITS`])
//! and reports [`RetireBackpressure`] instead of hanging on a GPU fence that
//! never completes.
//!
//! [`RetireQueue::push`] is the legacy *soft* path: it retains a live entry over
//! capacity and counts the refusal. It is kept only for existing single-threaded
//! callers that have no way to wait; the device must not use it, because a hard
//! bound is what makes a stuck completion a bounded, typed failure instead of
//! unbounded host memory growth.

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

/// Why a pin failed even though the caller waited for completion.
///
/// The queue stayed full after [`RetireQueue::MAX_BACKPRESSURE_WAITS`] waits on
/// the oldest live fence: the GPU or its completion handler never published a
/// value that covered that fence. The pin was **not** queued and no live entry
/// was dropped, so the caller may retry once it has other evidence of progress
/// (an external device reset, for example).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetireBackpressure {
    /// The capacity that is still full.
    pub capacity: usize,
    /// Fence value of the front entry that never completed.
    pub oldest: u64,
    /// Highest completion value observed when the budget was spent.
    pub completed: u64,
    /// How many waits the pin performed before giving up.
    pub waits: u32,
}

impl std::fmt::Display for RetireBackpressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "retire backpressure: capacity {} still full on fence {} after {} waits \
             (completed {})",
            self.capacity, self.oldest, self.waits, self.completed
        )
    }
}

impl std::error::Error for RetireBackpressure {}

/// Bounded retry budget for one pin that does not fit.
///
/// Shared by [`RetireQueue::push_backpressure`] and the transport, which drives
/// the same policy across its own mutex: the waiter must not hold the lock the
/// completion handler needs, so the transport locks per attempt and calls
/// [`Self::next`] for the same bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackpressureBudget {
    waits: u32,
}

impl Default for BackpressureBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl BackpressureBudget {
    /// A fresh budget that allows [`RetireQueue::MAX_BACKPRESSURE_WAITS`] waits.
    pub const fn new() -> Self {
        Self { waits: 0 }
    }

    /// How many waits this pin has already performed.
    pub const fn waits_used(&self) -> u32 {
        self.waits
    }

    /// Consumes one wait and returns the fence to wait on.
    ///
    /// Returns [`RetireBackpressure`] when the budget is spent; the caller must
    /// then fail the pin instead of waiting again.
    pub fn next(&mut self, full: RetireQueueFull) -> Result<u64, RetireBackpressure> {
        if self.waits >= RetireQueue::MAX_BACKPRESSURE_WAITS {
            return Err(RetireBackpressure {
                capacity: full.capacity,
                oldest: full.oldest,
                completed: full.completed,
                waits: self.waits,
            });
        }
        self.waits += 1;
        Ok(full.oldest)
    }
}

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
    /// Default hard capacity: a floor that keeps a normal frame's pins
    /// comfortable while making a stalled completion visible long before the
    /// queue could be mistaken for unbounded.
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// How many times [`Self::push_backpressure`] waits on the oldest live fence
    /// for one pin before giving up.
    ///
    /// Four waits cover the worst honest case on this transport: the in-flight
    /// cap is at most three, and a pin behind the newest fence may have to watch
    /// the whole in-flight window drain before its own fence is covered. A fifth
    /// unanswered wait means the GPU or its completion handler is stuck, and the
    /// pin fails with [`RetireBackpressure`] instead of blocking the host
    /// forever.
    pub const MAX_BACKPRESSURE_WAITS: u32 = 4;

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
    /// **Deprecated for the device path.** This retains a live pin over
    /// capacity and only counts the refusal, so a stalled completion grows the
    /// queue without bound. It exists for a single-threaded caller that has no
    /// way to wait on the GPU; the transport must use
    /// [`Self::push_backpressure`], which keeps the capacity hard.
    ///
    /// If the queue is at capacity and its front is still live, the pin is
    /// **retained anyway**: a stalled GPU completion cannot be fixed by
    /// discarding bookkeeping, and dropping either the front or the incoming
    /// resource would free a buffer the GPU may still read. The refusal is
    /// counted and visible through [`Self::refusals`], [`Self::high_watermark`]
    /// and [`Self::is_over_capacity`].
    pub fn push(&mut self, fence_value: u64, id: u64) {
        if let Err(full) = self.try_push(fence_value, id) {
            debug_assert_eq!(full.capacity, self.capacity);
            self.refusals += 1;
            self.pending.push_back(RetiredResource { fence_value, id });
            self.high_watermark = self.high_watermark.max(self.pending.len());
        }
    }

    /// Pins `id` until `fence_value` completes, applying bounded backpressure.
    ///
    /// The capacity is a hard bound: this never grows the queue and never drops
    /// a live entry. When the queue is full, `wait` is called with the oldest
    /// live fence and must make progress visible to the queue:
    ///
    /// * return `true` only when it has *observed* that the oldest fence
    ///   completed, i.e. the published contiguous watermark now covers it. The
    ///   queue then releases every entry up to that fence (the watermark is
    ///   contiguous by construction, so an out-of-order front entry behind it is
    ///   complete too) and retries the push;
    /// * return `false` when it made no progress; the queue retries without
    ///   releasing anything.
    ///
    /// At most [`Self::MAX_BACKPRESSURE_WAITS`] waits are performed. When the
    /// budget is spent the pin is **not** queued and [`RetireBackpressure`] is
    /// returned, so a stuck GPU is a typed failure rather than a hang.
    ///
    /// The ids released while making room are returned in FIFO order, so a
    /// caller that owns the resources can drop them without leaking. (The
    /// transport instead drains under its own mutex; see
    /// `DirectMetalTransport::retire_pin`.)
    pub fn push_backpressure<W>(
        &mut self,
        fence_value: u64,
        id: u64,
        mut wait: W,
    ) -> Result<Vec<u64>, RetireBackpressure>
    where
        W: FnMut(u64) -> bool,
    {
        let mut budget = BackpressureBudget::new();
        let mut released = Vec::new();
        loop {
            match self.try_push(fence_value, id) {
                Ok(()) => return Ok(released),
                Err(full) => {
                    let oldest = budget.next(full)?;
                    if wait(oldest) {
                        // Only the fence the waiter proved complete may be
                        // released: `drain_upto` keeps every later entry.
                        released.extend(self.drain_upto(oldest));
                    }
                }
            }
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

    /// 10,000 alloc/use/retire cycles with a hard bound of 8 and of 64: the
    /// depth never exceeds the capacity, no pin is ever refused, and every pin
    /// is eventually released in push order.
    #[test]
    fn backpressure_keeps_depth_bounded_under_churn() {
        for capacity in [8usize, 64] {
            let mut queue = RetireQueue::with_capacity(capacity);
            let mut released = Vec::new();
            let mut refused = 0u64;
            let mut waits = 0u32;
            for cycle in 0..10_000u64 {
                // Every pin sits behind a monotonically increasing fence; the
                // waiter simulates the GPU finishing the front fence.
                let fence = cycle + 1;
                match queue.push_backpressure(fence, cycle, |oldest| {
                    waits += 1;
                    assert!(oldest <= fence, "waiter asked for a future fence");
                    true
                }) {
                    Ok(mut ids) => released.append(&mut ids),
                    Err(_) => refused += 1,
                }
                assert!(
                    queue.len() <= capacity,
                    "capacity {capacity} exceeded: {} (cycle {cycle})",
                    queue.len()
                );
                assert!(
                    queue.high_watermark() <= capacity,
                    "high watermark {} exceeded capacity {capacity}",
                    queue.high_watermark()
                );
            }
            assert_eq!(refused, 0, "a wait that advances must never refuse a pin");
            assert_eq!(
                queue.refusals(),
                0,
                "the soft path must not be reachable through push_backpressure"
            );
            // Drain the residue and check FIFO release order over everything.
            released.extend(queue.drain_upto(u64::MAX));
            assert_eq!(
                released,
                (0..10_000u64).collect::<Vec<_>>(),
                "release order must match push order"
            );

            eprintln!(
                "retire backpressure: capacity={capacity} cycles=10000 waits={waits} \
                 high_watermark={}",
                queue.high_watermark()
            );
        }
    }

    /// A waiter that never observes a completion must hit the typed error after
    /// exactly the documented number of waits, leaving the queue untouched.
    #[test]
    fn stuck_waiter_returns_the_typed_backpressure_error() {
        let mut queue = RetireQueue::with_capacity(4);
        for fence in 1..=4 {
            queue.try_push(fence, fence * 10).expect("capacity 4");
        }
        let mut waits = 0u32;
        let error = queue
            .push_backpressure(5, 50, |_oldest| {
                waits += 1;
                false
            })
            .expect_err("a stuck completion must not pin");
        assert_eq!(waits, RetireQueue::MAX_BACKPRESSURE_WAITS);
        assert_eq!(
            error,
            RetireBackpressure {
                capacity: 4,
                oldest: 1,
                completed: 0,
                waits: RetireQueue::MAX_BACKPRESSURE_WAITS,
            }
        );
        assert_eq!(queue.len(), 4, "no entry may be dropped or added");
        assert_eq!(queue.oldest(), Some(1));
        assert_eq!(queue.refusals(), 0);
        assert_eq!(
            error.to_string(),
            "retire backpressure: capacity 4 still full on fence 1 after 4 waits \
             (completed 0)"
        );
        // Nothing was released: the incoming pin never became live.
        assert_eq!(queue.drain_upto(4), vec![10, 20, 30, 40]);
    }

    /// Backpressure releases in FIFO order and only what the waiter proved.
    #[test]
    fn backpressure_releases_in_push_order() {
        let mut queue = RetireQueue::with_capacity(2);
        queue.try_push(1, 10).expect("capacity 2");
        queue.try_push(2, 20).expect("capacity 2");

        // Prove only fence 1: the queue frees one slot and keeps fence 2.
        let mut waits = 0u32;
        let released = queue
            .push_backpressure(3, 30, |oldest| {
                waits += 1;
                assert_eq!(oldest, 1);
                true
            })
            .expect("fence 1 completing makes room");
        assert_eq!(waits, 1);
        assert_eq!(released, vec![10]);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.oldest(), Some(2));

        // The push order is preserved behind the stalled fence 2.
        assert_eq!(queue.drain_upto(u64::MAX), vec![20, 30]);
        assert_eq!(queue.completed(), u64::MAX);
        assert_eq!(queue.high_watermark(), 2);
        assert_eq!(queue.refusals(), 0);
    }

    /// The budget reports the wait count and builds the typed error itself.
    #[test]
    fn backpressure_budget_is_bounded_and_typed() {
        let full = RetireQueueFull {
            capacity: 8,
            oldest: 42,
            completed: 41,
        };
        let mut budget = BackpressureBudget::new();
        for expected in 1..=RetireQueue::MAX_BACKPRESSURE_WAITS {
            assert_eq!(budget.next(full), Ok(42));
            assert_eq!(budget.waits_used(), expected);
        }
        assert_eq!(
            budget.next(full),
            Err(RetireBackpressure {
                capacity: 8,
                oldest: 42,
                completed: 41,
                waits: RetireQueue::MAX_BACKPRESSURE_WAITS,
            })
        );
        assert_eq!(BackpressureBudget::default(), BackpressureBudget::new());
    }
}
