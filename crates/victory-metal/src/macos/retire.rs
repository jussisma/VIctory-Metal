//! Deferred retirement of GPU-pinned host allocations (ADR-017).
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

use std::collections::VecDeque;

/// One deferred release: `id` names the host resource, `fence_value` is the
/// completion value that must be observed before releasing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetiredResource {
    pub fence_value: u64,
    pub id: u64,
}

/// FIFO of resources awaiting GPU completion.
#[derive(Debug, Default)]
pub struct RetireQueue {
    pending: VecDeque<RetiredResource>,
}

impl RetireQueue {
    /// Creates an empty queue.
    pub const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }

    /// Number of resources still pinned.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True when nothing is pinned.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Pins `id` until `fence_value` completes.
    pub fn push(&mut self, fence_value: u64, id: u64) {
        self.pending.push_back(RetiredResource { fence_value, id });
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
}
