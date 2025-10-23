//! Window components for scanning windows

mod allocation;
mod lifecycle;
mod peak_detection;
mod progress;
mod segment;

use std::{
    fmt,
    hash::{Hash, Hasher},
};

pub use allocation::WindowAllocationComponent;
pub use lifecycle::{WindowLifecycleComponent, WindowLifecycleState};
pub use peak_detection::{Peak, PeakDetectionComponent, PeakDetectionState};
pub use progress::{WindowProgressComponent, WindowProgressState};
pub use segment::SegmentComponent;

use crate::ecs::TaskId;

/// Unique identifier for a scanning window
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowId {
    pub task_id: TaskId,
    pub window_index: usize,
}

impl WindowId {
    pub fn new(task_id: TaskId, window_index: usize) -> Self {
        Self {
            task_id,
            window_index,
        }
    }
}

impl Hash for WindowId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.task_id.hash(state);
        self.window_index.hash(state);
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.task_id, self.window_index)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_window_id_equality() {
        let task_id = TaskId::new("task_1");
        let id1 = WindowId::new(task_id.clone(), 0);
        let id2 = WindowId::new(task_id.clone(), 0);
        let id3 = WindowId::new(task_id, 1);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_window_id_hash() {
        let task_id = TaskId::new("task_1");
        let id1 = WindowId::new(task_id.clone(), 0);
        let id2 = WindowId::new(task_id.clone(), 0);
        let id3 = WindowId::new(task_id, 1);

        let mut set = HashSet::new();
        set.insert(id1);
        set.insert(id2);
        set.insert(id3);

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_window_id_different_scans() {
        let task_id1 = TaskId::new("task_1");
        let task_id2 = TaskId::new("task_2");
        let id1 = WindowId::new(task_id1, 0);
        let id2 = WindowId::new(task_id2, 0);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_window_id_display() {
        let task_id = TaskId::new("task_1");
        let id = WindowId::new(task_id, 5);

        assert_eq!(id.to_string(), "task_1-5");
    }
}
