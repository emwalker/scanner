//! Window components for scanning windows

mod allocation;
mod progress;
mod segment;

pub use allocation::WindowAllocationComponent;
pub use progress::{WindowProgressComponent, WindowProgressState};
pub use segment::SegmentComponent;

use crate::ecs::components::scan::ScanId;
use std::hash::{Hash, Hasher};

/// Unique identifier for a scanning window
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowId {
    pub scan_id: ScanId,
    pub window_index: usize,
}

impl WindowId {
    pub fn new(scan_id: ScanId, window_index: usize) -> Self {
        Self {
            scan_id,
            window_index,
        }
    }
}

impl Hash for WindowId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scan_id.hash(state);
        self.window_index.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_window_id_equality() {
        let scan_id = ScanId::new();
        let id1 = WindowId::new(scan_id, 0);
        let id2 = WindowId::new(scan_id, 0);
        let id3 = WindowId::new(scan_id, 1);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_window_id_hash() {
        let scan_id = ScanId::new();
        let id1 = WindowId::new(scan_id, 0);
        let id2 = WindowId::new(scan_id, 0);
        let id3 = WindowId::new(scan_id, 1);

        let mut set = HashSet::new();
        set.insert(id1);
        set.insert(id2);
        set.insert(id3);

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_window_id_different_scans() {
        let scan_id1 = ScanId::new();
        let scan_id2 = ScanId::new();
        let id1 = WindowId::new(scan_id1, 0);
        let id2 = WindowId::new(scan_id2, 0);

        assert_ne!(id1, id2);
    }
}
