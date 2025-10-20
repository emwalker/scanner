//! Window entity - represents a scanning window being processed

use crate::ecs::Entity;
use crate::ecs::components::scan::WindowTaskComponent;
use crate::ecs::components::window::{
    SegmentComponent, WindowAllocationComponent, WindowId, WindowProgressComponent,
};

/// Entity representing a single scanning window
///
/// A window entity combines allocation, segment, task, and progress
/// tracking for one frequency window during a band scan.
#[derive(Debug)]
pub struct WindowEntity {
    id: WindowId,
    pub allocation: WindowAllocationComponent,
    pub segment: Option<SegmentComponent>,
    pub task: Option<WindowTaskComponent>,
    pub progress: WindowProgressComponent,
}

impl WindowEntity {
    pub fn new(id: WindowId) -> Self {
        Self {
            id,
            allocation: WindowAllocationComponent::new(),
            segment: None,
            task: None,
            progress: WindowProgressComponent::new(),
        }
    }

    pub fn window_index(&self) -> usize {
        self.id.window_index
    }

    pub fn is_pending(&self) -> bool {
        self.progress.is_pending()
    }

    pub fn is_processing(&self) -> bool {
        self.progress.is_processing()
    }

    pub fn is_completed(&self) -> bool {
        self.progress.is_completed()
    }

    pub fn is_failed(&self) -> bool {
        self.progress.is_failed()
    }

    pub fn has_segment(&self) -> bool {
        self.segment.is_some()
    }

    pub fn has_task(&self) -> bool {
        self.task.is_some()
    }
}

impl Entity for WindowEntity {
    type Id = WindowId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::ScanId;

    #[test]
    fn test_window_entity_creation() {
        let scan_id = ScanId::new();
        let window_id = WindowId::new(scan_id, 0);
        let entity = WindowEntity::new(window_id);

        assert_eq!(entity.window_index(), 0);
        assert!(entity.is_pending());
        assert!(!entity.has_segment());
        assert!(!entity.has_task());
    }

    #[test]
    fn test_window_entity_lifecycle() {
        let scan_id = ScanId::new();
        let window_id = WindowId::new(scan_id, 5);
        let mut entity = WindowEntity::new(window_id);

        assert_eq!(entity.window_index(), 5);
        assert!(entity.is_pending());

        entity.progress.start_processing();
        assert!(entity.is_processing());

        entity.progress.mark_completed();
        assert!(entity.is_completed());
    }
}
