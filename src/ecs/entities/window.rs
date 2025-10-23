//! Window entity - represents a scanning window being processed

use crate::ecs::{
    Entity,
    components::{
        scan::WindowWorkerComponent,
        window::{
            PeakDetectionComponent, SegmentComponent, WindowAllocationComponent, WindowId,
            WindowLifecycleComponent, WindowProgressComponent,
        },
    },
    entities::task::TaskId,
};

/// Entity representing a single scanning window
///
/// A window entity combines allocation, segment, task, and progress
/// tracking for one frequency window during a band scan.
#[derive(Debug)]
pub struct WindowEntity {
    id: WindowId,
    pub task_id: TaskId,
    center_frequency_hz: f64,
    pub allocation: WindowAllocationComponent,
    pub segment: Option<SegmentComponent>,
    pub task: Option<WindowWorkerComponent>,
    pub progress: WindowProgressComponent,
    pub peak_detection: PeakDetectionComponent,
    pub lifecycle: WindowLifecycleComponent,
}

impl WindowEntity {
    pub fn new(id: WindowId, task_id: TaskId, center_frequency_hz: f64) -> Self {
        Self {
            id,
            task_id,
            center_frequency_hz,
            allocation: WindowAllocationComponent::new(),
            segment: None,
            task: None,
            progress: WindowProgressComponent::new(),
            peak_detection: PeakDetectionComponent::new(),
            lifecycle: WindowLifecycleComponent::new(),
        }
    }

    pub fn window_index(&self) -> usize {
        self.id.window_index
    }

    pub fn center_frequency_hz(&self) -> f64 {
        self.center_frequency_hz
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

    fn create_test_window() -> WindowEntity {
        let task_id = TaskId::new("test_scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        WindowEntity::new(window_id, task_id, 88.5e6)
    }

    #[test]
    fn test_window_entity_creation() {
        let entity = create_test_window();

        assert_eq!(entity.window_index(), 0);
        assert!(entity.is_pending());
        assert!(!entity.has_segment());
        assert!(!entity.has_task());
        assert_eq!(entity.task_id.0, "test_scan_1");
        assert!((entity.center_frequency_hz() - 88.5e6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_entity_lifecycle() {
        let task_id = TaskId::new("test_scan_1");
        let window_id = WindowId::new(task_id.clone(), 5);
        let mut entity = WindowEntity::new(window_id, task_id, 95.5e6);

        assert_eq!(entity.window_index(), 5);
        assert!(entity.is_pending());

        entity.progress.start_processing();
        assert!(entity.is_processing());

        entity.progress.mark_completed();
        assert!(entity.is_completed());
    }

    #[test]
    fn test_window_entity_has_peak_detection() {
        let window_id = WindowId::new(TaskId::new("scan_1"), 0);
        let task_id = TaskId::new("scan_1");
        let entity = WindowEntity::new(window_id, task_id, 101.0e6);

        assert!(entity.peak_detection.is_pending());
    }

    #[test]
    fn test_window_entity_peak_detection_lifecycle() {
        use crate::ecs::components::window::Peak;

        let window_id = WindowId::new(TaskId::new("scan_1"), 0);
        let task_id = TaskId::new("scan_1");
        let mut entity = WindowEntity::new(window_id, task_id, 101.0e6);

        assert!(entity.peak_detection.is_pending());

        let peaks = vec![Peak {
            frequency_hz: 88.1e6,
            magnitude: 0.8,
        }];
        entity.peak_detection.complete_detection(peaks);

        assert!(entity.peak_detection.is_complete());
    }
}
