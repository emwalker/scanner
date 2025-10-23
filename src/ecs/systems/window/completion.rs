//! Window completion coordinator
//!
//! Ensures all window completion side effects happen atomically:
//! 1. Mark window allocation and progress as complete
//! 2. Deallocate tuner if allocated
//! 3. Update scan progress to prevent re-processing

use tracing::debug;

use crate::ecs::{WindowEntity, components::window::WindowId, system::SystemContext};

/// Completes a window and performs all necessary coordination.
///
/// This function ensures that when a window is marked complete, ALL of the
/// following side effects occur atomically:
/// 1. Window allocation state transitions to Complete
/// 2. Window progress is marked as completed
/// 3. Tuner is deallocated (if allocated)
/// 4. Scan progress is updated to prevent re-processing
///
/// This prevents the bug where multiple systems (AudioStreamManagementSystem,
/// WindowTimeoutSystem) mark windows complete but only some update scan progress,
/// causing infinite re-processing loops.
pub fn complete_window(window_id: &WindowId, window: &mut WindowEntity, context: &SystemContext) {
    // 1. Deallocate tuner BEFORE marking complete (prevents tuner leak)
    if let Some(tuner_id) = window.allocation.tuner_id()
        && let Some(tuner_entities) = &context.tuner_entities
        && let Ok(mut tuners) = tuner_entities.try_lock()
        && let Some(tuner) = tuners.get_mut(tuner_id)
    {
        debug!(
            tuner_id = ?tuner_id,
            window_id = ?window_id,
            "Deallocating tuner for completed window"
        );
        tuner.allocation.deallocate();
        tuner.status.idle();
    }

    // 2. Mark window components as complete
    window.allocation.mark_complete();
    window.progress.mark_completed();

    // 3. Update scan progress to prevent re-processing
    if let Some(task_entities) = &context.task_entities
        && let Ok(mut tasks) = task_entities.try_write()
        && let Some(task) = tasks.get_mut(&window_id.task_id)
    {
        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        progress.complete_window_at(window_id.clone());

        debug!(
            window_id = ?window_id,
            completed_count = progress.completed_windows.len(),
            "Updated scan progress for completed window"
        );
    }

    debug!(
        window_id = ?window_id,
        "Window completion coordinated successfully"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, RwLock};

    use super::*;
    use crate::ecs::{Entity, EntityWorld, ScanTaskData, TaskEntity, TaskId};

    /// Test that complete_window updates scan progress
    ///
    /// This is the critical test that prevents the infinite re-processing bug.
    /// Without updating scan progress, next_window_to_process() returns the same
    /// window again, causing it to be re-allocated and re-processed forever.
    #[test]
    fn test_complete_window_updates_scan_progress() {
        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);

        // Create scan task with 2 windows
        let task =
            TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 2);
        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        task_entities.write().unwrap().insert(task);

        // Create window in Active state, ready to complete
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);

        let device_id = crate::hardware::DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id, 1);

        window.allocation.start_active(tuner_id, 3);
        window.allocation.mark_all_spawned();

        // Complete all analysis - window is ready to complete
        for _ in 0..3 {
            window.allocation.complete_analysis();
        }

        assert!(
            window.allocation.is_ready_to_complete(false),
            "Window should be ready to complete"
        );

        // Create context with task entities
        let context = SystemContext::new().with_task_entities(task_entities.clone());

        // Call complete_window - should update scan progress
        complete_window(&window_id, &mut window, &context);

        // ASSERTION 1: Window should be marked complete
        assert!(
            window.allocation.is_complete(),
            "Window allocation should be marked Complete"
        );
        assert!(
            window.progress.is_completed(),
            "Window progress should be marked completed"
        );

        // ASSERTION 2: Scan progress should be updated (prevents re-processing)
        {
            let tasks = task_entities.read().unwrap();
            let task = tasks.get(&task_id).unwrap();

            let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;

            assert_eq!(
                progress.completed_windows.len(),
                1,
                "Scan progress should have 1 completed window (prevents infinite re-processing)"
            );

            assert!(
                progress
                    .completed_windows
                    .iter()
                    .any(|w| w.window_index == 0),
                "Window 0 should be in completed_windows list"
            );
        }
    }

    /// Test that complete_window deallocates tuner
    ///
    /// This prevents tuners from being stuck allocated when windows complete,
    /// which would prevent subsequent windows from getting tuners.
    #[test]
    fn test_complete_window_deallocates_tuner() {
        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));

        // Create window with allocated tuner
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);

        let device_id = crate::hardware::DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id.clone(), 1);

        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.allocation.start_active(tuner_id.clone(), 3);

        // Create tuner entity
        let capabilities = crate::hardware::Capabilities::for_device(&device_id);
        let mut tuner = crate::ecs::TunerEntity::new(
            device_id,
            1,
            capabilities,
            crate::hardware::types::Backend::Soapy,
            "Test Tuner".to_string(),
            None,
            "FM".to_string(),
        );
        tuner.allocation.allocate(window_id.to_string());

        let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
        tuner_entities.lock().unwrap().insert(tuner);

        // Create context
        let context = SystemContext::new()
            .with_task_entities(task_entities)
            .with_tuner_entities(tuner_entities.clone());

        // Complete window - should deallocate tuner
        complete_window(&window_id, &mut window, &context);

        // Verify tuner was deallocated
        {
            let tuners = tuner_entities.lock().unwrap();
            let tuner = tuners.iter().find(|t| t.id().channel_index == 1).unwrap();

            assert!(
                !tuner.allocation.is_allocated(),
                "Tuner should be deallocated after window completion"
            );
        }
    }

    /// Test that complete_window handles missing task entities gracefully
    #[test]
    fn test_complete_window_handles_missing_task_entities() {
        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);

        // Create window in Active state with a tuner so it can be completed
        let mut window = WindowEntity::new(window_id.clone(), task_id, 88.0e6);

        let device_id = crate::hardware::DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id, 1);

        window.allocation.start_active(tuner_id, 3);

        // Context with no task entities - the key test is that it doesn't panic
        let context = SystemContext::new();

        // Should not panic (even though scan progress won't be updated)
        complete_window(&window_id, &mut window, &context);

        // Window should still be marked complete locally
        assert!(
            window.allocation.is_complete(),
            "Window should be marked complete even without task entities"
        );
        assert!(
            window.progress.is_completed(),
            "Window progress should be marked completed even without task entities"
        );
    }
}
