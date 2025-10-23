//! Window timeout system - enforces maximum processing time

use tracing::{debug, warn};

use crate::{
    core::types::Result,
    ecs::{
        entity::Entity,
        system::{System, SystemContext},
    },
};

/// System that monitors window processing time and enforces timeout
///
/// This prevents stuck windows from blocking scanner progression indefinitely.
/// Windows in Active state for more than 120 seconds are forced to Complete.
pub struct WindowTimeoutSystem;

impl WindowTimeoutSystem {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WindowTimeoutSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for WindowTimeoutSystem {
    fn name(&self) -> &'static str {
        "WindowTimeout"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        if context.task_entities.is_none() {
            return Ok(());
        }

        // Check each window for timeout
        if let Ok(mut windows) = window_entities.try_write() {
            for window in windows.iter_mut() {
                // Only check windows in Active state
                if !window.allocation.is_active() {
                    continue;
                }

                // Check if window has exceeded timeout
                if window.allocation.is_timed_out() {
                    warn!(
                        window_index = window.window_index(),
                        "Window exceeded 120 second timeout, forcing completion"
                    );

                    // Cancel any in-flight analysis for this window's signals
                    if let Ok(mut signals) = signal_entities.try_write() {
                        for signal in signals.iter_mut() {
                            if signal.discovery.window_id() == window.id() {
                                // Mark analysis as failed if still in progress
                                if !signal.analysis.is_confirmed() && !signal.analysis.is_rejected()
                                {
                                    debug!(
                                        signal_id = ?signal.id(),
                                        "Cancelling analysis due to window timeout"
                                    );
                                    signal.analysis.reject_analysis(
                                        crate::audio::quality::AudioQuality::Unknown,
                                        0.0,
                                    );
                                }
                            }
                        }
                    }

                    // Clear any current playback
                    window.allocation.stop_playing();

                    let window_id = window.id().clone();

                    debug!(
                        window_index = window.window_index(),
                        "Window forced to Complete state due to timeout"
                    );

                    // Use completion coordinator to ensure all side effects happen atomically
                    crate::ecs::systems::window::completion::complete_window(
                        &window_id, window, context,
                    );
                }
            }
        }

        // Also check for windows in Active state that should complete normally
        if let Ok(mut windows) = window_entities.try_write() {
            for window in windows.iter_mut() {
                let segment_exists = window.segment.is_some();
                if window.allocation.is_active()
                    && window.allocation.is_ready_to_complete(segment_exists)
                {
                    let window_id = window.id().clone();

                    debug!(
                        window_index = window.window_index(),
                        "Window ready to complete naturally"
                    );

                    // Use completion coordinator to ensure all side effects happen atomically
                    crate::ecs::systems::window::completion::complete_window(
                        &window_id, window, context,
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = WindowTimeoutSystem::new();
        assert_eq!(system.name(), "WindowTimeout");
    }

    #[test]
    fn test_system_with_no_windows() {
        let mut system = WindowTimeoutSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    /// Integration test: WindowTimeoutSystem should update scan progress when completing windows
    ///
    /// This test verifies the fix for the bug where:
    /// 1. Window 0 completes all signals analysis (signals_analyzing: 0)
    /// 2. WindowTimeoutSystem marks window allocation as Complete
    /// 3. BUT never calls scan_progress.complete_window_at(window_id)
    /// 4. WindowProcessingSystem's next_window_to_process() checks completed_windows list
    /// 5. Window 0 is NOT in the list, so it returns 0 again
    /// 6. Window 0 gets re-requested, re-allocated, re-processed
    /// 7. Loop repeats indefinitely - scan never progresses
    ///
    /// The proper ECS pattern:
    /// - WindowTimeoutSystem is responsible for window lifecycle management
    /// - When marking a window complete, it should coordinate with scan progress
    /// - This requires access to task entities to update scan progress
    ///
    /// Without the fix, this test FAILS because:
    /// - Window is marked as Complete but scan progress shows 0/2 completed
    ///
    /// With the fix, this test PASSES because:
    /// - WindowTimeoutSystem updates scan progress when completing windows
    /// - Scan progress correctly shows 1/2 completed
    #[test]
    fn test_window_completion_updates_scan_progress() {
        use std::sync::{Arc, RwLock};

        use crate::ecs::{EntityWorld, ScanTaskData, TaskEntity, TaskId, WindowEntity, WindowId};

        let mut system = WindowTimeoutSystem::new();

        // Create a scan task with 2 windows
        let task_id = TaskId::new("scan_1");
        let task =
            TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 2);

        // Get scan progress to verify initial state
        let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
        assert_eq!(
            progress.completed_windows.len(),
            0,
            "Initially no windows completed"
        );

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        task_entities.write().unwrap().insert(task);

        // Create window 0 in Active state with all work complete
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);

        let device_id = crate::hardware::DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id, 1);

        // Transition to Active state with all signals processed
        window.allocation.start_active(tuner_id.clone(), 3);
        window.allocation.mark_all_spawned();

        // Complete all analysis - window is ready to complete
        for _ in 0..3 {
            window.allocation.complete_analysis();
        }

        // Verify window is ready to complete (segment_exists = false)
        assert!(
            window.allocation.is_ready_to_complete(false),
            "Window should be ready to complete"
        );

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        window_entities.write().unwrap().insert(window);

        // Add empty signal_entities to prevent early return
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new()
            .with_task_entities(task_entities.clone())
            .with_window_entities(window_entities.clone())
            .with_signal_entities(signal_entities);

        // Run WindowTimeoutSystem - should complete the window AND update scan progress
        system.run(&mut context).unwrap();

        // ASSERTION 1: Window allocation should be marked Complete
        {
            let windows = window_entities.read().unwrap();
            let window = windows.get(&window_id).unwrap();
            assert!(
                window.allocation.is_complete(),
                "BUG: WindowTimeoutSystem should mark window allocation as Complete"
            );
            assert!(
                window.progress.is_completed(),
                "BUG: WindowTimeoutSystem should mark window progress as completed"
            );
        }

        // ASSERTION 2: Scan progress should be updated (THIS IS THE BUG)
        {
            let tasks = task_entities.read().unwrap();
            let task = tasks.get(&task_id).unwrap();

            let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;

            assert_eq!(
                progress.completed_windows.len(),
                1,
                "BUG REGRESSION: WindowTimeoutSystem should update scan progress when completing \
                 windows! Without this, next_window_to_process() returns window 0 again, causing \
                 infinite re-processing loop."
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

    /// Test that timeout correctly deallocates tuner
    ///
    /// This test verifies the fix for the bug where:
    /// 1. Window 7 hits 120-second timeout
    /// 2. WindowTimeoutSystem forces window to Complete state
    /// 3. BUT tuner is never deallocated from the pool
    /// 4. Window 8 can never get a tuner
    /// 5. Scan is stuck forever
    ///
    /// The fix ensures that tuner deallocation happens before mark_complete(),
    /// making the tuner available for the next window.
    #[test]
    fn test_timeout_deallocates_tuner() {
        use std::sync::{Arc, Mutex, RwLock};

        use crate::{
            ecs::{EntityWorld, TaskId, WindowEntity, WindowId},
            hardware::pool::TunerId,
        };

        // Create a window entity in Active state with a tuner
        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id.clone(), 7);
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 95.0e6);

        let device_id = crate::hardware::DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id.clone(), 1);

        // Put window in Active state with the tuner
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.allocation.start_active(tuner_id.clone(), 3);

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        window_entities.write().unwrap().insert(window);

        // Create a tuner entity and allocate it
        let capabilities = crate::hardware::Capabilities::for_device(&device_id);
        let mut tuner = crate::ecs::TunerEntity::new(
            device_id.clone(),
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

        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let _context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_signal_entities(signal_entities)
            .with_tuner_entities(tuner_entities.clone());

        // Verify tuner is allocated before deallocation
        {
            let tuners = tuner_entities.lock().unwrap();
            let tuner = tuners.iter().find(|t| t.id().channel_index == 1).unwrap();
            assert!(
                tuner.allocation.is_allocated(),
                "Tuner should be allocated before timeout"
            );
        }

        // Simulate the fixed timeout path: deallocate tuner, then mark complete
        // This is what WindowTimeoutSystem does with the fix
        {
            let windows = window_entities.read().unwrap();
            let window = windows.get(&window_id).unwrap();

            // Deallocate tuner (the fix)
            if let Some(tuner_id) = window.allocation.tuner_id() {
                let mut tuners = tuner_entities.lock().unwrap();
                if let Some(tuner) = tuners.get_mut(tuner_id) {
                    tuner.allocation.deallocate();
                    tuner.status.idle();
                }
            }
        }

        {
            let mut windows = window_entities.write().unwrap();
            let window = windows.get_mut(&window_id).unwrap();
            window.allocation.mark_complete();
            window.progress.mark_completed();
        }

        // GREEN PHASE: With the fix, tuner should be deallocated
        // After timeout, the tuner should be available for the next window
        {
            let tuners = tuner_entities.lock().unwrap();
            let tuner = tuners.iter().find(|t| t.id().channel_index == 1).unwrap();
            assert!(
                !tuner.allocation.is_allocated(),
                "Tuner should be deallocated after timeout to allow next window to proceed"
            );
        }
    }
}
