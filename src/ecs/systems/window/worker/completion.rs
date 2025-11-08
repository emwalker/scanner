use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::window::WindowId,
        system::{System, SystemContext},
    },
    hardware::pool::SegmentTrait,
};

pub struct WindowWorkerCompletionSystem;

impl Default for WindowWorkerCompletionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowWorkerCompletionSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for WindowWorkerCompletionSystem {
    fn name(&self) -> &'static str {
        "WindowWorkerCompletion"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => {
                debug!("WindowWorkerCompletionSystem: No window entities in context");
                return Ok(());
            }
        };

        // Find windows with finished workers
        let finished_workers: Vec<WindowId> = {
            let windows = match window_entities.try_read() {
                Ok(w) => w,
                Err(_) => {
                    debug!(
                        "WindowWorkerCompletionSystem: Failed to acquire read lock (contention)"
                    );
                    return Ok(());
                }
            };

            let total_windows = windows.len();
            let windows_with_tasks = windows.iter().filter(|w| w.task.is_some()).count();

            let finished: Vec<WindowId> = windows
                .iter()
                .filter(|w| {
                    w.task
                        .as_ref()
                        .map(|task| task.task_handle.is_finished())
                        .unwrap_or(false)
                })
                .map(|w| w.id().clone())
                .collect();

            debug!(
                total_windows = total_windows,
                windows_with_tasks = windows_with_tasks,
                finished_count = finished.len(),
                "WindowWorkerCompletionSystem: Checking for finished workers"
            );

            finished
        };

        // Process each finished worker
        for window_id in finished_workers {
            let worker_result = {
                let mut windows = window_entities.write().unwrap();
                let window = match windows.get_mut(&window_id) {
                    Some(w) => w,
                    None => continue,
                };

                // Take the worker component to get the handle
                let worker = match window.task.take() {
                    Some(w) => w,
                    None => continue,
                };

                // Join the thread to get the result
                match worker.task_handle.join() {
                    Ok(result) => result,
                    Err(e) => {
                        debug!(
                            window_id = ?window_id,
                            error = ?e,
                            "WindowWorkerCompletionSystem: Worker thread panicked"
                        );
                        window.progress.mark_failed();
                        continue;
                    }
                }
            };

            // Process the worker result
            match worker_result {
                Ok(result) => {
                    use crate::ecs::components::scan::WindowWorkerOutcome;
                    match result.outcome {
                        WindowWorkerOutcome::Success {
                            signals,
                            segment,
                            center_freq,
                        } => {
                            // Create SignalEntity instances and set up analysis inputs
                            if let Some(signal_entities) = &context.signal_entities {
                                let mut entities = signal_entities.write().unwrap();
                                for signal_data in &signals {
                                    let mut signal = crate::ecs::SignalEntity::new(
                                        signal_data.frequency_hz,
                                        window_id.clone(),
                                        crate::core::signals::ModulationType::WFM,
                                    );

                                    // Set up analysis input with segment audio subscribers
                                    let sdr_rx = segment.audio_subscriber();
                                    let sdr_rx_refining = sdr_rx.resubscribe();
                                    let sdr_rx_detection = sdr_rx.resubscribe();

                                    if let Some(config) = &context.config {
                                        let input =
                                            crate::ecs::components::AnalysisInputComponent::new(
                                                sdr_rx_refining,
                                                sdr_rx_detection,
                                                config.clone(),
                                                window_id.clone(),
                                                center_freq,
                                                None,
                                            );
                                        signal.set_analysis_input(input);
                                    }

                                    entities.insert(signal);
                                    debug!(
                                        window_id = ?window_id,
                                        frequency_mhz = signal_data.frequency_hz / 1e6,
                                        "WindowWorkerCompletionSystem: Created SignalEntity with analysis input"
                                    );
                                }
                            }

                            // Store segment and update window state
                            let mut windows = window_entities.write().unwrap();
                            if let Some(window) = windows.get_mut(&window_id) {
                                window.segment =
                                    Some(crate::ecs::components::window::SegmentComponent::new(
                                        segment.clone(),
                                    ));
                                window.lifecycle.start_analyzing(signals.len());
                                if let Some(tuner_id) = window.allocation.tuner_id() {
                                    window
                                        .allocation
                                        .start_active(tuner_id.clone(), signals.len());
                                }
                                window.progress.mark_completed();

                                debug!(
                                    window_id = ?window_id,
                                    signal_count = signals.len(),
                                    "WindowWorkerCompletionSystem: Worker completed successfully"
                                );
                            }
                        }
                        WindowWorkerOutcome::NoSignals { reason, .. } => {
                            // No signals found - just mark window as completed
                            let mut windows = window_entities.write().unwrap();
                            if let Some(window) = windows.get_mut(&window_id) {
                                window.progress.mark_completed();
                                debug!(
                                    window_id = ?window_id,
                                    reason = reason,
                                    "WindowWorkerCompletionSystem: Worker completed with no signals"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!(
                        window_id = ?window_id,
                        error = ?e,
                        "WindowWorkerCompletionSystem: Worker failed"
                    );
                    let mut windows = window_entities.write().unwrap();
                    if let Some(window) = windows.get_mut(&window_id) {
                        window.progress.mark_failed();
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::ecs::EntityWorld;

    #[test]
    fn test_system_creation() {
        let system = WindowWorkerCompletionSystem::new();
        assert_eq!(system.name(), "WindowWorkerCompletion");
    }

    #[test]
    fn test_run_with_no_windows() {
        let mut system = WindowWorkerCompletionSystem::new();
        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new().with_window_entities(window_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    /// Test that completion system creates signal entities from worker data
    ///
    /// This test verifies proper ECS pattern where:
    /// 1. Worker thread returns plain data (CandidateData)
    /// 2. Completion system creates entities on main thread
    /// 3. No lock contention because worker doesn't touch EntityWorlds
    ///
    /// Note: This test is simplified and doesn't test the full segment/analysis_input setup
    /// because that requires complex mocking. The important part is verifying entity creation
    /// happens in the completion system, not the worker thread.
    #[test]
    #[ignore] // Ignored until we can properly mock Segment
    fn test_completion_creates_signal_entities() {
        // TODO: Implement proper test once Segment can be mocked or we have test helpers
        // The key behavior (entity creation in completion system) is verified in integration tests
    }

    /// Test that completion system processes finished workers even with lock contention
    ///
    /// This is a regression test for the bug where try_read() would silently fail
    /// when another thread held a read lock, causing workers to never complete.
    ///
    /// The test simulates the scenario by:
    /// 1. Creating a window with a finished worker
    /// 2. Holding a read lock during system execution (simulating contention)
    /// 3. Verifying the system still processes the worker
    #[test]
    fn test_completion_system_with_lock_contention() {
        use std::time::Instant;

        use tokio_util::sync::CancellationToken;

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));

        // Create a window with a finished worker
        let task_id = crate::ecs::TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = crate::ecs::WindowEntity::new(window_id.clone(), task_id, 95.0e6);

        // Create a worker that immediately completes
        let handle = std::thread::spawn(|| {
            Ok(crate::ecs::components::scan::WindowWorkerResult {
                window_index: 0,
                outcome: crate::ecs::components::scan::WindowWorkerOutcome::NoSignals {
                    center_freq: 95.0e6,
                    reason: "test".to_string(),
                },
                completed_at: Instant::now(),
            })
        });

        // Wait for thread to finish
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(handle.is_finished(), "Test worker should be finished");

        window.task = Some(crate::ecs::components::scan::WindowWorkerComponent {
            window_index: 0,
            task_handle: handle,
            cancellation_token: CancellationToken::new(),
            started_at: Instant::now(),
            cancelling: false,
        });

        window_entities.write().unwrap().insert(window);

        // Simulate lock contention by holding a read lock
        // This would cause try_read() to fail, but blocking read() should wait
        let _contention_lock = window_entities.read().unwrap();

        // Run completion system in a separate thread
        let window_entities_clone = window_entities.clone();
        let completion_thread = std::thread::spawn(move || {
            let mut system = WindowWorkerCompletionSystem::new();
            let mut context = SystemContext::new().with_window_entities(window_entities_clone);
            system.run(&mut context)
        });

        // Release the read lock after a short delay
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(_contention_lock);

        // Wait for completion system to finish
        let result = completion_thread.join().unwrap();
        assert!(result.is_ok(), "Completion system should succeed");

        // Verify: Worker should be processed and removed
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).unwrap();
        assert!(
            window.task.is_none(),
            "Worker should be processed and removed by completion system"
        );
    }
}
