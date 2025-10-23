//! Window processing system - manages window task lifecycle

use std::sync::Arc;

use tracing::{debug, info};

use crate::{
    core::types::{Result, ScanningConfig},
    ecs::{
        Entity, ScanPauseState, TaskId, WindowEntity,
        components::{
            scan::{ScanConfigComponent, ScanLifecycleComponent, ScanProgressComponent},
            window::WindowId,
        },
        system::{System, SystemContext},
    },
    hardware::pool::{Pool, TaskRequirements, TunerActivity},
    shutdown::ShutdownCoordinator,
};

pub struct WindowProcessingSystem {
    config: Arc<ScanningConfig>,
    #[allow(dead_code)]
    pool: Arc<Pool>,
    #[allow(dead_code)]
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    enabled: bool,
}

impl WindowProcessingSystem {
    pub fn new(
        config: Arc<ScanningConfig>,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            config,
            pool,
            shutdown_coordinator,
            enabled: false,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    fn next_window_to_process(&self, progress: &ScanProgressComponent) -> Option<usize> {
        (0..progress.total_windows).find(|&window_index| {
            let indices_match = progress
                .completed_windows
                .iter()
                .any(|w| w.window_index == window_index);
            !indices_match
        })
    }

    fn request_window_allocation(
        &self,
        window_index: usize,
        config: &ScanConfigComponent,
        task_id: &TaskId,
        context: &SystemContext,
    ) {
        let center_freq = config.freq_min + (window_index as f64 * config.step_size);

        let requirements = TaskRequirements {
            frequency_hz: center_freq,
            bandwidth_hz: self.config.samp_rate,
            required_sample_rate: self.config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let requester_id = format!("{}_window_{}", task_id, window_index);
        let window_id = WindowId::new(task_id.clone(), window_index);

        // Queue allocation request to unified queue
        if let Some(allocation_queue) = &context.tuner_allocation_queue {
            let request = crate::ecs::TunerAllocationRequest {
                requester: crate::ecs::TunerRequester::Window(window_id.clone()),
                requirements: requirements.clone(),
                activity: TunerActivity::Scanning,
                requester_id: requester_id.clone(),
            };

            allocation_queue.lock().unwrap().push_back(request);

            // Update window state: None → Requested
            if let Some(window_entities) = &context.window_entities {
                let mut windows = window_entities.write().unwrap();
                if let Some(window) = windows.get_mut(&window_id) {
                    window.allocation.request(
                        requirements,
                        TunerActivity::Scanning,
                        requester_id.clone(),
                    );
                }
            }

            debug!(
                task_id = ?task_id,
                window_index = window_index,
                center_freq_mhz = center_freq / 1e6,
                "Queued tuner allocation request for window"
            );
        }
    }

    fn handle_pending_state(
        &self,
        progress: &mut ScanProgressComponent,
        lifecycle: &mut ScanLifecycleComponent,
        task_id: &TaskId,
    ) {
        info!(
            task_id = ?task_id,
            "WindowProcessingSystem: Scan pending, transitioning to Scanning"
        );
        progress.state = ScanPauseState::Scanning;
        lifecycle.start();
    }

    fn handle_scanning_state(
        &mut self,
        config: &ScanConfigComponent,
        progress: &mut ScanProgressComponent,
        task_id: &TaskId,
        context: &mut SystemContext,
    ) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        // Debug: Log step_size and sample_rate for first window only
        if progress.windows_completed == 0 {
            debug!(
                step_size_hz = config.step_size,
                step_size_mhz = config.step_size / 1e6,
                sample_rate_hz = config.sample_rate,
                sample_rate_mhz = config.sample_rate / 1e6,
                total_windows = progress.total_windows,
                "WindowProcessingSystem: Creating windows with step_size"
            );
        }

        for window_index in 0..progress.total_windows {
            let window_id = WindowId::new(task_id.clone(), window_index);
            let center_freq = config.freq_min + (window_index as f64 * config.step_size);

            // Debug: Log window 22 specifically to diagnose the 110 MHz issue
            if window_index == 22 {
                debug!(
                    window_index = 22,
                    center_freq_hz = center_freq,
                    center_freq_mhz = center_freq / 1e6,
                    freq_min_mhz = config.freq_min / 1e6,
                    step_size_mhz = config.step_size / 1e6,
                    calculation = format!(
                        "{} + 22 * {} = {}",
                        config.freq_min / 1e6,
                        config.step_size / 1e6,
                        center_freq / 1e6
                    ),
                    "WindowProcessingSystem: Window 22 center frequency"
                );
            }

            let mut windows = window_entities.write().unwrap();
            if windows.get(&window_id).is_none() {
                windows.insert(WindowEntity::new(window_id, task_id.clone(), center_freq));
            }
        }

        // NOTE: Worker task processing is handled by WindowWorkerCompletionSystem
        // Do NOT process finished workers here - that violates ECS single responsibility principle
        // and causes WindowWorkerCompletionSystem to miss the finished workers

        let needs_allocation = {
            let windows = window_entities.read().unwrap();
            windows
                .iter()
                .any(|w| &w.id().task_id == task_id && w.allocation.is_none() && w.is_pending())
        };

        if needs_allocation {
            self.handle_no_allocation(config, progress, task_id, context);
        }

        Ok(())
    }

    // NOTE: handle_window_task and process_window_results methods removed
    // This responsibility now belongs to WindowWorkerCompletionSystem
    // Violating ECS single responsibility by processing workers here caused
    // WindowWorkerCompletionSystem to miss finished workers

    fn deallocate_window_tuner(&self, window_id: &WindowId, context: &SystemContext) {
        if let Some(window_entities) = &context.window_entities {
            let windows = window_entities.read().unwrap();
            if let Some(window) = windows.get(window_id)
                && let Some(tuner_id) = window.allocation.tuner_id()
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
        }
    }

    pub(crate) fn handle_no_allocation(
        &self,
        config: &ScanConfigComponent,
        progress: &ScanProgressComponent,
        task_id: &TaskId,
        context: &SystemContext,
    ) {
        let window_entities = match &context.window_entities {
            Some(we) => we,
            None => return,
        };

        // Check if any window is still Processing or Active (serial processing constraint)
        let has_active_window = {
            let windows = window_entities.read().unwrap();
            windows.iter().any(|w| {
                &w.id().task_id == task_id
                    && (w.allocation.is_processing() || w.allocation.is_active())
            })
        };

        if has_active_window {
            // Don't request next window while current one is still processing/analyzing
            return;
        }

        debug!(
            task_id = ?task_id,
            total_windows = progress.total_windows,
            completed_windows = progress.completed_windows.len(),
            "WindowProcessingSystem: Checking for next window"
        );
        if let Some(next_window) = self.next_window_to_process(progress) {
            debug!(
                task_id = ?task_id,
                window_index = next_window,
                "WindowProcessingSystem: Requesting tuner allocation for next window"
            );
            self.request_window_allocation(next_window, config, task_id, context);
        } else {
            debug!(
                task_id = ?task_id,
                "WindowProcessingSystem: No windows to process"
            );
        }
    }

    fn handle_paused_or_listening_state(
        &self,
        task_id: &TaskId,
        state_name: &str,
        window_entities: &std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<WindowEntity>>>,
        context: &SystemContext,
    ) {
        // Collect windows that need tuner deallocation
        let windows_to_deallocate: Vec<WindowId> = {
            let windows = window_entities.read().unwrap();
            windows
                .iter()
                .filter(|w| &w.id().task_id == task_id && !w.allocation.is_none())
                .map(|w| w.id().clone())
                .collect()
        };

        // Cancel tasks
        let mut windows = window_entities.write().unwrap();
        for window in windows.iter_mut() {
            if &window.id().task_id != task_id {
                continue;
            }

            if let Some(mut task) = window.task.take() {
                if !task.cancelling {
                    debug!(
                        task_id = ?task_id,
                        window_index = task.window_index,
                        "WindowProcessingSystem: Cancelling task ({})", state_name
                    );
                    task.cancellation_token.cancel();
                    task.cancelling = true;
                }

                if task.task_handle.is_finished() {
                    debug!(
                        task_id = ?task_id,
                        window_index = task.window_index,
                        "WindowProcessingSystem: Task finished, cleaning up"
                    );
                    let _ = task.task_handle.join();
                } else {
                    debug!(
                        task_id = ?task_id,
                        window_index = task.window_index,
                        "WindowProcessingSystem: Task still running, will check again"
                    );
                    window.task = Some(task);
                }
            }
        }

        drop(windows); // Release lock before deallocating tuners

        // Deallocate tuners from pool (but keep window allocation state)
        for window_id in &windows_to_deallocate {
            self.deallocate_window_tuner(window_id, context);
        }
    }
}

impl System for WindowProcessingSystem {
    fn name(&self) -> &'static str {
        "WindowProcessing"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Don't process windows during global pause
        // This prevents spawning new work or allocating tuners while paused
        if context.is_globally_paused() {
            return Ok(());
        }

        let task_entities = match &context.task_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        let mut tasks = task_entities.write().unwrap();

        for task in tasks.iter_mut() {
            let task_id = task.id().clone();

            let crate::ecs::TaskComponents::Scan {
                config,
                progress,
                lifecycle,
                ..
            } = &mut task.components;
            match progress.state {
                ScanPauseState::Pending => self.handle_pending_state(progress, lifecycle, &task_id),
                ScanPauseState::Scanning => {
                    self.handle_scanning_state(config, progress, &task_id, context)?
                }
                ScanPauseState::PausedAtWindow { .. } => self.handle_paused_or_listening_state(
                    &task_id,
                    "paused",
                    &window_entities,
                    context,
                ),
                ScanPauseState::PausedGlobally { .. } => self.handle_paused_or_listening_state(
                    &task_id,
                    "globally paused",
                    &window_entities,
                    context,
                ),
                ScanPauseState::Listening { .. } => self.handle_paused_or_listening_state(
                    &task_id,
                    "listening",
                    &window_entities,
                    context,
                ),
                ScanPauseState::Completed => {}
                ScanPauseState::WaitingForTuner => {}
                ScanPauseState::TunerOffline => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
