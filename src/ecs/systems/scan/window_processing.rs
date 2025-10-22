//! Window processing system - manages window task lifecycle

use crate::core::types::{Result, ScanningConfig};
use crate::ecs::components::scan::{WindowTaskComponent, WindowTaskResult};
use crate::ecs::components::window::WindowId;
use crate::ecs::system::{System, SystemContext};
use crate::ecs::{Entity, ScanPauseState, WindowEntity};
use crate::hardware::pool::{Pool, TaskRequirements, TunerActivity};
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

pub struct WindowProcessingSystem {
    config: Arc<ScanningConfig>,
    pool: Arc<Pool>,
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

    fn next_window_to_process(&self, scan: &crate::ecs::ScanEntity) -> Option<usize> {
        (0..scan.progress.total_windows)
            .find(|&window_index| !scan.progress.is_window_completed(window_index))
    }

    fn request_window_allocation(
        &self,
        window_index: usize,
        scan: &crate::ecs::ScanEntity,
        window_entities: &std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<WindowEntity>>>,
    ) {
        let center_freq = scan.config.freq_min + (window_index as f64 * scan.config.window_size);

        let requirements = TaskRequirements {
            frequency_hz: center_freq,
            bandwidth_hz: self.config.samp_rate,
            required_sample_rate: self.config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let requester_id = format!("scan_{}_window_{}", scan.id().value(), window_index);

        let window_id = WindowId::new(*scan.id(), window_index);
        let mut windows = window_entities.write().unwrap();
        if let Some(window) = windows.get_mut(&window_id) {
            window
                .allocation
                .request(requirements, TunerActivity::Scanning, requester_id);
        }

        debug!(
            scan_id = ?scan.id(),
            window_index = window_index,
            center_freq_mhz = center_freq / 1e6,
            "Requested tuner allocation for window"
        );
    }

    fn spawn_window_task_with_tuner(
        &self,
        window_index: usize,
        tuner_id: crate::hardware::pool::TunerId,
        scan: &crate::ecs::ScanEntity,
        context: &SystemContext,
    ) -> Result<WindowTaskComponent> {
        let cancellation_token = CancellationToken::new();
        let cancel_clone = cancellation_token.clone();

        let center_freq = scan.config.freq_min + (window_index as f64 * scan.config.window_size);
        let config = self.config.clone();
        let pool = self.pool.clone();
        let shutdown_coordinator = self.shutdown_coordinator.clone();
        let total_windows = scan.progress.total_windows;
        let candidate_entities = context.candidate_entities.clone();
        let station_entities = context.station_entities.clone();
        let scan_id = *scan.id();

        debug!(
            scan_id = ?scan_id,
            window_index = window_index,
            tuner_id = ?tuner_id,
            center_freq_mhz = center_freq / 1e6,
            "WindowProcessingSystem: Spawning window task with pre-allocated tuner"
        );

        let task_handle = std::thread::spawn(move || {
            debug!(window_index = window_index, tuner_id = ?tuner_id, "Window task started with tuner");

            if cancel_clone.is_cancelled() {
                debug!(
                    window_index = window_index,
                    "Window task cancelled before work"
                );
                return Err(crate::core::types::ScannerError::Custom(
                    "Task cancelled".to_string(),
                ));
            }

            let tuner = match pool.create_tuner_from_allocated(tuner_id.clone()) {
                Some(t) => t,
                None => {
                    debug!(
                        window_index = window_index,
                        tuner_id = ?tuner_id,
                        "Failed to create tuner from allocated tuner_id"
                    );
                    return Err(crate::core::types::ScannerError::Custom(
                        "Tuner not found or not allocated".to_string(),
                    ));
                }
            };

            let mut segment = match crate::hardware::pool::Segment::from_tuner(
                tuner,
                center_freq,
                &config,
                cancel_clone.clone(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    debug!(
                        window_index = window_index,
                        error = ?e,
                        "Failed to create segment from tuner"
                    );
                    return Err(e);
                }
            };

            let window_config = crate::scanning::window::WindowConfig {
                center_freq,
                window_num: window_index,
                total_windows,
                tuner_provider: pool.clone(),
                config: config.clone(),
                shutdown_coordinator: shutdown_coordinator.clone(),
                window_cancellation: Some(cancel_clone.clone()),
                pause_signal: None,
                station_entities,
                candidate_entities,
                scan_id,
            };

            let window = crate::scanning::window::Window::new(window_config);

            let result = window.process(&segment);

            debug!(
                window_index = window_index,
                "Processing complete, stopping stream"
            );
            if let Err(e) = segment.stop_stream() {
                debug!(window_index = window_index, error = ?e, "Error stopping stream");
            }

            match result {
                Ok(()) => {
                    debug!(window_index = window_index, "Window task completed");
                    Ok(WindowTaskResult {
                        window_index,
                        candidates: Vec::new(),
                        completed_at: Instant::now(),
                    })
                }
                Err(e) => {
                    debug!(
                        window_index = window_index,
                        error = ?e,
                        "Window task failed"
                    );
                    Err(e)
                }
            }
        });

        Ok(WindowTaskComponent {
            window_index,
            task_handle,
            cancellation_token,
            started_at: Instant::now(),
            cancelling: false,
        })
    }

    fn process_window_results(
        &self,
        result: &WindowTaskResult,
        scan: &mut crate::ecs::ScanEntity,
        _context: &mut SystemContext,
    ) -> Result<()> {
        debug!(
            scan_id = ?scan.id(),
            window_index = result.window_index,
            candidate_count = result.candidates.len(),
            "WindowProcessingSystem: Processing window results"
        );

        scan.progress.complete_window_at(result.window_index);

        Ok(())
    }

    fn handle_pending_state(&self, scan: &mut crate::ecs::ScanEntity) {
        info!(
            scan_id = ?scan.id(),
            "WindowProcessingSystem: Scan pending, transitioning to Scanning"
        );
        scan.progress.state = ScanPauseState::Scanning;
        scan.lifecycle.start();
    }

    fn handle_scanning_state(
        &mut self,
        scan: &mut crate::ecs::ScanEntity,
        context: &mut SystemContext,
    ) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        for window_index in 0..scan.progress.total_windows {
            let window_id = WindowId::new(*scan.id(), window_index);
            let mut windows = window_entities.write().unwrap();
            if windows.get(&window_id).is_none() {
                windows.insert(WindowEntity::new(window_id));
            }
        }

        let task_to_handle = {
            let mut windows = window_entities.write().unwrap();
            windows
                .iter_mut()
                .find(|w| w.id().scan_id == *scan.id() && w.task.is_some())
                .and_then(|w| w.task.take())
        };

        if let Some(task) = task_to_handle {
            self.handle_window_task(task, scan, context)?;
            return Ok(());
        }

        let allocation_to_handle = {
            let mut windows = window_entities.write().unwrap();
            windows
                .iter_mut()
                .find(|w| w.id().scan_id == *scan.id() && w.allocation.is_allocated())
                .map(|w| {
                    let tuner_id = w.allocation.tuner_id().unwrap().clone();
                    let window_index = w.window_index();
                    w.allocation.clear();
                    (window_index, tuner_id)
                })
        };

        if let Some((window_index, tuner_id)) = allocation_to_handle {
            self.handle_allocated_tuner(window_index, tuner_id, scan, context)?;
            return Ok(());
        }

        let needs_allocation = {
            let windows = window_entities.read().unwrap();
            windows
                .iter()
                .any(|w| w.id().scan_id == *scan.id() && w.allocation.is_none() && w.is_pending())
        };

        if needs_allocation {
            self.handle_no_allocation(scan, &window_entities);
        }

        Ok(())
    }

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

    fn handle_window_task(
        &self,
        task: WindowTaskComponent,
        scan: &mut crate::ecs::ScanEntity,
        context: &mut SystemContext,
    ) -> Result<()> {
        let window_id = WindowId::new(*scan.id(), task.window_index);

        if task.task_handle.is_finished() {
            debug!(
                scan_id = ?scan.id(),
                window_index = task.window_index,
                cancelling = task.cancelling,
                "WindowProcessingSystem: Task finished, extracting results"
            );

            match task.task_handle.join() {
                Ok(Ok(result)) => {
                    if !task.cancelling {
                        self.process_window_results(&result, scan, context)?;

                        if let Some(window_entities) = &context.window_entities {
                            let mut windows = window_entities.write().unwrap();
                            if let Some(window) = windows.get_mut(&window_id) {
                                window.progress.mark_completed();
                            }
                        }

                        if scan.progress.completed_windows.len() >= scan.progress.total_windows {
                            info!(
                                scan_id = ?scan.id(),
                                "WindowProcessingSystem: All windows complete"
                            );
                            scan.progress.state = ScanPauseState::Completed;
                            scan.lifecycle.complete();
                        }
                    } else {
                        debug!(
                            scan_id = ?scan.id(),
                            window_index = task.window_index,
                            "WindowProcessingSystem: Task was cancelled, resetting window to pending"
                        );

                        if let Some(window_entities) = &context.window_entities {
                            let mut windows = window_entities.write().unwrap();
                            if let Some(window) = windows.get_mut(&window_id) {
                                window.progress.reset_to_pending();
                            }
                        }
                    }

                    self.deallocate_window_tuner(&window_id, context);
                }
                Ok(Err(e)) => {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        error = ?e,
                        "WindowProcessingSystem: Task failed, continuing"
                    );

                    if let Some(window_entities) = &context.window_entities {
                        let mut windows = window_entities.write().unwrap();
                        if let Some(window) = windows.get_mut(&window_id) {
                            window.progress.mark_failed();
                        }
                    }

                    self.deallocate_window_tuner(&window_id, context);
                }
                Err(e) => {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        error = ?e,
                        "WindowProcessingSystem: Task panicked, continuing"
                    );

                    if let Some(window_entities) = &context.window_entities {
                        let mut windows = window_entities.write().unwrap();
                        if let Some(window) = windows.get_mut(&window_id) {
                            window.progress.mark_failed();
                        }
                    }

                    self.deallocate_window_tuner(&window_id, context);
                }
            }
        } else if let Some(window_entities) = &context.window_entities {
            let mut windows = window_entities.write().unwrap();
            if let Some(window) = windows.get_mut(&window_id) {
                window.task = Some(task);
            }
        }

        Ok(())
    }

    fn handle_allocated_tuner(
        &mut self,
        window_index: usize,
        tuner_id: crate::hardware::pool::TunerId,
        scan: &mut crate::ecs::ScanEntity,
        context: &SystemContext,
    ) -> Result<()> {
        debug!(
            scan_id = ?scan.id(),
            window_index = window_index,
            tuner_id = ?tuner_id,
            "WindowProcessingSystem: Allocation complete, spawning window task"
        );
        let task = self.spawn_window_task_with_tuner(window_index, tuner_id, scan, context)?;

        let window_id = WindowId::new(*scan.id(), window_index);
        if let Some(window_entities) = &context.window_entities {
            let mut windows = window_entities.write().unwrap();
            if let Some(window) = windows.get_mut(&window_id) {
                window.task = Some(task);
                window.progress.start_processing();
            }
        }
        Ok(())
    }

    fn handle_no_allocation(
        &self,
        scan: &crate::ecs::ScanEntity,
        window_entities: &std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<WindowEntity>>>,
    ) {
        debug!(
            scan_id = ?scan.id(),
            total_windows = scan.progress.total_windows,
            completed_windows = scan.progress.completed_windows.len(),
            "WindowProcessingSystem: Checking for next window"
        );
        if let Some(next_window) = self.next_window_to_process(scan) {
            debug!(
                scan_id = ?scan.id(),
                window_index = next_window,
                "WindowProcessingSystem: Requesting tuner allocation for next window"
            );
            self.request_window_allocation(next_window, scan, window_entities);
        } else {
            debug!(
                scan_id = ?scan.id(),
                "WindowProcessingSystem: No windows to process"
            );
        }
    }

    fn handle_paused_or_listening_state(
        &self,
        scan: &crate::ecs::ScanEntity,
        state_name: &str,
        window_entities: &std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<WindowEntity>>>,
    ) {
        let mut windows = window_entities.write().unwrap();
        for window in windows.iter_mut() {
            if window.id().scan_id != *scan.id() {
                continue;
            }

            if let Some(mut task) = window.task.take() {
                if !task.cancelling {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        "WindowProcessingSystem: Cancelling task ({})", state_name
                    );
                    task.cancellation_token.cancel();
                    task.cancelling = true;
                }

                if task.task_handle.is_finished() {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        "WindowProcessingSystem: Task finished, cleaning up"
                    );
                    let _ = task.task_handle.join();
                } else {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        "WindowProcessingSystem: Task still running, will check again"
                    );
                    window.task = Some(task);
                }
            }

            if !window.allocation.is_none() {
                debug!(
                    scan_id = ?scan.id(),
                    window_index = window.window_index(),
                    "WindowProcessingSystem: Clearing window allocation ({})", state_name
                );
                window.allocation.clear();
            }
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

        let scan_entities = match &context.scan_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        let mut scans = scan_entities.write().unwrap();

        for scan in scans.iter_mut() {
            match scan.progress.state {
                ScanPauseState::Pending => self.handle_pending_state(scan),
                ScanPauseState::Scanning => self.handle_scanning_state(scan, context)?,
                ScanPauseState::PausedAtWindow { .. } => {
                    self.handle_paused_or_listening_state(scan, "paused", &window_entities)
                }
                ScanPauseState::PausedGlobally { .. } => {
                    self.handle_paused_or_listening_state(scan, "globally paused", &window_entities)
                }
                ScanPauseState::Listening { .. } => {
                    self.handle_paused_or_listening_state(scan, "listening", &window_entities)
                }
                ScanPauseState::Completed => {}
            }
        }

        Ok(())
    }
}
