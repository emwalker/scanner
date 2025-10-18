//! Window processing system - manages window task lifecycle

use crate::core::types::{Result, ScanningConfig};
use crate::ecs::components::scan::{
    WindowAllocationRequest, WindowTaskComponent, WindowTaskResult,
};
use crate::ecs::system::{System, SystemContext};
use crate::ecs::{CandidateEntity, Entities, Entity, ScanPauseState, StationEntity};
use crate::hardware::pool::{Pool, TaskRequirements, TunerActivity};
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::debug;

pub struct WindowProcessingSystem {
    config: Arc<ScanningConfig>,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    candidate_entities: Option<Entities<CandidateEntity>>,
    station_entities: Option<Entities<StationEntity>>,
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
            candidate_entities: None,
            station_entities: None,
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

    fn request_window_allocation(&self, window_index: usize, scan: &mut crate::ecs::ScanEntity) {
        let center_freq = scan.config.freq_min + (window_index as f64 * scan.config.window_size);

        let requirements = TaskRequirements {
            frequency_hz: center_freq,
            bandwidth_hz: self.config.samp_rate,
            required_sample_rate: self.config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let requester_id = format!("scan_{}_window_{}", scan.id().value(), window_index);

        scan.window_allocation = Some(WindowAllocationRequest::Requested {
            window_index,
            requirements,
            activity: TunerActivity::Scanning,
            requester_id,
        });

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
    ) -> Result<WindowTaskComponent> {
        let cancellation_token = CancellationToken::new();
        let cancel_clone = cancellation_token.clone();

        let center_freq = scan.config.freq_min + (window_index as f64 * scan.config.window_size);
        let config = self.config.clone();
        let pool = self.pool.clone();
        let shutdown_coordinator = self.shutdown_coordinator.clone();
        let total_windows = scan.progress.total_windows;
        let candidate_entities = self.candidate_entities.clone();
        let station_entities = self.station_entities.clone();
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
        debug!(
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
        debug!(
            scan_id = ?scan.id(),
            has_window_task = scan.window_task.is_some(),
            has_window_allocation = scan.window_allocation.is_some(),
            "WindowProcessingSystem: Processing scan in Scanning state"
        );

        if let Some(task) = scan.window_task.take() {
            self.handle_window_task(task, scan, context)?;
        } else if let Some(WindowAllocationRequest::Allocated {
            window_index,
            tuner_id,
            ..
        }) = scan.window_allocation.take()
        {
            self.handle_allocated_tuner(window_index, tuner_id, scan)?;
        } else if scan.window_allocation.is_none() {
            self.handle_no_allocation(scan);
        }

        Ok(())
    }

    fn handle_window_task(
        &self,
        task: WindowTaskComponent,
        scan: &mut crate::ecs::ScanEntity,
        context: &mut SystemContext,
    ) -> Result<()> {
        if task.task_handle.is_finished() {
            debug!(
                scan_id = ?scan.id(),
                window_index = task.window_index,
                "WindowProcessingSystem: Task finished, extracting results"
            );

            match task.task_handle.join() {
                Ok(Ok(result)) => {
                    self.process_window_results(&result, scan, context)?;

                    if scan.progress.completed_windows.len() >= scan.progress.total_windows {
                        debug!(
                            scan_id = ?scan.id(),
                            "WindowProcessingSystem: All windows complete"
                        );
                        scan.progress.state = ScanPauseState::Completed;
                        scan.lifecycle.complete();
                    }
                }
                Ok(Err(e)) => {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        error = ?e,
                        "WindowProcessingSystem: Task failed, continuing"
                    );
                }
                Err(e) => {
                    debug!(
                        scan_id = ?scan.id(),
                        window_index = task.window_index,
                        error = ?e,
                        "WindowProcessingSystem: Task panicked, continuing"
                    );
                }
            }
        } else {
            scan.window_task = Some(task);
        }

        Ok(())
    }

    fn handle_allocated_tuner(
        &mut self,
        window_index: usize,
        tuner_id: crate::hardware::pool::TunerId,
        scan: &mut crate::ecs::ScanEntity,
    ) -> Result<()> {
        debug!(
            scan_id = ?scan.id(),
            window_index = window_index,
            tuner_id = ?tuner_id,
            "WindowProcessingSystem: Allocation complete, spawning window task"
        );
        let task = self.spawn_window_task_with_tuner(window_index, tuner_id, scan)?;
        scan.window_task = Some(task);
        Ok(())
    }

    fn handle_no_allocation(&self, scan: &mut crate::ecs::ScanEntity) {
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
            self.request_window_allocation(next_window, scan);
        } else {
            debug!(
                scan_id = ?scan.id(),
                "WindowProcessingSystem: No windows to process"
            );
        }
    }

    fn handle_paused_or_listening_state(
        &self,
        scan: &mut crate::ecs::ScanEntity,
        state_name: &str,
    ) {
        if let Some(task) = scan.window_task.take() {
            debug!(
                scan_id = ?scan.id(),
                window_index = task.window_index,
                "WindowProcessingSystem: Cancelling task ({})", state_name
            );
            task.cancellation_token.cancel();
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

        if self.candidate_entities.is_none() && context.candidate_entities.is_some() {
            self.candidate_entities = context.candidate_entities.clone();
        }

        if self.station_entities.is_none() && context.station_entities.is_some() {
            self.station_entities = context.station_entities.clone();
        }

        let scan_entities = match &context.scan_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let mut scans = scan_entities.write().unwrap();

        for scan in scans.iter_mut() {
            match scan.progress.state {
                ScanPauseState::Pending => self.handle_pending_state(scan),
                ScanPauseState::Scanning => self.handle_scanning_state(scan, context)?,
                ScanPauseState::PausedAtWindow { .. } => {
                    self.handle_paused_or_listening_state(scan, "paused")
                }
                ScanPauseState::Listening { .. } => {
                    self.handle_paused_or_listening_state(scan, "listening")
                }
                ScanPauseState::Completed => {}
            }
        }

        Ok(())
    }
}
