//! Station scanning task

use super::context::{LoopControl, ScanContext};
use crate::core::types::{Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::hardware::types::Backend;
use crate::scanner_state::{PauseSignal, ScanMode, ScannerState};
use crate::shutdown::ShutdownCoordinator;
use crate::signal;
use crate::task::TaskContinuation;
use crate::ui::{ProgressReporter, ScannerCommand, TuiEvent};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Station scanning task (coordinator - doesn't hold tuners)
#[allow(dead_code)]
pub struct ScanStationsTask {
    config: ScanningConfig,
    stations: Vec<f64>,
    progress_reporter: Arc<dyn ProgressReporter>,
    pause_signal: PauseSignal,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,

    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,

    window_index: usize,
}

impl ScanStationsTask {
    /// Phase 1 constructor - simple scan without state machine
    #[allow(dead_code)]
    pub fn new(
        config: ScanningConfig,
        stations: Vec<f64>,
        progress_reporter: Arc<dyn ProgressReporter>,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            config,
            stations,
            progress_reporter,
            pause_signal: PauseSignal::new(),
            pool,
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            window_index: 0,
        }
    }

    /// Phase 2 constructor - full state machine with TUI integration
    #[allow(dead_code)]
    pub fn new_full(
        config: ScanningConfig,
        stations: Vec<f64>,
        progress_reporter: Arc<dyn ProgressReporter>,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        command_receiver: Option<Receiver<ScannerCommand>>,
        tui_event_sender: Option<Sender<TuiEvent>>,
    ) -> Self {
        Self {
            config,
            stations,
            progress_reporter,
            pause_signal: PauseSignal::new(),
            pool,
            shutdown_coordinator,
            command_receiver,
            tui_event_sender,
            window_index: 0,
        }
    }

    /// Access to pause signal (for external control)
    #[allow(dead_code)]
    pub fn pause_signal(&self) -> &PauseSignal {
        &self.pause_signal
    }

    #[allow(dead_code)]
    pub fn backend(&self) -> Backend {
        Backend::Soapy
    }

    #[allow(dead_code)]
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<TaskContinuation> {
        if self.command_receiver.is_none() && self.tui_event_sender.is_none() {
            return self.run_simple(shutdown);
        }

        if self.window_index == 0 {
            debug!(
                station_count = self.stations.len(),
                "ScanStationsTask starting with state machine"
            );
            signal::clear_processed_frequencies();
        } else {
            debug!(
                window_index = self.window_index,
                total_windows = self.stations.len(),
                "ScanStationsTask resuming from window"
            );
        }

        let window_centers = self.stations.clone();
        let windows_to_process = window_centers.len();

        let mut context = ScanContext {
            config: &self.config,
            pool: &self.pool,
            shutdown_coordinator: &self.shutdown_coordinator,
            progress_reporter: &self.progress_reporter,
            pause_signal: &self.pause_signal,
            command_receiver: &mut self.command_receiver,
            tui_event_sender: &self.tui_event_sender,
            scanner_state: ScannerState::new(),
            current_playing: None,
            audio_session: None,
            window_centers,
            windows_to_process,
            window_index: self.window_index,
        };

        loop {
            if shutdown.is_cancelled() || self.shutdown_coordinator.is_shutdown() {
                context.scanner_state.shutdown();
            }

            let control = match &context.scanner_state.mode {
                ScanMode::ShuttingDown(_) => context.handle_shutting_down_mode(),
                ScanMode::ScanComplete(_) => context.handle_scan_complete_mode(),
                ScanMode::ScanCompletePaused(_) => context.handle_scan_complete_paused_mode(),
                ScanMode::Paused(_) => context.handle_paused_mode(),
                ScanMode::Listening(_) => context.handle_listening_mode(),
                ScanMode::Scanning(_) => context.handle_scanning_mode(),
            }?;

            match control {
                LoopControl::Break => break,
                LoopControl::Continue => continue,
                LoopControl::Advance => {
                    context.window_index += 1;
                    self.window_index = context.window_index;
                    // Yield backend semaphore after each window to allow enumeration
                    if context.window_index < context.windows_to_process {
                        debug!(
                            completed_window = context.window_index,
                            remaining_windows = context.windows_to_process - context.window_index,
                            "Yielding backend semaphore to allow enumeration"
                        );
                        return Ok(TaskContinuation::Resubmit);
                    }
                }
                LoopControl::ResubmitAfter(delay) => {
                    debug!(
                        delay_ms = delay.as_millis(),
                        "ScanStationsTask yielding with delay - will resubmit after delay"
                    );
                    return Ok(TaskContinuation::ResubmitAfter(delay));
                }
            }
        }

        debug!(
            station_count = self.stations.len(),
            "ScanStationsTask completed"
        );
        Ok(TaskContinuation::Complete)
    }

    fn run_simple(&mut self, shutdown: CancellationToken) -> Result<TaskContinuation> {
        debug!(
            station_count = self.stations.len(),
            "Starting station scan task"
        );

        for (idx, station_freq) in self.stations.iter().enumerate() {
            if shutdown.is_cancelled() {
                debug!("Shutdown requested, stopping scan");
                break;
            }

            while self.pause_signal.is_paused() {
                if shutdown.is_cancelled() {
                    debug!("Shutdown requested during pause, stopping scan");
                    return Ok(TaskContinuation::Complete);
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            debug!(
                station_num = idx + 1,
                total_stations = self.stations.len(),
                station_freq_mhz = station_freq / 1e6,
                "Processing station"
            );

            let window = crate::scanning::window::Window::for_station(
                *station_freq,
                idx + 1,
                self.stations.len(),
                self.pool.clone(),
                Arc::new(self.config.clone()),
                self.progress_reporter.clone(),
                self.shutdown_coordinator.clone(),
            );

            window.process_with_pool()?;
        }

        debug!(
            station_count = self.stations.len(),
            "Station scan task completed"
        );
        Ok(TaskContinuation::Complete)
    }

    #[allow(dead_code)]
    pub fn description(&self) -> String {
        if self.stations.len() <= 3 {
            let freqs: Vec<String> = self
                .stations
                .iter()
                .map(|f| format!("{:.1}", f / 1e6))
                .collect();
            format!("Scanning Stations: {} MHz", freqs.join(", "))
        } else {
            format!("Scanning {} Stations", self.stations.len())
        }
    }

    #[allow(dead_code)]
    pub fn on_start(&mut self) {
        debug!(
            station_count = self.stations.len(),
            "ScanStationsTask starting"
        );
    }

    #[allow(dead_code)]
    pub fn on_complete(&mut self) {
        debug!(
            station_count = self.stations.len(),
            "ScanStationsTask completed successfully"
        );
    }

    #[allow(dead_code)]
    pub fn on_error(&mut self, error: &ScannerError) {
        debug!(station_count = self.stations.len(), error = ?error, "ScanStationsTask encountered error");
    }
}
