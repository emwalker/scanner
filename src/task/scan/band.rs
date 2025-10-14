//! Band scanning task

use super::context::{LoopControl, ScanContext};
use crate::core::types::{Band, Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::hardware::types::Backend;
use crate::scanner_state::{PauseSignal, ScanMode, ScannerState};
use crate::shutdown::ShutdownCoordinator;
use crate::signal;
use crate::ui::{ProgressReporter, ScannerCommand, TuiEvent};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Band scanning task (coordinator - doesn't hold tuners)
#[allow(dead_code)]
pub struct ScanBandTask {
    config: ScanningConfig,
    band: Band,
    progress_reporter: Arc<dyn ProgressReporter>,
    pause_signal: PauseSignal,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,

    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
}

impl ScanBandTask {
    /// Phase 1 constructor - simple scan without state machine
    #[allow(dead_code)]
    pub fn new(
        config: ScanningConfig,
        band: Band,
        progress_reporter: Arc<dyn ProgressReporter>,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            config,
            band,
            progress_reporter,
            pause_signal: PauseSignal::new(),
            pool,
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
        }
    }

    /// Phase 2 constructor - full state machine with TUI integration
    #[allow(dead_code)]
    pub fn new_full(
        config: ScanningConfig,
        band: Band,
        progress_reporter: Arc<dyn ProgressReporter>,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        command_receiver: Option<Receiver<ScannerCommand>>,
        tui_event_sender: Option<Sender<TuiEvent>>,
    ) -> Self {
        Self {
            config,
            band,
            progress_reporter,
            pause_signal: PauseSignal::new(),
            pool,
            shutdown_coordinator,
            command_receiver,
            tui_event_sender,
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
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<()> {
        if self.command_receiver.is_none() && self.tui_event_sender.is_none() {
            return self.run_simple(shutdown);
        }

        debug!(band = ?self.band, "ScanBandTask starting with state machine");
        signal::clear_processed_frequencies();

        let window_centers = self.band.windows(
            self.config.samp_rate,
            self.config.signal_processing.window_overlap,
        );
        let windows_to_process = self
            .config
            .scanning_windows
            .map(|n| n.min(window_centers.len()))
            .unwrap_or(window_centers.len());

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
            window_index: 0,
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
                LoopControl::Advance => context.window_index += 1,
            }
        }

        debug!(band = ?self.band, "ScanBandTask completed");
        Ok(())
    }

    fn run_simple(&mut self, shutdown: CancellationToken) -> Result<()> {
        debug!(band = ?self.band, "Starting band scan task");

        let window_centers = self.band.windows(
            self.config.samp_rate,
            self.config.signal_processing.window_overlap,
        );

        debug!(window_count = window_centers.len(), "Scanning windows");

        for (idx, center_freq) in window_centers.iter().enumerate() {
            if shutdown.is_cancelled() {
                debug!("Shutdown requested, stopping scan");
                break;
            }

            while self.pause_signal.is_paused() {
                if shutdown.is_cancelled() {
                    debug!("Shutdown requested during pause, stopping scan");
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            debug!(
                window_num = idx + 1,
                total_windows = window_centers.len(),
                center_freq_mhz = center_freq / 1e6,
                "Processing window"
            );

            let window =
                crate::scanning::window::Window::new(crate::scanning::window::WindowConfig {
                    center_freq: *center_freq,
                    window_num: idx + 1,
                    total_windows: window_centers.len(),
                    tuner_provider: self.pool.clone(),
                    config: Arc::new(self.config.clone()),
                    progress_reporter: self.progress_reporter.clone(),
                    shutdown_coordinator: self.shutdown_coordinator.clone(),
                    pause_signal: Some(self.pause_signal.clone()),
                });

            window.process_with_pool()?;
        }

        debug!(band = ?self.band, "Band scan task completed");
        Ok(())
    }

    #[allow(dead_code)]
    pub fn description(&self) -> String {
        let (start, end) = self.band.frequency_range();
        format!(
            "Scanning Band: {:?} ({:.1}-{:.1} MHz)",
            self.band,
            start / 1e6,
            end / 1e6,
        )
    }

    #[allow(dead_code)]
    pub fn on_start(&mut self) {
        debug!(band = ?self.band, "ScanBandTask starting");
    }

    #[allow(dead_code)]
    pub fn on_complete(&mut self) {
        debug!(band = ?self.band, "ScanBandTask completed successfully");
    }

    #[allow(dead_code)]
    pub fn on_error(&mut self, error: &ScannerError) {
        debug!(band = ?self.band, error = ?error, "ScanBandTask encountered error");
    }
}
