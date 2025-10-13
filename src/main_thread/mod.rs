mod audio_coordinator;
mod commands;
mod runner;
mod state_manager;
mod window_processing;

use crate::core::types::{ConsoleWriter, Logger, Result, ScanningConfig};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::scanner_state::{PauseSignal, ScannerState};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::{NoOpProgressReporter, ProgressReporter, ScannerCommand, TuiEvent};
use audio_coordinator::TuneParams;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use tracing::{debug, info};

pub struct MainThread {
    config: Arc<ScanningConfig>,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    _backend: Arc<dyn crate::hardware::Backend>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    scanner_state: ScannerState,
    pause_signal: PauseSignal,
    current_playing: Option<TuneParams>,
    pool: Arc<Pool>,
    #[allow(dead_code)]
    discovered_devices: Vec<crate::hardware::DeviceInfo>,
}

impl MainThread {
    pub fn new(
        config: Arc<ScanningConfig>,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        backend: Arc<dyn crate::hardware::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let filter = PoolFilter::new()
            .with_driver("sdrplay")
            .with_mode(TuningMode::SingleTuner);
        let pool = Pool::new(filter, None);

        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            progress_reporter: Arc::new(NoOpProgressReporter),
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            pool: Arc::new(pool),
            discovered_devices: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_progress(
        config: Arc<ScanningConfig>,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        backend: Arc<dyn crate::hardware::Backend>,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        pool: Arc<Pool>,
        discovered_devices: Vec<crate::hardware::DeviceInfo>,
    ) -> Result<Self> {
        let main_thread = MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            progress_reporter,
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            pool,
            discovered_devices,
        };

        Ok(main_thread)
    }

    pub fn with_command_receiver(mut self, receiver: Receiver<ScannerCommand>) -> Self {
        self.command_receiver = Some(receiver);
        self
    }

    pub fn with_tui_event_sender(mut self, sender: Sender<TuiEvent>) -> Self {
        self.tui_event_sender = Some(sender.clone());

        self.pool.add_state_change_callback(Box::new(move |status| {
            let event = TuiEvent::ActiveTunersUpdated { status };
            let _ = sender.send(event);
        }));

        self
    }

    pub fn run(mut self, stations: Option<String>) -> Result<()> {
        // Logging is now initialized in main() before SDR operations
        // Pool is already populated with initial device by scanner.rs

        // Verify pool is populated
        let pool_status = self.pool.status();
        debug!(
            device_count = pool_status.device_count,
            available_tuners = pool_status.available_tuner_count,
            "Pool status at startup"
        );

        self.console_writer.write_info("Scanning for stations ...");

        if let Some(stations_str) = stations {
            self.scan_stations(&stations_str)?;
        } else {
            self.scan_band()?;
        }

        self.console_writer.write_info("Scan complete.");
        Ok(())
    }
}

// Default implementations for production use
pub struct DefaultConsoleWriter;

impl ConsoleWriter for DefaultConsoleWriter {
    fn write_info(&self, message: &str) {
        info!("{}", message);
    }

    fn write_debug(&self, message: &str) {
        debug!("{}", message);
    }
}

impl Drop for MainThread {
    fn drop(&mut self) {
        self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests;
