pub(crate) mod audio_coordinator;
mod runner;

use crate::core::types::{ConsoleWriter, Logger, Result, ScanningConfig};
use crate::ecs::{ScanId, WorkerCommand, WorkerEvent};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::{NoOpProgressReporter, ProgressReporter, ScannerCommand, TuiEvent};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tracing::{debug, info};

pub struct WorkerChannels {
    pub event_rx: Receiver<WorkerEvent>,
    pub command_tx: Sender<WorkerCommand>,
}

pub struct WorkerHandle {
    pub event_tx: Sender<WorkerEvent>,
    pub command_rx: Receiver<WorkerCommand>,
}

impl WorkerChannels {
    pub fn new() -> (Self, WorkerHandle) {
        let (event_tx, event_rx) = channel();
        let (command_tx, command_rx) = channel();

        let channels = Self {
            event_rx,
            command_tx,
        };

        let handle = WorkerHandle {
            event_tx,
            command_rx,
        };

        (channels, handle)
    }
}

pub struct MainThread {
    config: Arc<ScanningConfig>,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    _backend: Arc<dyn crate::hardware::Backend>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    pool: Arc<Pool>,
    scheduler: Arc<TaskScheduler>,
    #[allow(dead_code)]
    discovered_devices: Vec<crate::hardware::DeviceInfo>,

    #[allow(dead_code)]
    coordinator_handle: Option<JoinHandle<()>>,
    #[allow(dead_code)]
    coordinator_shutdown: Arc<AtomicBool>,
    #[allow(dead_code)]
    worker_channels: Arc<Mutex<HashMap<ScanId, WorkerChannels>>>,
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
        let pool = Arc::new(Pool::new(filter, None));
        let scheduler = Arc::new(TaskScheduler::new(
            pool.clone(),
            shutdown_coordinator.clone(),
        ));

        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            progress_reporter: Arc::new(NoOpProgressReporter),
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            pool,
            scheduler,
            discovered_devices: Vec::new(),
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
            worker_channels: Arc::new(Mutex::new(HashMap::new())),
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
        scheduler: Arc<TaskScheduler>,
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
            pool,
            scheduler,
            discovered_devices,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
            worker_channels: Arc::new(Mutex::new(HashMap::new())),
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

    fn spawn_coordinator(&mut self) {
        use crate::ecs::Coordinator;
        use std::sync::atomic::Ordering;

        let coordinator = Coordinator::new(&self.pool);
        let worker_channels = Arc::clone(&self.worker_channels);
        let shutdown = Arc::clone(&self.coordinator_shutdown);
        let shutdown_coordinator = Arc::clone(&self.shutdown_coordinator);

        let handle = std::thread::spawn(move || {
            let mut coordinator = coordinator;
            let tick_interval = std::time::Duration::from_millis(100);

            loop {
                if shutdown.load(Ordering::SeqCst) || shutdown_coordinator.is_shutdown() {
                    debug!("Coordinator shutting down");
                    break;
                }

                if let Ok(channels) = worker_channels.try_lock() {
                    for (scan_id, worker_channel) in channels.iter() {
                        if shutdown.load(Ordering::SeqCst) || shutdown_coordinator.is_shutdown() {
                            break;
                        }

                        while let Ok(event) = worker_channel.event_rx.try_recv() {
                            debug!(scan_id = ?scan_id, event = ?event, "Coordinator received event");

                            let command = Self::decide_command(&event);

                            if let Err(e) = worker_channel.command_tx.send(command.clone()) {
                                debug!(
                                    scan_id = ?scan_id,
                                    error = ?e,
                                    "Failed to send command to worker"
                                );
                            } else {
                                debug!(scan_id = ?scan_id, command = ?command, "Sent command to worker");
                            }
                        }
                    }
                }

                if !shutdown.load(Ordering::SeqCst)
                    && !shutdown_coordinator.is_shutdown()
                    && let Err(e) = coordinator.tick()
                {
                    debug!(error = ?e, "Coordinator tick error");
                }

                std::thread::sleep(tick_interval);
            }

            debug!("Coordinator thread exited");
        });

        self.coordinator_handle = Some(handle);
    }

    fn decide_command(event: &WorkerEvent) -> WorkerCommand {
        use WorkerEvent::*;

        match event {
            ScanStarted { .. } => WorkerCommand::ProcessNextWindow { window_num: 0 },
            WindowCompleted { window_num, .. } => WorkerCommand::ProcessNextWindow {
                window_num: window_num + 1,
            },
            ScanPaused { .. } => WorkerCommand::ResumeScan,
            ScanResumed { .. } => WorkerCommand::ProcessNextWindow { window_num: 0 },
            TunerAllocated { .. } => WorkerCommand::ProcessNextWindow { window_num: 0 },
            TunerReleased { .. } => WorkerCommand::ProcessNextWindow { window_num: 0 },
            StationDiscovered { .. } => WorkerCommand::ProcessNextWindow { window_num: 0 },
        }
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

        self.spawn_coordinator();

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
        use std::sync::atomic::Ordering;

        self.coordinator_shutdown.store(true, Ordering::SeqCst);

        if let Some(handle) = self.coordinator_handle.take() {
            let _ = handle.join();
        }

        self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests;
