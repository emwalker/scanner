pub(crate) mod audio_coordinator;
mod runner;

use crate::core::types::{ConsoleWriter, Logger, Result, ScanningConfig};
use crate::ecs::{AudioEntity, EntityWorld, ScanEntity, StationEntity};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::{NoOpProgressReporter, ProgressReporter, ScannerCommand, TuiEvent};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
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
    pool: Arc<Pool>,
    scheduler: Arc<TaskScheduler>,
    #[allow(dead_code)]
    discovered_devices: Vec<crate::hardware::DeviceInfo>,

    scan_entities: Arc<RwLock<EntityWorld<ScanEntity>>>,
    station_entities: Arc<RwLock<EntityWorld<StationEntity>>>,
    audio_entities: Arc<RwLock<EntityWorld<AudioEntity>>>,

    coordinator_handle: Option<JoinHandle<()>>,
    coordinator_shutdown: Arc<AtomicBool>,
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

        let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut main_thread = MainThread {
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
            scan_entities,
            station_entities,
            audio_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
        };

        main_thread.spawn_coordinator();

        Ok(main_thread)
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
        let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut main_thread = MainThread {
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
            scan_entities,
            station_entities,
            audio_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
        };

        main_thread.spawn_coordinator();

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

        let pool = Arc::clone(&self.pool);
        let scan_entities = Arc::clone(&self.scan_entities);
        let station_entities = Arc::clone(&self.station_entities);
        let audio_entities = Arc::clone(&self.audio_entities);
        let shutdown = Arc::clone(&self.coordinator_shutdown);
        let shutdown_coordinator = Arc::clone(&self.shutdown_coordinator);

        let handle = std::thread::spawn(move || {
            let mut coordinator = Coordinator::new(&pool)
                .with_scan_entities(scan_entities)
                .with_station_entities(station_entities)
                .with_audio_entities(audio_entities);

            coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::CoordinationSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::ManagementSystem::new()));

            debug!(
                system_count = coordinator.system_count(),
                "Coordinator thread starting"
            );

            while !shutdown.load(Ordering::SeqCst) && !shutdown_coordinator.is_shutdown() {
                if let Err(e) = coordinator.tick() {
                    debug!(error = ?e, "Coordinator tick failed");
                }

                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            debug!("Coordinator thread shutting down");
        });

        self.coordinator_handle = Some(handle);
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

        debug!("MainThread shutting down");

        self.coordinator_shutdown.store(true, Ordering::SeqCst);

        if let Some(handle) = self.coordinator_handle.take()
            && let Err(e) = handle.join()
        {
            tracing::error!(error = ?e, "Coordinator thread panicked");
        }

        self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests;
