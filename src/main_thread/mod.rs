pub(crate) mod audio_coordinator;

use crate::core::types::{ConsoleWriter, Logger, Result, ScanningConfig};
use crate::ecs::{AudioEntity, CandidateEntity, Entities, EntityWorld, ScanEntity, StationEntity};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::TuiEvent;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use tracing::{debug, info};

pub struct MainThread {
    config: Arc<ScanningConfig>,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    _backend: Arc<dyn crate::hardware::Backend>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    pool: Arc<Pool>,
    scheduler: Arc<TaskScheduler>,
    #[allow(dead_code)]
    discovered_devices: Vec<crate::hardware::DeviceInfo>,

    scan_entities: Entities<ScanEntity>,
    station_entities: Entities<StationEntity>,
    audio_entities: Entities<AudioEntity>,
    candidate_entities: Entities<CandidateEntity>,

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
        let candidate_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let main_thread = MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            shutdown_coordinator,
            tui_event_sender: None,
            pool,
            scheduler,
            discovered_devices: Vec::new(),
            scan_entities,
            station_entities,
            audio_entities,
            candidate_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
        };

        Ok(main_thread)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_entities(
        config: Arc<ScanningConfig>,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        backend: Arc<dyn crate::hardware::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        pool: Arc<Pool>,
        scheduler: Arc<TaskScheduler>,
        discovered_devices: Vec<crate::hardware::DeviceInfo>,
        scan_entities: Entities<ScanEntity>,
        station_entities: Entities<StationEntity>,
        audio_entities: Entities<AudioEntity>,
        candidate_entities: Entities<CandidateEntity>,
    ) -> Result<Self> {
        let main_thread = MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            shutdown_coordinator,
            tui_event_sender: None,
            pool,
            scheduler,
            discovered_devices,
            scan_entities,
            station_entities,
            audio_entities,
            candidate_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
        };

        Ok(main_thread)
    }

    pub fn with_tui_event_sender(mut self, sender: Sender<TuiEvent>) -> Self {
        self.tui_event_sender = Some(sender);
        self.spawn_coordinator();
        self
    }

    pub fn start(mut self) -> Self {
        if self.coordinator_handle.is_none() {
            self.spawn_coordinator();
        }
        self
    }

    fn spawn_coordinator(&mut self) {
        use crate::ecs::Coordinator;
        use std::sync::atomic::Ordering;

        let config = Arc::clone(&self.config);
        let pool = Arc::clone(&self.pool);
        let _scheduler = Arc::clone(&self.scheduler);
        let scan_entities = Arc::clone(&self.scan_entities);
        let station_entities = Arc::clone(&self.station_entities);
        let audio_entities = Arc::clone(&self.audio_entities);
        let candidate_entities = Arc::clone(&self.candidate_entities);
        let shutdown = Arc::clone(&self.coordinator_shutdown);
        let shutdown_coordinator = Arc::clone(&self.shutdown_coordinator);
        let tui_event_sender = self.tui_event_sender.clone();

        let handle = std::thread::spawn(move || {
            let mut coordinator = Coordinator::new(&pool)
                .with_scan_entities(scan_entities)
                .with_station_entities(station_entities)
                .with_audio_entities(audio_entities)
                .with_candidate_entities(candidate_entities);

            coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));

            let mut window_processing_system = crate::ecs::systems::WindowProcessingSystem::new(
                config.clone(),
                pool.clone(),
                shutdown_coordinator.clone(),
            );
            window_processing_system.enable();
            coordinator.add_system(Box::new(window_processing_system));

            coordinator.add_system(Box::new(crate::ecs::systems::ScanCoordinationSystem::new()));
            coordinator.add_system(Box::new(
                crate::ecs::systems::ScanRequestProcessorSystem::new(),
            ));
            coordinator.add_system(Box::new(crate::ecs::systems::AudioCoordinationSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::ManagementSystem::new()));

            let mut ui_update_system = crate::ecs::systems::UIUpdateSystem::new();
            if let Some(sender) = tui_event_sender {
                ui_update_system = ui_update_system.with_tui_event_sender(sender);
            }
            coordinator.add_system(Box::new(ui_update_system));

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

    pub fn run(self, _stations: Option<String>) -> Result<()> {
        let pool_status = self.pool.status();
        debug!(
            device_count = pool_status.device_count,
            available_tuners = pool_status.available_tuner_count,
            "Pool status at startup"
        );

        self.console_writer.write_info("Scanning for stations ...");

        while !self.shutdown_coordinator.is_shutdown() {
            if let Ok(entities) = self.scan_entities.read()
                && entities.iter().all(|s| s.is_completed())
                && !entities.is_empty()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
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
