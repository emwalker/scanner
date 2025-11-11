use std::{
    sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Sender},
    thread::{JoinHandle, Thread},
};

use tracing::{debug, info};

use crate::{
    core::types::{Result, ScanningConfig},
    ecs::{
        AudioEntity, DeviceEntity, Entities, EntityWorld, SignalEntity, TunerEntity, WindowEntity,
        resources::LocationResource,
    },
    hardware::pool::{Pool, PoolFilter, TuningMode},
    shutdown::ShutdownCoordinator,
    task::TaskScheduler,
    ui::TuiEvent,
};

pub struct MainThread {
    config: Arc<ScanningConfig>,
    _backend: Arc<dyn crate::hardware::Backend>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    pool: Arc<Pool>,
    scheduler: Arc<TaskScheduler>,
    #[allow(dead_code)]
    discovered_devices: Vec<crate::hardware::DeviceInfo>,

    task_entities: Entities<crate::ecs::TaskEntity>,
    window_entities: Entities<WindowEntity>,
    audio_entities: Entities<AudioEntity>,
    signal_entities: Entities<SignalEntity>,

    coordinator_handle: Option<JoinHandle<()>>,
    coordinator_shutdown: Arc<AtomicBool>,
    coordinator_thread: Arc<Mutex<Option<Thread>>>,
    pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
    global_pause_resource: crate::ecs::GlobalPauseResource,
    pending_scan_request: Arc<RwLock<Option<crate::ecs::components::scan::PendingScanRequest>>>,
    discovery_rx: Option<std::sync::mpsc::Receiver<crate::discovery::Event>>,
    location_resource: LocationResource,
}

impl MainThread {
    pub fn new(
        config: Arc<ScanningConfig>,
        backend: Arc<dyn crate::hardware::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let filter = PoolFilter::new()
            .with_driver("sdrplay")
            .with_mode(TuningMode::SingleTuner);

        let tuner_entities = Arc::new(Mutex::new(EntityWorld::<TunerEntity>::new()));
        let device_entities = Arc::new(Mutex::new(EntityWorld::<DeviceEntity>::new()));

        let pool = Arc::new(Pool::with_entity_worlds(
            filter,
            None,
            tuner_entities,
            device_entities,
        ));
        let scheduler = Arc::new(TaskScheduler::new(
            pool.clone(),
            shutdown_coordinator.clone(),
        ));

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let pause_request_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<
            crate::ecs::PauseAndTuneRequest,
        >::new()));

        let global_pause_resource =
            Arc::new(std::sync::Mutex::new(crate::ecs::GlobalPauseState::Active));

        let pending_scan_request = Arc::new(RwLock::new(None));
        let (_discovery_tx, discovery_rx) = std::sync::mpsc::channel();

        // Create a dummy LocationResource for testing
        let location_resource = crate::ecs::resources::new_location_resource();

        let main_thread = MainThread {
            config,
            _backend: backend,
            shutdown_coordinator,
            tui_event_sender: None,
            pool,
            scheduler,
            discovered_devices: Vec::new(),
            task_entities,
            window_entities,
            audio_entities,
            signal_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
            coordinator_thread: Arc::new(Mutex::new(None)),
            pause_request_queue,
            global_pause_resource,
            pending_scan_request,
            discovery_rx: Some(discovery_rx),
            location_resource,
        };

        Ok(main_thread)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_entities(
        config: Arc<ScanningConfig>,
        backend: Arc<dyn crate::hardware::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        pool: Arc<Pool>,
        scheduler: Arc<TaskScheduler>,
        discovered_devices: Vec<crate::hardware::DeviceInfo>,
        task_entities: Entities<crate::ecs::TaskEntity>,
        window_entities: Entities<WindowEntity>,
        audio_entities: Entities<AudioEntity>,
        signal_entities: Entities<SignalEntity>,
        pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
        global_pause_resource: crate::ecs::GlobalPauseResource,
        pending_scan_request: Arc<RwLock<Option<crate::ecs::components::scan::PendingScanRequest>>>,
        discovery_rx: std::sync::mpsc::Receiver<crate::discovery::Event>,
        location_resource: LocationResource,
    ) -> Result<Self> {
        let main_thread = MainThread {
            config,
            _backend: backend,
            shutdown_coordinator,
            tui_event_sender: None,
            pool,
            scheduler,
            discovered_devices,
            task_entities,
            window_entities,
            audio_entities,
            signal_entities,
            coordinator_handle: None,
            coordinator_shutdown: Arc::new(AtomicBool::new(false)),
            coordinator_thread: Arc::new(Mutex::new(None)),
            pause_request_queue,
            global_pause_resource,
            pending_scan_request,
            discovery_rx: Some(discovery_rx),
            location_resource,
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
        use std::sync::atomic::Ordering;

        use crate::ecs::Coordinator;

        let config = Arc::clone(&self.config);
        let pool = Arc::clone(&self.pool);
        let _scheduler = Arc::clone(&self.scheduler);
        let task_entities_clone = Arc::clone(&self.task_entities);
        let task_entities = Arc::clone(&self.task_entities);
        let window_entities = Arc::clone(&self.window_entities);
        // StationEntity removed during migration - no longer needed
        let audio_entities = Arc::clone(&self.audio_entities);
        let signal_entities = Arc::clone(&self.signal_entities);
        let shutdown = Arc::clone(&self.coordinator_shutdown);
        let shutdown_coordinator = Arc::clone(&self.shutdown_coordinator);
        let tui_event_sender = self.tui_event_sender.clone();
        let thread_handle = Arc::clone(&self.coordinator_thread);
        let pending_scan_request = self.pending_scan_request.clone();
        let discovery_rx = self.discovery_rx.take();

        let pause_request_queue = self.pause_request_queue.clone();
        let global_pause_resource = self.global_pause_resource.clone();
        let location_resource = self.location_resource.clone();

        let handle = std::thread::spawn(move || {
            if let Ok(mut guard) = thread_handle.lock() {
                *guard = Some(std::thread::current());
            }

            let mut coordinator = Coordinator::new(&pool, &config, &shutdown_coordinator)
                .with_task_entities(task_entities)
                .with_window_entities(window_entities)
                .with_audio_entities(audio_entities)
                .with_signal_entities(signal_entities)
                .with_pause_request_queue(pause_request_queue)
                .with_global_pause_resource(global_pause_resource)
                .with_location_resource(location_resource);

            coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));

            if let Some(discovery_rx) = discovery_rx {
                coordinator.add_system(Box::new(
                    crate::ecs::systems::scan::ScanFactorySystem::new(
                        task_entities_clone.clone(),
                        discovery_rx,
                        pool.clone(),
                        pending_scan_request,
                    ),
                ));
            }

            coordinator.add_system(Box::new(crate::ecs::systems::ScanCoordinationSystem::new()));
            coordinator.add_system(Box::new(
                crate::ecs::systems::ScanRequestProcessorSystem::new(),
            ));
            coordinator.add_system(Box::new(crate::ecs::systems::TuneTransitionSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::TaskCoordinationSystem));

            let mut window_processing_system = crate::ecs::systems::WindowProcessingSystem::new(
                config.clone(),
                pool.clone(),
                shutdown_coordinator.clone(),
            );
            window_processing_system.enable();
            coordinator.add_system(Box::new(window_processing_system));

            coordinator.add_system(Box::new(crate::ecs::systems::WindowWorkerSpawnSystem::new(
                config.clone(),
                pool.clone(),
                shutdown_coordinator.clone(),
            )));
            coordinator.add_system(Box::new(
                crate::ecs::systems::WindowWorkerCompletionSystem::new(),
            ));

            coordinator.add_system(Box::new(crate::ecs::systems::PeakDetectionSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::PeakCompletionSystem::new()));

            coordinator.add_system(Box::new(
                crate::ecs::systems::SignalAnalysisSpawnSystem::new(),
            ));
            coordinator.add_system(Box::new(
                crate::ecs::systems::scan::AudioStreamManagementSystem::new(),
            ));
            coordinator.add_system(Box::new(crate::ecs::systems::WindowTimeoutSystem::new()));

            let analyzer = std::sync::Arc::new(crate::audio::quality::AudioAnalyzer::new(
                Box::new(crate::audio::quality::heuristic2::Classifier::new(
                    config.audio.sample_rate as f32,
                )),
            ));
            coordinator.add_system(Box::new(crate::ecs::systems::SignalAnalysisSystem::new(
                analyzer,
            )));
            coordinator.add_system(Box::new(crate::ecs::systems::PeakAnalysisSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::AudioSpawnSystem::new()));

            coordinator.add_system(Box::new(crate::ecs::systems::AudioPlaybackSystem::new()));
            coordinator.add_system(Box::new(crate::ecs::systems::AudioCoordinationSystem::new()));
            coordinator.add_system(Box::new(
                crate::ecs::systems::ManagementSystem::new().with_max_duration(config.duration),
            ));

            let mut ui_update_system = crate::ecs::systems::UIUpdateSystem::new();
            if let Some(sender) = tui_event_sender {
                ui_update_system = ui_update_system.with_tui_event_sender(sender);
            }
            coordinator.add_system(Box::new(ui_update_system));

            info!(
                system_count = coordinator.system_count(),
                "Coordinator thread starting"
            );

            while !shutdown.load(Ordering::SeqCst) && !shutdown_coordinator.is_shutdown() {
                if let Err(e) = coordinator.tick() {
                    debug!(error = ?e, "Coordinator tick failed");
                }

                std::thread::park_timeout(std::time::Duration::from_millis(100));
            }

            info!("Coordinator thread shutting down");
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

        info!("Scanning for stations ...");

        let is_interactive = self.tui_event_sender.is_some();

        while !self.shutdown_coordinator.is_shutdown() {
            if let Ok(entities) = self.task_entities.read()
                && entities.iter().all(|t| t.state.is_completed())
                && !entities.is_empty()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        if is_interactive {
            // In interactive mode, scan completion is just a milestone
            // Keep coordinator running to handle user interactions
            info!("Scan complete. Coordinator remains active for user interactions.");
            while !self.shutdown_coordinator.is_shutdown() {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        } else {
            // In non-interactive mode, exit when scan completes
            info!("Scan complete.");
        }

        Ok(())
    }
}

impl Drop for MainThread {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        info!("MainThread shutting down");

        self.coordinator_shutdown.store(true, Ordering::SeqCst);

        if let Ok(guard) = self.coordinator_thread.lock()
            && let Some(thread) = guard.as_ref()
        {
            thread.unpark();
        }

        if let Some(handle) = self.coordinator_handle.take()
            && let Err(e) = handle.join()
        {
            tracing::error!(error = ?e, "Coordinator thread panicked");
        }

        self.shutdown_coordinator.shutdown();
        self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests;
