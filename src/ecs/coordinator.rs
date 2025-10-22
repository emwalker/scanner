//! ECS Coordinator - executes systems in the proper order each tick

use crate::core::types::{Result, ScanningConfig};
use crate::ecs::EntityWorld;
use crate::ecs::Scheduler;
use crate::ecs::queue::{PauseRequestQueue, TunerRequestQueue};
use crate::ecs::system::SystemContext;
use crate::ecs::{
    AudioEntity, AudioId, CandidateEntity, Entities, ScanEntity, StationEntity, TunerEntity,
    WindowEntity,
};
use crate::hardware::pool::{Pool, Segment};
use crate::shutdown::ShutdownCoordinator;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Type alias for ECS resources (thread-safe collections)
pub type Resource<T> = Arc<Mutex<T>>;

/// Coordinator manages system execution and provides the update loop
///
/// The coordinator:
/// - Maintains references to all entity worlds
/// - Maintains ECS resources (audio streams, segments)
/// - Executes systems in the correct order each tick
/// - Provides a clean interface for the main loop to interact with ECS
pub struct Coordinator {
    scheduler: Scheduler,
    tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>,
    scan_entities: Option<Entities<ScanEntity>>,
    window_entities: Option<Entities<WindowEntity>>,
    station_entities: Option<Entities<StationEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    candidate_entities: Option<Entities<CandidateEntity>>,

    audio_streams: Resource<HashMap<AudioId, cpal::Stream>>,
    audio_segments: Resource<HashMap<AudioId, Segment>>,
    tuner_request_queue: Resource<TunerRequestQueue>,
    pause_request_queue: Resource<PauseRequestQueue>,
    global_pause_resource: crate::ecs::GlobalPauseResource,

    pool: Arc<Pool>,
    config: Arc<ScanningConfig>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

impl Coordinator {
    /// Create a new coordinator with a pool (which contains tuner entities)
    pub fn new(
        pool: &Arc<Pool>,
        config: &Arc<ScanningConfig>,
        shutdown_coordinator: &Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            scheduler: Scheduler::new(),
            tuner_entities: Arc::clone(&pool.tuner_entities),
            scan_entities: None,
            window_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
            #[allow(clippy::arc_with_non_send_sync)]
            audio_streams: Arc::new(Mutex::new(HashMap::new())),
            #[allow(clippy::arc_with_non_send_sync)]
            audio_segments: Arc::new(Mutex::new(HashMap::new())),
            tuner_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            pause_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            global_pause_resource: Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active)),
            pool: Arc::clone(pool),
            config: Arc::clone(config),
            shutdown_coordinator: Arc::clone(shutdown_coordinator),
        }
    }

    /// Add scan entities to the coordinator
    pub fn with_scan_entities(mut self, entities: Entities<ScanEntity>) -> Self {
        self.scan_entities = Some(entities);
        self
    }

    /// Add window entities to the coordinator
    pub fn with_window_entities(mut self, entities: Entities<WindowEntity>) -> Self {
        self.window_entities = Some(entities);
        self
    }

    /// Add station entities to the coordinator
    pub fn with_station_entities(mut self, entities: Entities<StationEntity>) -> Self {
        self.station_entities = Some(entities);
        self
    }

    /// Add audio entities to the coordinator
    pub fn with_audio_entities(mut self, entities: Entities<AudioEntity>) -> Self {
        self.audio_entities = Some(entities);
        self
    }

    /// Add candidate entities to the coordinator
    pub fn with_candidate_entities(mut self, entities: Entities<CandidateEntity>) -> Self {
        self.candidate_entities = Some(entities);
        self
    }

    /// Set the pause request queue (replaces the default empty queue)
    pub fn with_pause_request_queue(mut self, queue: Resource<PauseRequestQueue>) -> Self {
        self.pause_request_queue = queue;
        self
    }

    /// Set the global pause resource (replaces the default Active state)
    pub fn with_global_pause_resource(mut self, resource: crate::ecs::GlobalPauseResource) -> Self {
        self.global_pause_resource = resource;
        self
    }

    /// Get a clone of the global pause resource for external access (e.g., TUI)
    pub fn global_pause_resource(&self) -> crate::ecs::GlobalPauseResource {
        Arc::clone(&self.global_pause_resource)
    }

    /// Add a system to the execution schedule
    pub fn add_system(&mut self, system: Box<dyn crate::ecs::system::System>) {
        self.scheduler.add_system(system);
    }

    /// Execute one tick of the update loop
    ///
    /// This runs all registered systems in order:
    /// 1. DeviceDiscoverySystem - Handle device connect/disconnect
    /// 2. InputHandlingSystem - Process user commands (future)
    /// 3. AllocationSystem - Allocate tuners based on priorities
    /// 4. CoordinationSystem - Coordinate scan operations
    /// 5. ManagementSystem - Manage audio playback
    /// 6. UIUpdateSystem - Update TUI model (future)
    pub fn tick(&mut self) -> Result<()> {
        let mut context = SystemContext::new()
            .with_tuner_entities(Arc::clone(&self.tuner_entities))
            .with_audio_streams(Arc::clone(&self.audio_streams))
            .with_audio_segments(Arc::clone(&self.audio_segments))
            .with_tuner_request_queue(Arc::clone(&self.tuner_request_queue))
            .with_pause_request_queue(Arc::clone(&self.pause_request_queue))
            .with_global_pause_resource(Arc::clone(&self.global_pause_resource))
            .with_pool(Arc::clone(&self.pool))
            .with_config(Arc::clone(&self.config))
            .with_shutdown_coordinator(Arc::clone(&self.shutdown_coordinator));

        if let Some(ref scan_entities) = self.scan_entities {
            context = context.with_scan_entities(Arc::clone(scan_entities));
        }

        if let Some(ref window_entities) = self.window_entities {
            context = context.with_window_entities(Arc::clone(window_entities));
        }

        if let Some(ref station_entities) = self.station_entities {
            context = context.with_station_entities(Arc::clone(station_entities));
        }

        if let Some(ref audio_entities) = self.audio_entities {
            context = context.with_audio_entities(Arc::clone(audio_entities));
        }

        if let Some(ref candidate_entities) = self.candidate_entities {
            context = context.with_candidate_entities(Arc::clone(candidate_entities));
        }

        self.scheduler.run(&mut context)
    }

    /// Get the number of registered systems
    pub fn system_count(&self) -> usize {
        self.scheduler.system_count()
    }

    /// Get a clone of the pause request queue for external access (e.g., TUI)
    pub fn pause_request_queue(&self) -> Resource<PauseRequestQueue> {
        Arc::clone(&self.pause_request_queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::pool::PoolFilter;
    use std::sync::RwLock;

    #[test]
    fn test_coordinator_creation() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let coordinator = Coordinator::new(&pool, &config, &shutdown);

        assert_eq!(coordinator.system_count(), 0);
        assert!(coordinator.scan_entities.is_none());
        assert!(coordinator.station_entities.is_none());
        assert!(coordinator.audio_entities.is_none());
    }

    #[test]
    fn test_coordinator_with_entities() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let coordinator = Coordinator::new(&pool, &config, &shutdown)
            .with_scan_entities(scan_entities)
            .with_station_entities(station_entities)
            .with_audio_entities(audio_entities);

        assert!(coordinator.scan_entities.is_some());
        assert!(coordinator.station_entities.is_some());
        assert!(coordinator.audio_entities.is_some());
    }

    #[test]
    fn test_coordinator_tick_with_no_systems() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let mut coordinator = Coordinator::new(&pool, &config, &shutdown);

        let result = coordinator.tick();
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordinator_add_system() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let mut coordinator = Coordinator::new(&pool, &config, &shutdown);

        coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
        assert_eq!(coordinator.system_count(), 1);

        coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));
        assert_eq!(coordinator.system_count(), 2);
    }

    #[test]
    fn test_coordinator_tick_with_systems() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let mut coordinator = Coordinator::new(&pool, &config, &shutdown);

        coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
        coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));

        let result = coordinator.tick();
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordinator_with_global_pause_resource() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());

        let global_pause = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
        let coordinator =
            Coordinator::new(&pool, &config, &shutdown).with_global_pause_resource(global_pause);

        let retrieved = coordinator.global_pause_resource();
        let state = retrieved.lock().unwrap();
        assert!(matches!(*state, crate::ecs::GlobalPauseState::Active));
    }

    #[test]
    fn test_audio_resources_persist_across_ticks() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let mut coordinator = Coordinator::new(&pool, &config, &shutdown);

        coordinator.tick().unwrap();
        {
            let streams = coordinator.audio_streams.lock().unwrap();
            let segments = coordinator.audio_segments.lock().unwrap();
            assert_eq!(streams.len(), 0, "Resources should start empty");
            assert_eq!(segments.len(), 0, "Resources should start empty");
        }

        coordinator.tick().unwrap();
        {
            let streams = coordinator.audio_streams.lock().unwrap();
            let segments = coordinator.audio_segments.lock().unwrap();
            assert_eq!(
                streams.len(),
                0,
                "Resources should still be empty after second tick"
            );
            assert_eq!(
                segments.len(),
                0,
                "Resources should still be empty after second tick"
            );
        }

        coordinator.tick().unwrap();
        {
            let streams = coordinator.audio_streams.lock().unwrap();
            let segments = coordinator.audio_segments.lock().unwrap();
            assert_eq!(
                streams.len(),
                0,
                "Empty resources should remain empty across ticks"
            );
            assert_eq!(
                segments.len(),
                0,
                "Empty resources should remain empty across ticks"
            );
        }
    }
}
