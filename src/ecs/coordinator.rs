//! ECS Coordinator - executes systems in the proper order each tick

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use crate::{
    core::types::{Result, ScanningConfig},
    ecs::{
        AudioEntity, AudioId, Entities, EntityWorld, Scheduler, SignalEntity, TaskEntity,
        TunerEntity, WindowEntity,
        queue::{PauseRequestQueue, TunerAllocationQueue, TunerRequestQueue},
        resources::LocationResource,
        system::SystemContext,
    },
    hardware::pool::{Pool, Segment},
    shutdown::ShutdownCoordinator,
};

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
    task_entities: Option<Entities<TaskEntity>>,
    window_entities: Option<Entities<WindowEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    signal_entities: Option<Entities<SignalEntity>>,

    audio_streams: Resource<HashMap<AudioId, cpal::Stream>>,
    audio_segments: Resource<HashMap<AudioId, Segment>>,
    tuner_request_queue: Resource<TunerRequestQueue>,
    tuner_allocation_queue: Resource<TunerAllocationQueue>,
    pause_request_queue: Resource<PauseRequestQueue>,
    global_pause_resource: crate::ecs::GlobalPauseResource,
    location_resource: Option<LocationResource>,

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
            task_entities: None,
            window_entities: None,
            audio_entities: None,
            signal_entities: None,
            #[allow(clippy::arc_with_non_send_sync)]
            audio_streams: Arc::new(Mutex::new(HashMap::new())),
            #[allow(clippy::arc_with_non_send_sync)]
            audio_segments: Arc::new(Mutex::new(HashMap::new())),
            tuner_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            tuner_allocation_queue: Arc::new(Mutex::new(VecDeque::new())),
            pause_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            global_pause_resource: Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active)),
            location_resource: None,
            pool: Arc::clone(pool),
            config: Arc::clone(config),
            shutdown_coordinator: Arc::clone(shutdown_coordinator),
        }
    }

    /// Add task entities to the coordinator
    pub fn with_task_entities(mut self, entities: Entities<TaskEntity>) -> Self {
        self.task_entities = Some(entities);
        self
    }

    /// Add window entities to the coordinator
    pub fn with_window_entities(mut self, entities: Entities<WindowEntity>) -> Self {
        self.window_entities = Some(entities);
        self
    }

    /// Add audio entities to the coordinator
    pub fn with_audio_entities(mut self, entities: Entities<AudioEntity>) -> Self {
        self.audio_entities = Some(entities);
        self
    }

    /// Add signal entities to the coordinator
    pub fn with_signal_entities(mut self, entities: Entities<SignalEntity>) -> Self {
        self.signal_entities = Some(entities);
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

    /// Set the location resource for IP-based location detection
    pub fn with_location_resource(mut self, resource: LocationResource) -> Self {
        self.location_resource = Some(resource);
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
            .with_tuner_allocation_queue(Arc::clone(&self.tuner_allocation_queue))
            .with_pause_request_queue(Arc::clone(&self.pause_request_queue))
            .with_global_pause_resource(Arc::clone(&self.global_pause_resource))
            .with_pool(Arc::clone(&self.pool))
            .with_config(Arc::clone(&self.config))
            .with_shutdown_coordinator(Arc::clone(&self.shutdown_coordinator));

        if let Some(ref location_resource) = self.location_resource {
            context = context.with_location_resource(Arc::clone(location_resource));
        }

        if let Some(ref task_entities) = self.task_entities {
            context = context.with_task_entities(Arc::clone(task_entities));
        }

        if let Some(ref window_entities) = self.window_entities {
            context = context.with_window_entities(Arc::clone(window_entities));
        }

        if let Some(ref audio_entities) = self.audio_entities {
            context = context.with_audio_entities(Arc::clone(audio_entities));
        }

        if let Some(ref signal_entities) = self.signal_entities {
            context = context.with_signal_entities(Arc::clone(signal_entities));
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
    use std::sync::RwLock;

    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let pool = Arc::new(Pool::new_unfiltered());
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let coordinator = Coordinator::new(&pool, &config, &shutdown);

        assert_eq!(coordinator.system_count(), 0);
        assert!(coordinator.task_entities.is_none());
        assert!(coordinator.signal_entities.is_none());
        assert!(coordinator.audio_entities.is_none());
    }

    #[test]
    fn test_coordinator_with_entities() {
        let pool = Arc::new(Pool::new_unfiltered());
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let coordinator = Coordinator::new(&pool, &config, &shutdown)
            .with_task_entities(task_entities)
            .with_signal_entities(signal_entities)
            .with_audio_entities(audio_entities);

        assert!(coordinator.task_entities.is_some());
        assert!(coordinator.signal_entities.is_some());
        assert!(coordinator.audio_entities.is_some());
    }

    #[test]
    fn test_coordinator_tick_with_no_systems() {
        let pool = Arc::new(Pool::new_unfiltered());
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let mut coordinator = Coordinator::new(&pool, &config, &shutdown);

        let result = coordinator.tick();
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordinator_add_system() {
        let pool = Arc::new(Pool::new_unfiltered());
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
        let pool = Arc::new(Pool::new_unfiltered());
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
        let pool = Arc::new(Pool::new_unfiltered());
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
        let pool = Arc::new(Pool::new_unfiltered());
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
