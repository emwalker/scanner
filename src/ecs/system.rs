//! System trait and execution framework

use crate::core::types::Result;

/// Trait for ECS systems that operate on entities and components
///
/// Systems are pure functions that query entities, read/write components,
/// and implement game logic. They should be stateless where possible.
pub trait System: Send {
    /// System name for debugging and logging
    fn name(&self) -> &'static str;

    /// Execute the system
    ///
    /// Systems receive access to the world state through the context parameter.
    /// They should query entities, update components, and return any errors.
    fn run(&mut self, context: &mut SystemContext) -> Result<()>;
}

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    core::types::ScanningConfig,
    ecs::{
        AudioEntity, Entities, EntityWorld, SignalEntity, TaskEntity, TunerEntity, WindowEntity,
        components::audio::{AudioId, AudioQueueComponent},
        queue::{PauseRequestQueue, TunerAllocationQueue, TunerRequestQueue},
    },
    hardware::pool::{Pool, Segment},
    shutdown::ShutdownCoordinator,
};

/// Type alias for ECS resources
pub type Resource<T> = Arc<Mutex<T>>;

/// Context provided to systems during execution
///
/// This provides access to all entity worlds and shared resources needed
/// for system execution.
pub struct SystemContext {
    pub tuner_entities: Option<Arc<Mutex<EntityWorld<TunerEntity>>>>,
    pub task_entities: Option<Entities<TaskEntity>>,
    pub window_entities: Option<Entities<WindowEntity>>,
    pub audio_entities: Option<Entities<AudioEntity>>,
    pub signal_entities: Option<Entities<SignalEntity>>,

    /// Audio streams resource (can't be in entities - cpal::Stream is not Send)
    pub audio_streams: Option<Resource<HashMap<AudioId, cpal::Stream>>>,

    /// SDR segments resource (can't be in entities - Segment contains PooledTuner which is not
    /// Send)
    pub audio_segments: Option<Resource<HashMap<AudioId, Segment>>>,

    /// Audio playback queue for discovery and listening requests
    pub audio_queue: Option<Resource<AudioQueueComponent>>,

    /// FIFO queue for tuner acquisition requests
    pub tuner_request_queue: Option<Resource<TunerRequestQueue>>,

    /// Unified FIFO queue for tuner allocation (windows, stations)
    pub tuner_allocation_queue: Option<Resource<TunerAllocationQueue>>,

    /// FIFO queue for pause requests
    pub pause_request_queue: Option<Resource<PauseRequestQueue>>,

    /// Global pause resource
    pub global_pause_resource: Option<crate::ecs::GlobalPauseResource>,

    pub pool: Option<Arc<Pool>>,
    pub config: Option<Arc<ScanningConfig>>,
    pub shutdown_coordinator: Option<Arc<ShutdownCoordinator>>,
}

impl Default for SystemContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemContext {
    pub fn new() -> Self {
        Self {
            tuner_entities: None,
            task_entities: None,
            window_entities: None,
            audio_entities: None,
            signal_entities: None,
            audio_streams: None,
            audio_segments: None,
            audio_queue: None,
            tuner_request_queue: None,
            tuner_allocation_queue: None,
            pause_request_queue: None,
            global_pause_resource: None,
            pool: None,
            config: None,
            shutdown_coordinator: None,
        }
    }

    pub fn with_tuner_entities(mut self, entities: Arc<Mutex<EntityWorld<TunerEntity>>>) -> Self {
        self.tuner_entities = Some(entities);
        self
    }

    pub fn with_task_entities(mut self, entities: Entities<TaskEntity>) -> Self {
        self.task_entities = Some(entities);
        self
    }

    pub fn with_window_entities(mut self, entities: Entities<WindowEntity>) -> Self {
        self.window_entities = Some(entities);
        self
    }

    pub fn with_audio_entities(mut self, entities: Entities<AudioEntity>) -> Self {
        self.audio_entities = Some(entities);
        self
    }

    pub fn with_signal_entities(mut self, entities: Entities<SignalEntity>) -> Self {
        self.signal_entities = Some(entities);
        self
    }

    pub fn with_audio_streams(mut self, streams: Resource<HashMap<AudioId, cpal::Stream>>) -> Self {
        self.audio_streams = Some(streams);
        self
    }

    pub fn with_audio_segments(mut self, segments: Resource<HashMap<AudioId, Segment>>) -> Self {
        self.audio_segments = Some(segments);
        self
    }

    pub fn with_audio_queue(mut self, queue: Resource<AudioQueueComponent>) -> Self {
        self.audio_queue = Some(queue);
        self
    }

    pub fn with_tuner_request_queue(mut self, queue: Resource<TunerRequestQueue>) -> Self {
        self.tuner_request_queue = Some(queue);
        self
    }

    pub fn with_tuner_allocation_queue(mut self, queue: Resource<TunerAllocationQueue>) -> Self {
        self.tuner_allocation_queue = Some(queue);
        self
    }

    pub fn with_pause_request_queue(mut self, queue: Resource<PauseRequestQueue>) -> Self {
        self.pause_request_queue = Some(queue);
        self
    }

    pub fn with_global_pause_resource(mut self, resource: crate::ecs::GlobalPauseResource) -> Self {
        self.global_pause_resource = Some(resource);
        self
    }

    pub fn with_pool(mut self, pool: Arc<Pool>) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn with_config(mut self, config: Arc<ScanningConfig>) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_shutdown_coordinator(mut self, shutdown: Arc<ShutdownCoordinator>) -> Self {
        self.shutdown_coordinator = Some(shutdown);
        self
    }

    /// Check if the application is globally paused
    ///
    /// Returns true if the global_pause_resource is present and in Paused state.
    /// Uses try_lock() for shutdown safety.
    pub fn is_globally_paused(&self) -> bool {
        if let Some(ref resource) = self.global_pause_resource
            && let Ok(state) = resource.try_lock()
        {
            return matches!(*state, crate::ecs::GlobalPauseState::Paused { .. });
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSystem {
        run_count: usize,
    }

    impl System for TestSystem {
        fn name(&self) -> &'static str {
            "TestSystem"
        }

        fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
            self.run_count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_system_execution() {
        let mut system = TestSystem { run_count: 0 };
        let mut context = SystemContext::new();

        assert_eq!(system.name(), "TestSystem");
        assert_eq!(system.run_count, 0);

        system.run(&mut context).unwrap();
        assert_eq!(system.run_count, 1);

        system.run(&mut context).unwrap();
        assert_eq!(system.run_count, 2);
    }

    struct FailingSystem;

    impl System for FailingSystem {
        fn name(&self) -> &'static str {
            "FailingSystem"
        }

        fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
            Err(crate::core::types::ScannerError::Custom(
                "System failure".to_string(),
            ))
        }
    }

    #[test]
    fn test_system_error_handling() {
        let mut system = FailingSystem;
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_globally_paused_without_resource() {
        let context = SystemContext::new();
        assert!(!context.is_globally_paused());
    }

    #[test]
    fn test_is_globally_paused_when_active() {
        use std::sync::{Arc, Mutex};

        use crate::ecs::GlobalPauseState;

        let resource = Arc::new(Mutex::new(GlobalPauseState::Active));
        let context = SystemContext::new().with_global_pause_resource(resource);

        assert!(!context.is_globally_paused());
    }

    #[test]
    fn test_is_globally_paused_when_paused() {
        use std::sync::{Arc, Mutex};

        use crate::ecs::GlobalPauseState;

        let resource = Arc::new(Mutex::new(GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));
        let context = SystemContext::new().with_global_pause_resource(resource);

        assert!(context.is_globally_paused());
    }
}
