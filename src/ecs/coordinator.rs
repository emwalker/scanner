//! ECS Coordinator - executes systems in the proper order each tick

use crate::core::types::Result;
use crate::ecs::EntityWorld;
use crate::ecs::Scheduler;
use crate::ecs::system::SystemContext;
use crate::ecs::{AudioEntity, CandidateEntity, Entities, ScanEntity, StationEntity, TunerEntity};
use crate::hardware::pool::Pool;
use std::sync::{Arc, Mutex};

/// Coordinator manages system execution and provides the update loop
///
/// The coordinator:
/// - Maintains references to all entity worlds
/// - Executes systems in the correct order each tick
/// - Provides a clean interface for the main loop to interact with ECS
pub struct Coordinator {
    scheduler: Scheduler,
    tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>,
    scan_entities: Option<Entities<ScanEntity>>,
    station_entities: Option<Entities<StationEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    candidate_entities: Option<Entities<CandidateEntity>>,
}

impl Coordinator {
    /// Create a new coordinator with a pool (which contains tuner entities)
    pub fn new(pool: &Arc<Pool>) -> Self {
        Self {
            scheduler: Scheduler::new(),
            tuner_entities: Arc::clone(&pool.tuner_entities),
            scan_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
        }
    }

    /// Add scan entities to the coordinator
    pub fn with_scan_entities(mut self, entities: Entities<ScanEntity>) -> Self {
        self.scan_entities = Some(entities);
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
        let mut context =
            SystemContext::new().with_tuner_entities(Arc::clone(&self.tuner_entities));

        if let Some(ref scan_entities) = self.scan_entities {
            context = context.with_scan_entities(Arc::clone(scan_entities));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::pool::PoolFilter;
    use std::sync::RwLock;

    #[test]
    fn test_coordinator_creation() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let coordinator = Coordinator::new(&pool);

        assert_eq!(coordinator.system_count(), 0);
        assert!(coordinator.scan_entities.is_none());
        assert!(coordinator.station_entities.is_none());
        assert!(coordinator.audio_entities.is_none());
    }

    #[test]
    fn test_coordinator_with_entities() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let coordinator = Coordinator::new(&pool)
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
        let mut coordinator = Coordinator::new(&pool);

        let result = coordinator.tick();
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordinator_add_system() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let mut coordinator = Coordinator::new(&pool);

        coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
        assert_eq!(coordinator.system_count(), 1);

        coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));
        assert_eq!(coordinator.system_count(), 2);
    }

    #[test]
    fn test_coordinator_tick_with_systems() {
        let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
        let mut coordinator = Coordinator::new(&pool);

        coordinator.add_system(Box::new(crate::ecs::systems::DiscoverySystem::new()));
        coordinator.add_system(Box::new(crate::ecs::systems::AllocationSystem::new()));

        let result = coordinator.tick();
        assert!(result.is_ok());
    }
}
