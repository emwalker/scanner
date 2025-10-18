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

use crate::ecs::EntityWorld;
use crate::ecs::{AudioEntity, ScanEntity, StationEntity, TunerEntity};
use std::sync::{Arc, Mutex};

/// Context provided to systems during execution
///
/// This provides access to all entity worlds and shared resources needed
/// for system execution.
pub struct SystemContext {
    pub tuner_entities: Option<Arc<Mutex<EntityWorld<TunerEntity>>>>,
    pub scan_entities: Option<Arc<Mutex<EntityWorld<ScanEntity>>>>,
    pub station_entities: Option<Arc<Mutex<EntityWorld<StationEntity>>>>,
    pub audio_entities: Option<Arc<Mutex<EntityWorld<AudioEntity>>>>,
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
            scan_entities: None,
            station_entities: None,
            audio_entities: None,
        }
    }

    pub fn with_tuner_entities(mut self, entities: Arc<Mutex<EntityWorld<TunerEntity>>>) -> Self {
        self.tuner_entities = Some(entities);
        self
    }

    pub fn with_scan_entities(mut self, entities: Arc<Mutex<EntityWorld<ScanEntity>>>) -> Self {
        self.scan_entities = Some(entities);
        self
    }

    pub fn with_station_entities(
        mut self,
        entities: Arc<Mutex<EntityWorld<StationEntity>>>,
    ) -> Self {
        self.station_entities = Some(entities);
        self
    }

    pub fn with_audio_entities(mut self, entities: Arc<Mutex<EntityWorld<AudioEntity>>>) -> Self {
        self.audio_entities = Some(entities);
        self
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
}
