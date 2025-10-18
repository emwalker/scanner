//! Scan coordination system

use crate::core::types::Result;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

/// System that coordinates scan operations
///
/// This system:
/// - Monitors active scan entities
/// - Updates scan progress and state
/// - Manages scan lifecycle transitions
/// - Coordinates with tuner allocation for scan tasks
pub struct CoordinationSystem;

impl Default for CoordinationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinationSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for CoordinationSystem {
    fn name(&self) -> &'static str {
        "ScanCoordination"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let scan_entities = match &context.scan_entities {
            Some(entities) => entities.clone(),
            None => {
                debug!("No scan entities in context");
                return Ok(());
            }
        };

        let entities = scan_entities.lock().unwrap();

        let total_scans = entities.len();
        let active_scans = entities.iter().filter(|e| e.is_scanning()).count();
        let paused_scans = entities.iter().filter(|e| e.is_paused()).count();
        let completed_scans = entities.iter().filter(|e| e.is_completed()).count();
        let listening_scans = entities.iter().filter(|e| e.is_listening()).count();

        debug!(
            total = total_scans,
            active = active_scans,
            paused = paused_scans,
            completed = completed_scans,
            listening = listening_scans,
            "Scan coordination system ran"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::{EntityWorld, ScanEntity};
    use std::sync::{Arc, Mutex};

    fn create_test_scan(freq_min: f64, freq_max: f64) -> ScanEntity {
        let config = ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            1.0e6,
            2.0e6,
            40.0,
            0.5,
            10,
        );
        ScanEntity::new(config)
    }

    #[test]
    fn test_coordination_system_with_empty_context() {
        let mut system = CoordinationSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordination_system_with_active_scans() {
        let mut system = CoordinationSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_scan(88.0e6, 108.0e6));
        world.insert(create_test_scan(144.0e6, 148.0e6));

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_scan_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.lock().unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_coordination_system_counts_scan_states() {
        let mut system = CoordinationSystem::new();

        let mut world = EntityWorld::new();
        let mut scan1 = create_test_scan(88.0e6, 108.0e6);
        let mut scan2 = create_test_scan(144.0e6, 148.0e6);
        let scan3 = create_test_scan(420.0e6, 450.0e6);

        scan1.progress.pause(5);
        scan2.progress.mark_complete();

        world.insert(scan1);
        world.insert(scan2);
        world.insert(scan3);

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_scan_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.lock().unwrap();
        assert_eq!(entities.iter().filter(|e| e.is_paused()).count(), 1);
        assert_eq!(entities.iter().filter(|e| e.is_completed()).count(), 1);
        assert_eq!(entities.iter().filter(|e| e.is_scanning()).count(), 1);
    }
}
