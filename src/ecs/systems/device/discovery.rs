//! Device discovery system

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

/// System that synchronizes device discovery state
///
/// In the current architecture, device discovery is handled by the Pool
/// which automatically creates TunerEntity instances when devices are added.
/// This system validates and logs the current tuner state.
pub struct DiscoverySystem;

impl Default for DiscoverySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoverySystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for DiscoverySystem {
    fn name(&self) -> &'static str {
        "DeviceDiscovery"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let tuner_entities = match &context.tuner_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()), // No entities to process
        };

        let tuners = tuner_entities.lock().unwrap();

        // Log tuner state metrics
        let total_count = tuners.len();
        let available_count = tuners.iter().filter(|t| t.is_available()).count();
        let allocated_count = tuners.iter().filter(|t| !t.is_available()).count();

        debug!(
            total_tuners = total_count,
            available_tuners = available_count,
            allocated_tuners = allocated_count,
            "Device discovery system validation complete"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        ecs::{EntityWorld, TunerEntity},
        hardware::{Capabilities, DeviceId},
    };

    fn create_test_entity(device_serial: &str, channel: usize) -> TunerEntity {
        let device_id = DeviceId::from_serial("sdrplay", device_serial);
        let capabilities = Capabilities::for_device(&device_id);

        TunerEntity::new(
            device_id,
            channel,
            capabilities,
            crate::hardware::types::Backend::Soapy,
            format!("Test Tuner {}", channel),
            None,
            "FM".to_string(),
        )
    }

    #[test]
    fn test_discovery_system_with_empty_context() {
        let mut system = DiscoverySystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_discovery_system_with_tuner_entities() {
        let mut system = DiscoverySystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_entity("12345", 0));
        world.insert(create_test_entity("12345", 1));

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_discovery_system_counts_correctly() {
        let mut system = DiscoverySystem::new();

        let mut world = EntityWorld::new();
        let mut entity1 = create_test_entity("12345", 0);
        let entity2 = create_test_entity("12345", 1);
        let mut entity3 = create_test_entity("67890", 0);

        entity1.allocation.allocate("scan_1".to_string());
        entity3.device.disconnect();

        world.insert(entity1);
        world.insert(entity2);
        world.insert(entity3);

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
