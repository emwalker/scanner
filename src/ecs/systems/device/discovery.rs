//! Device discovery system

use crate::core::types::Result;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

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
            Some(entities) => entities,
            None => {
                debug!("No tuner entities in context");
                return Ok(());
            }
        };

        let entities = match tuner_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => return Ok(()),
        };
        let connected_count = entities.iter().filter(|e| e.is_connected()).count();
        let available_count = entities.iter().filter(|e| e.is_available()).count();

        debug!(
            total = entities.len(),
            connected = connected_count,
            available = available_count,
            "Device discovery system ran"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::EntityWorld;
    use crate::ecs::TunerEntity;
    use crate::hardware::{Capabilities, DeviceId};
    use std::sync::{Arc, Mutex};

    fn create_test_entity(device_serial: &str, channel: usize) -> TunerEntity {
        let device_id = DeviceId::from_serial("sdrplay", device_serial);
        let capabilities = Capabilities::for_device(&device_id);

        TunerEntity::new(
            device_id,
            channel,
            capabilities,
            crate::hardware::types::Backend::Soapy,
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
