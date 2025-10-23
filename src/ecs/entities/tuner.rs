//! Tuner entity - represents a physical SDR tuner (RX channel)

use crate::{
    ecs::{
        Entity,
        components::{
            AllocationComponent, ConstraintComponent, DeviceComponent, DisplayNameComponent,
            PriorityComponent, StatusComponent,
        },
    },
    hardware,
    hardware::pool::TunerId,
};

/// Entity representing a tuner (RX channel) within an SDR device
///
/// A tuner entity combines device information, allocation status, and
/// current activity into a single cohesive unit that can be managed
/// by ECS systems.
#[derive(Debug, Clone)]
pub struct TunerEntity {
    /// Unique identifier for this tuner
    id: TunerId,

    /// Device information
    pub device: DeviceComponent,

    /// Display name for the tuner
    pub display_name: DisplayNameComponent,

    /// Tuner mode (e.g., "ST", "DT" for RSPduo modes)
    pub mode: String,

    /// Allocation status
    pub allocation: AllocationComponent,

    /// Current status and activity
    pub status: StatusComponent,

    /// Allocation priorities
    pub priorities: PriorityComponent,

    /// Allocation constraints
    pub constraints: ConstraintComponent,
}

impl TunerEntity {
    /// Create a new tuner entity
    pub fn new(
        device_id: hardware::DeviceId,
        channel_index: usize,
        capabilities: hardware::Capabilities,
        backend: hardware::types::Backend,
        display_name: String,
        antenna: Option<String>,
        mode: String,
    ) -> Self {
        let id = TunerId::new(device_id.clone(), channel_index);

        Self {
            id,
            device: DeviceComponent::new(device_id, channel_index, capabilities, backend, antenna),
            display_name: DisplayNameComponent::new(display_name),
            mode,
            allocation: AllocationComponent::new(),
            status: StatusComponent::new(),
            priorities: PriorityComponent::default(),
            constraints: ConstraintComponent::default(),
        }
    }

    /// Check if tuner is available for allocation
    pub fn is_available(&self) -> bool {
        self.device.connected && self.allocation.is_available()
    }

    /// Check if tuner is connected
    pub fn is_connected(&self) -> bool {
        self.device.connected
    }
}

impl Entity for TunerEntity {
    type Id = TunerId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        ecs::{EntityWorld, components::TunerActivity},
        hardware::{Capabilities, DeviceId},
    };

    fn create_test_entity(device_serial: &str, channel: usize) -> TunerEntity {
        let device_id = DeviceId::from_serial("sdrplay", device_serial);
        let capabilities = Capabilities::for_device(&device_id);

        TunerEntity::new(
            device_id,
            channel,
            capabilities,
            hardware::types::Backend::Soapy,
            format!("Test Tuner {}", channel),
            None,
            "FM".to_string(),
        )
    }

    fn arb_backend() -> impl Strategy<Value = hardware::types::Backend> {
        prop_oneof![
            Just(hardware::types::Backend::Soapy),
            Just(hardware::types::Backend::Mock),
            Just(hardware::types::Backend::Usb),
        ]
    }

    fn arb_device_id() -> impl Strategy<Value = DeviceId> {
        ("[a-z]{3,8}", "[0-9]{4,8}")
            .prop_map(|(driver, serial)| DeviceId::from_serial(&driver, &serial))
    }

    fn arb_tuner_entity() -> impl Strategy<Value = TunerEntity> {
        (
            arb_device_id(),
            0..4usize,
            arb_backend(),
            any::<bool>(),
            any::<bool>(),
        )
            .prop_map(|(device_id, channel, backend, connected, allocated)| {
                let capabilities = Capabilities::for_device(&device_id);
                let mut entity = TunerEntity::new(
                    device_id,
                    channel,
                    capabilities,
                    backend,
                    format!("Test Tuner {}", channel),
                    None,
                    "FM".to_string(),
                );
                if !connected {
                    entity.device.disconnect();
                }
                if allocated && connected {
                    entity.allocation.allocate("test_alloc".to_string());
                    entity.status.start_scanning();
                }
                entity
            })
    }

    #[test]
    fn test_entity_creation() {
        let entity = create_test_entity("12345", 0);

        assert_eq!(entity.id.channel_index, 0);
        assert_eq!(entity.device.channel_index, 0);
        assert!(entity.device.connected);
        assert!(entity.allocation.is_available());
        assert_eq!(entity.status.activity, TunerActivity::Idle);
    }

    #[test]
    fn test_entity_trait_implementation() {
        let entity = create_test_entity("12345", 0);
        let id = entity.id();

        assert_eq!(id.channel_index, 0);
    }

    #[test]
    fn test_is_available_when_connected_and_unallocated() {
        let entity = create_test_entity("12345", 0);
        assert!(entity.is_available());
    }

    #[test]
    fn test_is_not_available_when_allocated() {
        let mut entity = create_test_entity("12345", 0);
        entity.allocation.allocate("scan_1".to_string());
        assert!(!entity.is_available());
    }

    #[test]
    fn test_is_not_available_when_disconnected() {
        let mut entity = create_test_entity("12345", 0);
        entity.device.disconnect();
        assert!(!entity.is_available());
    }

    #[test]
    fn test_is_connected() {
        let entity = create_test_entity("12345", 0);
        assert!(entity.is_connected());
    }

    #[test]
    fn test_is_not_connected_after_disconnect() {
        let mut entity = create_test_entity("12345", 0);
        entity.device.disconnect();
        assert!(!entity.is_connected());
    }

    #[test]
    fn test_entity_in_world() {
        let mut world = EntityWorld::new();
        let entity = create_test_entity("12345", 0);
        let id = entity.id().clone();

        world.insert(entity);
        assert_eq!(world.len(), 1);

        let retrieved = world.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id().channel_index, 0);
    }

    #[test]
    fn test_multiple_tuners_in_world() {
        let mut world = EntityWorld::new();

        let entity1 = create_test_entity("12345", 0);
        let entity2 = create_test_entity("12345", 1);
        let entity3 = create_test_entity("67890", 0);

        let id1 = entity1.id().clone();
        let id2 = entity2.id().clone();
        let id3 = entity3.id().clone();

        world.insert(entity1);
        world.insert(entity2);
        world.insert(entity3);

        assert_eq!(world.len(), 3);
        assert!(world.get(&id1).is_some());
        assert!(world.get(&id2).is_some());
        assert!(world.get(&id3).is_some());
    }

    #[test]
    fn test_modify_entity_in_world() {
        let mut world = EntityWorld::new();
        let entity = create_test_entity("12345", 0);
        let id = entity.id().clone();

        world.insert(entity);

        {
            let entity_mut = world.get_mut(&id).unwrap();
            entity_mut.allocation.allocate("scan_1".to_string());
            entity_mut.status.start_scanning();
        }

        let entity = world.get(&id).unwrap();
        assert!(entity.allocation.is_allocated());
        assert_eq!(entity.status.activity, TunerActivity::Scanning);
    }

    #[test]
    fn test_iterate_available_tuners() {
        let mut world = EntityWorld::new();

        let mut entity1 = create_test_entity("12345", 0);
        let entity2 = create_test_entity("12345", 1);
        let mut entity3 = create_test_entity("67890", 0);

        entity1.allocation.allocate("scan_1".to_string());
        entity3.device.disconnect();

        world.insert(entity1);
        world.insert(entity2);
        world.insert(entity3);

        let available_count = world.iter().filter(|e| e.is_available()).count();
        assert_eq!(available_count, 1);
    }

    proptest! {
        #[test]
        fn prop_allocation_consistency(entity in arb_tuner_entity()) {
            if entity.allocation.is_allocated() {
                prop_assert!(entity.allocation.allocated_to.is_some());
            } else {
                prop_assert!(entity.allocation.allocated_to.is_none());
            }
        }

        #[test]
        fn prop_connection_prerequisite(entity in arb_tuner_entity()) {
            if !entity.device.connected {
                prop_assert!(!entity.allocation.is_allocated());
            }
        }

        #[test]
        fn prop_activity_consistency(entity in arb_tuner_entity()) {
            if entity.allocation.is_allocated() {
                prop_assert_ne!(entity.status.activity, TunerActivity::Idle);
            }
        }

        #[test]
        fn prop_is_available_correctness(entity in arb_tuner_entity()) {
            let expected = entity.device.connected && entity.allocation.is_available();
            prop_assert_eq!(entity.is_available(), expected);
        }
    }
}
