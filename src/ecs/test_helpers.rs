//! Test helpers for working with ECS entities in tests

use std::sync::{Arc, Mutex};

use crate::{
    ecs::{DeviceEntity, EntityWorld, TunerEntity},
    hardware::{
        Capabilities, DeviceId,
        pool::{Pool, PoolFilter},
        types::{Backend, TunerInfo},
    },
};

/// Create test device and tuner entities for use in tests
///
/// This helper mimics the entity creation pattern used by DeviceEnumerationTask
/// in production code, but simplified for test usage.
///
/// # Arguments
/// * `device_id` - Device identifier
/// * `capabilities` - Device capabilities (includes channel count)
/// * `backend` - Backend type
/// * `label` - Optional device label (defaults to "Test Device")
/// * `tuner_info` - Tuner-specific info; if empty, creates simple tuners based on channel count
///
/// # Returns
/// Tuple of (DeviceEntity, Vec<TunerEntity>)
///
/// # Example
/// ```no_run
/// use scanner::{
///     ecs::test_helpers::create_test_device_and_tuners,
///     hardware::{Capabilities, DeviceId, types::Backend},
/// };
///
/// let device_id = DeviceId::from_serial("mock", "test001");
/// let mut caps = Capabilities::for_mock("mock", "test001");
/// caps.channels = 2;
///
/// let (device, tuners) =
///     create_test_device_and_tuners(device_id, caps, Backend::Mock, None, vec![]);
///
/// assert_eq!(tuners.len(), 2);
/// ```
pub fn create_test_device_and_tuners(
    device_id: DeviceId,
    capabilities: Capabilities,
    backend: Backend,
    label: Option<String>,
    tuner_info: Vec<TunerInfo>,
) -> (DeviceEntity, Vec<TunerEntity>) {
    let device_entity = DeviceEntity::new_metadata_only(
        device_id.clone(),
        label.unwrap_or_else(|| "Test Device".to_string()),
        capabilities.clone(),
        backend.clone(),
    );

    let tuner_entities = if tuner_info.is_empty() {
        // No tuner info provided - create simple tuners based on channel count
        (0..capabilities.channels)
            .map(|channel_index| {
                TunerEntity::new(
                    device_id.clone(),
                    channel_index,
                    capabilities.clone(),
                    backend.clone(),
                    format!("Test Tuner {}", channel_index),
                    None,
                    "Test".to_string(),
                )
            })
            .collect()
    } else {
        // Use provided tuner info
        tuner_info
            .into_iter()
            .map(|info| {
                TunerEntity::new(
                    device_id.clone(),
                    info.id.channel_index,
                    capabilities.clone(),
                    backend.clone(),
                    info.label,
                    info.antenna,
                    info.mode,
                )
            })
            .collect()
    };

    (device_entity, tuner_entities)
}

/// Create a Pool with empty EntityWorlds for testing
///
/// Returns tuple of (Pool, tuner_entities, device_entities) to allow tests
/// to inspect and manipulate entities directly.
///
/// # Arguments
/// * `filter` - PoolFilter to apply
/// * `parent_log_file` - Optional parent log file path
///
/// # Returns
/// Tuple of (Arc<Pool>, Arc<Mutex<EntityWorld<TunerEntity>>>,
/// Arc<Mutex<EntityWorld<DeviceEntity>>>)
///
/// # Example
/// ```no_run
/// use scanner::{ecs::test_helpers::create_test_pool_with_entities, hardware::pool::PoolFilter};
///
/// let (pool, tuner_entities, device_entities) =
///     create_test_pool_with_entities(PoolFilter::allow_all(), None);
/// ```
#[allow(clippy::type_complexity)]
pub fn create_test_pool_with_entities(
    filter: PoolFilter,
    parent_log_file: Option<String>,
) -> (
    Arc<Pool>,
    Arc<Mutex<EntityWorld<TunerEntity>>>,
    Arc<Mutex<EntityWorld<DeviceEntity>>>,
) {
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let device_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let pool = Arc::new(Pool::with_entity_worlds(
        filter,
        parent_log_file,
        tuner_entities.clone(),
        device_entities.clone(),
    ));
    (pool, tuner_entities, device_entities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ecs::Entity, hardware::pool::TunerId};

    #[test]
    fn test_create_entities_without_tuner_info() {
        let device_id = DeviceId::from_serial("mock", "test001");
        let mut caps = Capabilities::for_mock("mock", "test001");
        caps.channels = 3;

        let (device, tuners) = create_test_device_and_tuners(
            device_id.clone(),
            caps,
            Backend::Mock,
            Some("Test Device".to_string()),
            vec![],
        );

        assert_eq!(device.id(), &device_id);
        assert_eq!(tuners.len(), 3);

        for (i, tuner) in tuners.iter().enumerate() {
            assert_eq!(tuner.id().channel_index, i);
        }
    }

    #[test]
    fn test_create_entities_with_tuner_info() {
        let device_id = DeviceId::from_serial("mock", "test002");
        let mut caps = Capabilities::for_mock("mock", "test002");
        caps.channels = 2;

        let tuner_info = vec![
            TunerInfo {
                id: TunerId::new(device_id.clone(), 0),
                label: "Channel A".to_string(),
                mode: "ST".to_string(),
                antenna: Some("Antenna 1".to_string()),
            },
            TunerInfo {
                id: TunerId::new(device_id.clone(), 1),
                label: "Channel B".to_string(),
                mode: "DT".to_string(),
                antenna: Some("Antenna 2".to_string()),
            },
        ];

        let (device, tuners) =
            create_test_device_and_tuners(device_id.clone(), caps, Backend::Mock, None, tuner_info);

        assert_eq!(device.id(), &device_id);
        assert_eq!(tuners.len(), 2);
        assert_eq!(tuners[0].display_name.name, "Channel A");
        assert_eq!(tuners[0].mode, "ST");
        assert_eq!(tuners[1].display_name.name, "Channel B");
        assert_eq!(tuners[1].mode, "DT");
    }
}
