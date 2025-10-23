//! Test utilities for populating Pool EntityWorlds
//!
//! This module provides test-only helpers for creating device and tuner entities
//! directly in Pool's EntityWorlds. This mirrors the pattern used by
//! DeviceEnumerationTask in production code, keeping Pool as a passive query
//! interface following ECS design principles.

use tracing::{debug, info};

use super::Pool;
use crate::{
    ecs::{DeviceEntity, TunerEntity},
    hardware::{Capabilities, DeviceId, pool::TunerId, types::Backend},
};

/// Result of adding a test device to the pool
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddTestDeviceResult {
    /// Device and tuners were successfully added
    Added {
        device_id: DeviceId,
        tuner_count: usize,
    },
    /// Device was filtered out (no tuners passed filter)
    FilteredOut { device_id: DeviceId, reason: String },
    /// Could not acquire EntityWorld locks (pool busy)
    PoolBusy,
}

/// Add a test device with tuners to the pool's EntityWorlds
///
/// This is a test-only helper that creates DeviceEntity and TunerEntity instances
/// and inserts them directly into the Pool's EntityWorlds. This mirrors the pattern
/// used by DeviceEnumerationTask in production code.
///
/// # Arguments
/// * `pool` - The pool to add entities to
/// * `device_id` - Unique device identifier
/// * `capabilities` - Device capabilities (includes channel count)
/// * `backend` - Backend type (Mock, Soapy, etc.)
/// * `label` - Optional device label (defaults to device_id string representation)
///
/// # Returns
/// `AddTestDeviceResult` indicating success, filter rejection, or lock contention
///
/// # Example
/// ```no_run
/// use scanner::hardware::{
///     Capabilities, DeviceId,
///     pool::{Pool, test_utils::add_test_device_to_pool},
///     types::Backend,
/// };
///
/// let pool = Pool::new_unfiltered();
/// let device_id = DeviceId::from_serial("mock", "test001");
/// let capabilities = Capabilities::for_mock("mock", "test001");
///
/// let result = add_test_device_to_pool(&pool, device_id, capabilities, Backend::Mock, None);
/// ```
pub fn add_test_device_to_pool(
    pool: &Pool,
    device_id: DeviceId,
    capabilities: Capabilities,
    backend: Backend,
    label: Option<String>,
) -> AddTestDeviceResult {
    let num_tuners = capabilities.channels;

    // Create DeviceEntity (metadata only for tests - no actual device handle)
    let mut device_entities = match pool.device_entities.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            debug!(device_id = ?device_id, "Add device failed - device entities locked");
            return AddTestDeviceResult::PoolBusy;
        }
    };

    let label = label.unwrap_or_else(|| device_id.as_str().to_string());
    let device_entity = DeviceEntity::new_metadata_only(
        device_id.clone(),
        label.clone(),
        capabilities.clone(),
        backend.clone(),
    );

    device_entities.insert(device_entity);
    debug!(device_id = ?device_id, "Created DeviceEntity");
    drop(device_entities);

    // Create TunerEntity instances
    let mut tuner_entities = match pool.tuner_entities.try_lock() {
        Ok(entities) => entities,
        Err(_) => {
            debug!(device_id = ?device_id, "Failed to expose tuners - entities locked");
            return AddTestDeviceResult::PoolBusy;
        }
    };

    let allocated_count = tuner_entities
        .iter()
        .filter(|e| e.allocation.is_allocated())
        .count();

    let mut exposed_count = 0;

    for channel_index in 0..num_tuners {
        let tuner_id = TunerId::new(device_id.clone(), channel_index);

        // Check if this tuner passes the filter
        if !pool
            .filter
            .is_allowed(&tuner_id, &backend, allocated_count, "")
        {
            debug!(
                tuner_id = ?tuner_id,
                "Tuner filtered out - not exposing"
            );
            continue;
        }

        debug!(
            tuner_id = ?tuner_id,
            "Exposing tuner {}/{}", channel_index + 1, num_tuners
        );

        let display_name = if num_tuners > 1 {
            format!("{} Ch{}", label, channel_index)
        } else {
            label.clone()
        };

        let entity = TunerEntity::new(
            device_id.clone(),
            channel_index,
            capabilities.clone(),
            backend.clone(),
            display_name.clone(),
            None,
            "".to_string(),
        );
        tuner_entities.insert(entity);
        exposed_count += 1;
        info!(
            tuner_id = ?tuner_id,
            display_name = display_name,
            "Created TunerEntity"
        );
    }

    drop(tuner_entities);

    if exposed_count == 0 {
        // No tuners passed the filter - remove the device
        debug!(
            device_id = ?device_id,
            "Removing device - no tuners passed filter"
        );
        if let Err(e) = pool.remove_device(&device_id) {
            debug!(
                device_id = ?device_id,
                error = ?e,
                "Failed to remove filtered device (ignoring)"
            );
        }
        return AddTestDeviceResult::FilteredOut {
            device_id,
            reason: "No tuners passed filter criteria".to_string(),
        };
    }

    AddTestDeviceResult::Added {
        device_id,
        tuner_count: exposed_count,
    }
}
