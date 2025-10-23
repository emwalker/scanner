mod common;
mod enumerator;
mod service;
pub mod tracker;

#[cfg(target_os = "linux")]
mod udev_discovery;

mod polling;

use std::time::Duration;

pub use enumerator::{DeviceEnumerator, MultiEnumerator, SourcePriority, SubprocessEnumerator};
pub use service::{Event, Service};
pub use tracker::DeviceTracker;

use crate::{core::types::Result, hardware};

pub enum DiscoveryMode {
    Auto,
    ForcePolling(Duration),
    #[cfg(target_os = "linux")]
    ForceUdev,
}

pub fn create(
    backends: Vec<hardware::types::Backend>,
    mode: DiscoveryMode,
    scheduler: std::sync::Arc<crate::task::TaskScheduler>,
    pool: std::sync::Arc<crate::hardware::pool::Pool>,
    tuner_entities: std::sync::Arc<
        std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>,
    >,
    device_entities: std::sync::Arc<
        std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>,
    >,
) -> Box<dyn Service> {
    match mode {
        DiscoveryMode::ForcePolling(interval) => Box::new(polling::Polling::new(
            scheduler,
            pool,
            backends,
            interval,
            tuner_entities,
            device_entities,
        )),
        #[cfg(target_os = "linux")]
        DiscoveryMode::ForceUdev => Box::new(udev_discovery::Udev::new(
            scheduler,
            pool,
            backends,
            tuner_entities,
            device_entities,
        )),
        DiscoveryMode::Auto => {
            #[cfg(target_os = "linux")]
            {
                Box::new(udev_discovery::Udev::new(
                    scheduler,
                    pool,
                    backends,
                    tuner_entities,
                    device_entities,
                ))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Box::new(polling::Polling::new(
                    scheduler,
                    pool,
                    backends,
                    Duration::from_secs(3),
                    tuner_entities,
                    device_entities,
                ))
            }
        }
    }
}

/// Create a discovery service for testing
///
/// For testing, use Backend::Mock and pass a scheduler/pool configured for testing.
pub fn create_for_testing(
    backends: Vec<hardware::types::Backend>,
    mode: DiscoveryMode,
    scheduler: std::sync::Arc<crate::task::TaskScheduler>,
    pool: std::sync::Arc<crate::hardware::pool::Pool>,
    tuner_entities: std::sync::Arc<
        std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>,
    >,
    device_entities: std::sync::Arc<
        std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>,
    >,
) -> Box<dyn Service> {
    create(
        backends,
        mode,
        scheduler,
        pool,
        tuner_entities,
        device_entities,
    )
}

/// Synchronously enumerate devices via subprocess worker matching an optional filter
///
/// This provides a one-time synchronous device enumeration using subprocess isolation,
/// useful for initial device selection before starting the async discovery service.
///
/// # Arguments
/// * `backend_names` - List of backend names to query (e.g., ["soapy"])
/// * `filter` - Optional driver filter (e.g., "driver=sdrplay")
///
/// # Returns
/// List of discovered devices matching the filter
pub fn enumerate_once_subprocess(
    backends: &[hardware::types::Backend],
    filter: Option<&str>,
    parent_log_file: Option<String>,
) -> Result<Vec<hardware::DeviceInfo>> {
    use enumerator::{MultiEnumerator, SourcePriority, SubprocessEnumerator};

    let backend_enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = backends
        .iter()
        .map(|backend| {
            (
                Box::new(SubprocessEnumerator::new(
                    backend.as_str().to_string(),
                    parent_log_file.clone(),
                )) as Box<dyn DeviceEnumerator>,
                SourcePriority::Backend,
            )
        })
        .collect();

    let enumerator = MultiEnumerator {
        enumerators: backend_enumerators,
    };

    let all_devices = enumerator.enumerate();

    // Apply filter if provided
    if let Some(filter_str) = filter {
        // Parse filter format: "key=value"
        if let Some((key, value)) = filter_str.split_once('=') {
            Ok(all_devices
                .into_iter()
                .filter(|device| match (&device.id, key) {
                    (hardware::DeviceId::Driver { driver, .. }, "driver") => driver == value,
                    _ => false,
                })
                .collect())
        } else {
            Ok(all_devices)
        }
    } else {
        Ok(all_devices)
    }
}

/// Synchronously enumerate devices matching an optional filter
///
/// This provides a one-time synchronous device enumeration, useful for
/// initial device selection before starting the async discovery service.
///
/// # Arguments
/// * `backends` - List of backends to query for devices
/// * `filter` - Optional driver filter (e.g., "driver=sdrplay")
///
/// # Returns
/// List of discovered devices matching the filter
pub fn enumerate_once(
    backends: &[Box<dyn hardware::Backend>],
    filter: Option<&str>,
) -> Result<Vec<hardware::DeviceInfo>> {
    let mut all_devices = Vec::new();

    for backend in backends {
        let devices = backend.enumerate_devices()?;
        all_devices.extend(devices);
    }

    // Apply filter if provided
    if let Some(filter_str) = filter {
        // Parse filter format: "key=value"
        if let Some((key, value)) = filter_str.split_once('=') {
            Ok(all_devices
                .into_iter()
                .filter(|device| match (&device.id, key) {
                    (hardware::DeviceId::Driver { driver, .. }, "driver") => driver == value,
                    _ => false,
                })
                .collect())
        } else {
            Ok(all_devices)
        }
    } else {
        Ok(all_devices)
    }
}
