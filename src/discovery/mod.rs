mod common;
mod enumerator;
mod service;

#[cfg(target_os = "linux")]
mod udev_discovery;

mod polling;

pub use enumerator::{DeviceEnumerator, MultiEnumerator, SourcePriority};
pub use service::{Event, Service};

use crate::core::types::Result;
use crate::hardware;
use std::time::Duration;

pub enum DiscoveryMode {
    Auto,
    ForcePolling(Duration),
    #[cfg(target_os = "linux")]
    ForceUdev,
}

pub fn create(backend_names: Vec<String>, mode: DiscoveryMode) -> Box<dyn Service> {
    use enumerator::{MultiEnumerator, SourcePriority, SubprocessEnumerator};

    let mut enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = backend_names
        .into_iter()
        .map(|name| {
            (
                Box::new(SubprocessEnumerator::new(name)) as Box<dyn DeviceEnumerator>,
                SourcePriority::Backend,
            )
        })
        .collect();

    #[cfg(target_os = "linux")]
    {
        use enumerator::UsbEnumerator;
        enumerators.push((
            Box::new(UsbEnumerator::new()),
            SourcePriority::UsbInspection,
        ));
    }

    let enumerator = MultiEnumerator { enumerators };

    match mode {
        DiscoveryMode::ForcePolling(interval) => {
            Box::new(polling::Polling::new(enumerator, interval))
        }
        #[cfg(target_os = "linux")]
        DiscoveryMode::ForceUdev => Box::new(udev_discovery::Udev::new(enumerator)),
        DiscoveryMode::Auto => {
            #[cfg(target_os = "linux")]
            {
                Box::new(udev_discovery::Udev::new(enumerator))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Box::new(polling::Polling::new(enumerator, Duration::from_secs(3)))
            }
        }
    }
}

/// Create a discovery service for testing that only uses backend enumeration
///
/// This bypasses USB enumeration to avoid detecting real hardware during tests.
/// Intended for test use only.
pub fn create_for_testing(
    backends: Vec<Box<dyn hardware::Backend>>,
    mode: DiscoveryMode,
) -> Box<dyn Service> {
    use enumerator::{DirectEnumerator, MultiEnumerator, SourcePriority};

    let enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = vec![(
        Box::new(DirectEnumerator { backends }),
        SourcePriority::Backend,
    )];

    let enumerator = MultiEnumerator { enumerators };

    match mode {
        DiscoveryMode::ForcePolling(interval) => {
            Box::new(polling::Polling::new(enumerator, interval))
        }
        #[cfg(target_os = "linux")]
        DiscoveryMode::ForceUdev => Box::new(udev_discovery::Udev::new(enumerator)),
        DiscoveryMode::Auto => {
            #[cfg(target_os = "linux")]
            {
                Box::new(udev_discovery::Udev::new(enumerator))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Box::new(polling::Polling::new(enumerator, Duration::from_secs(3)))
            }
        }
    }
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
    backend_names: &[String],
    filter: Option<&str>,
) -> Result<Vec<hardware::DeviceInfo>> {
    use enumerator::{MultiEnumerator, SourcePriority, SubprocessEnumerator};

    let backend_enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = backend_names
        .iter()
        .map(|name| {
            (
                Box::new(SubprocessEnumerator::new(name.clone())) as Box<dyn DeviceEnumerator>,
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
                    (hardware::DeviceId::Backend { backend, .. }, "driver") => backend == value,
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
                    (hardware::DeviceId::Backend { backend, .. }, "driver") => backend == value,
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
