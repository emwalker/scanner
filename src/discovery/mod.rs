mod common;
mod enumerator;
mod service;

#[cfg(target_os = "linux")]
mod udev_discovery;

mod polling;

pub use enumerator::{DeviceEnumerator, MultiEnumerator, SourcePriority};
pub use service::{Event, Service};

use crate::sdr;
use std::time::Duration;

pub enum DiscoveryMode {
    Auto,
    ForcePolling(Duration),
    #[cfg(target_os = "linux")]
    ForceUdev,
}

pub fn create(backends: Vec<Box<dyn sdr::Backend>>, mode: DiscoveryMode) -> Box<dyn Service> {
    use enumerator::{BackendEnumerator, MultiEnumerator, SourcePriority};

    let mut enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = vec![(
        Box::new(BackendEnumerator { backends }),
        SourcePriority::Backend,
    )];

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
    backends: Vec<Box<dyn sdr::Backend>>,
    mode: DiscoveryMode,
) -> Box<dyn Service> {
    use enumerator::{BackendEnumerator, MultiEnumerator, SourcePriority};

    let enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = vec![(
        Box::new(BackendEnumerator { backends }),
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
