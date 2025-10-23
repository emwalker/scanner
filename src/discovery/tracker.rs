//! Device tracking for hotplug detection

use std::collections::HashMap;

use tracing::debug;

use crate::hardware::{DeviceId, DeviceInfo};

/// Tracks discovered devices and detects additions/removals
pub struct DeviceTracker {
    /// Currently known devices, indexed by DeviceId
    known_devices: HashMap<DeviceId, DeviceInfo>,
}

impl DeviceTracker {
    pub fn new() -> Self {
        Self {
            known_devices: HashMap::new(),
        }
    }

    /// Update the tracker with a new enumeration result
    ///
    /// Returns (added_devices, removed_device_ids)
    pub fn update(&mut self, discovered: Vec<DeviceInfo>) -> (Vec<DeviceInfo>, Vec<DeviceId>) {
        let mut added = Vec::new();
        let mut new_devices = HashMap::new();

        for device in discovered {
            let device_id = device.id.clone();

            if !self.known_devices.contains_key(&device_id) {
                debug!(device_id = ?device_id, "Device added");
                added.push(device.clone());
            }

            new_devices.insert(device_id, device);
        }

        let removed: Vec<DeviceId> = self
            .known_devices
            .keys()
            .filter(|id| !new_devices.contains_key(id))
            .cloned()
            .collect();

        for device_id in &removed {
            debug!(device_id = ?device_id, "Device removed");
        }

        self.known_devices = new_devices;

        (added, removed)
    }

    /// Clear all tracked devices
    pub fn clear(&mut self) {
        self.known_devices.clear();
    }
}

impl Default for DeviceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{
        pool::TunerId,
        types::{Backend, TunerInfo},
    };

    fn create_test_device(driver: &str, serial: &str) -> DeviceInfo {
        let device_id = DeviceId::Driver {
            backend: Backend::Soapy,
            driver: driver.to_string(),
            serial: serial.to_string(),
        };

        DeviceInfo {
            id: device_id.clone(),
            label: format!("Test {} {}", driver, serial),
            tuners: vec![TunerInfo {
                id: TunerId::new(device_id, 0),
                label: format!("Test {} {}", driver, serial),
                mode: String::new(),
                antenna: None,
            }],
        }
    }

    #[test]
    fn test_initial_enumeration() {
        let mut tracker = DeviceTracker::new();

        let devices = vec![
            create_test_device("rtlsdr", "001"),
            create_test_device("sdrplay", "002"),
        ];

        let (added, removed) = tracker.update(devices);

        assert_eq!(added.len(), 2);
        assert_eq!(removed.len(), 0);
    }

    #[test]
    fn test_device_addition() {
        let mut tracker = DeviceTracker::new();

        let devices1 = vec![create_test_device("rtlsdr", "001")];
        tracker.update(devices1);

        let devices2 = vec![
            create_test_device("rtlsdr", "001"),
            create_test_device("sdrplay", "002"),
        ];
        let (added, removed) = tracker.update(devices2);

        assert_eq!(added.len(), 1);
        assert_eq!(removed.len(), 0);
        assert_eq!(added[0].id, DeviceId::from_serial("sdrplay", "002"));
    }

    #[test]
    fn test_device_removal() {
        let mut tracker = DeviceTracker::new();

        let devices1 = vec![
            create_test_device("rtlsdr", "001"),
            create_test_device("sdrplay", "002"),
        ];
        tracker.update(devices1);

        let devices2 = vec![create_test_device("sdrplay", "002")];
        let (added, removed) = tracker.update(devices2);

        assert_eq!(added.len(), 0);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], DeviceId::from_serial("rtlsdr", "001"));
    }

    #[test]
    fn test_device_replacement() {
        let mut tracker = DeviceTracker::new();

        let devices1 = vec![create_test_device("rtlsdr", "001")];
        tracker.update(devices1);

        let devices2 = vec![create_test_device("rtlsdr", "002")];
        let (added, removed) = tracker.update(devices2);

        assert_eq!(added.len(), 1);
        assert_eq!(removed.len(), 1);
        assert_eq!(added[0].id, DeviceId::from_serial("rtlsdr", "002"));
        assert_eq!(removed[0], DeviceId::from_serial("rtlsdr", "001"));
    }

    #[test]
    fn test_no_changes() {
        let mut tracker = DeviceTracker::new();

        let devices = vec![
            create_test_device("rtlsdr", "001"),
            create_test_device("sdrplay", "002"),
        ];
        tracker.update(devices.clone());

        let (added, removed) = tracker.update(devices);

        assert_eq!(added.len(), 0);
        assert_eq!(removed.len(), 0);
    }
}
