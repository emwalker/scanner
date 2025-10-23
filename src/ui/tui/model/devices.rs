//! Device management methods

use tracing::debug;

use super::{state::Model, types::TunerState};
use crate::hardware::{DeviceId, DeviceInfo};

impl Model {
    /// Extract serial number from a device ID for matching purposes
    fn device_serial(device_id: &DeviceId) -> Option<String> {
        match device_id {
            DeviceId::Driver { serial, .. } => Some(serial.clone()),
            DeviceId::Usb { serial, .. } => Some(serial.clone()),
        }
    }

    /// Check if a device with the same serial already exists
    fn find_device_by_serial(&self, serial: &str) -> Option<DeviceId> {
        self.devices
            .keys()
            .find(|id| Self::device_serial(id).as_deref() == Some(serial))
            .cloned()
    }

    /// Add a newly discovered device
    ///
    /// If a device with the same serial number already exists, prefer USB labels
    /// over backend labels while keeping the functional device ID for operations.
    pub fn add_device(&mut self, device: DeviceInfo) {
        self.mark_dirty();
        let device_id = device.id.clone();
        let serial = Self::device_serial(&device_id);

        if let Some(serial_str) = &serial
            && let Some(existing_id) = self.find_device_by_serial(serial_str)
        {
            match (&device_id, &existing_id) {
                (DeviceId::Usb { .. }, DeviceId::Driver { .. }) => {
                    debug!(
                        usb_label = %device.label,
                        existing_id = ?existing_id,
                        "Enriching backend device with USB label"
                    );

                    if let Some(existing_device) = self.devices.get_mut(&existing_id) {
                        existing_device.label = device.label.clone();

                        for (usb_tuner, existing_tuner) in
                            device.tuners.iter().zip(existing_device.tuners.iter_mut())
                        {
                            existing_tuner.label = usb_tuner.label.clone();

                            let old_tuner = self
                                .tuners
                                .iter()
                                .find(|t| t.id == existing_tuner.id)
                                .cloned();
                            if let Some(old) = old_tuner {
                                self.tuners.remove(&old);
                                self.tuners.insert(crate::ui::tui::model::TunerInfo {
                                    id: existing_tuner.id.clone(),
                                    label: usb_tuner.label.clone(),
                                });
                                debug!(
                                    tuner_id = ?existing_tuner.id,
                                    new_label = %usb_tuner.label,
                                    "Updated tuner label from USB"
                                );
                            }
                        }
                    }
                    return;
                }
                (DeviceId::Driver { .. }, DeviceId::Usb { .. }) => {
                    debug!(
                        driver_label = %device.label,
                        usb_id = ?existing_id,
                        "Backend device arrived after USB - using USB label from existing device"
                    );

                    if let Some(usb_device) = self.devices.remove(&existing_id) {
                        for usb_tuner in &usb_device.tuners {
                            let old_tuner = self
                                .tuners
                                .iter()
                                .find(|t| {
                                    if let (
                                        DeviceId::Usb {
                                            serial: usb_serial, ..
                                        },
                                        DeviceId::Driver {
                                            serial: driver_serial,
                                            ..
                                        },
                                    ) = (&t.id.device_id, &device_id)
                                    {
                                        usb_serial == driver_serial
                                            && t.id.channel_index == usb_tuner.id.channel_index
                                    } else {
                                        false
                                    }
                                })
                                .cloned();

                            if let Some(old) = old_tuner {
                                self.tuners.remove(&old);
                                debug!(
                                    removed_tuner = ?old.id,
                                    "Removed USB tuner before replacing with enriched driver tuner"
                                );
                            }
                        }

                        let mut enriched_device = device.clone();
                        enriched_device.label = usb_device.label.clone();

                        // Keep driver tuner labels (they have antenna info from SoapySDR)
                        // USB enumeration cannot query antenna information

                        for tuner in &enriched_device.tuners {
                            let tuner_info = crate::ui::tui::model::TunerInfo {
                                id: tuner.id.clone(),
                                label: tuner.label.clone(),
                            };
                            self.tuners.insert(tuner_info.clone());
                            debug!(
                                tuner_id = ?tuner_info.id,
                                label = %tuner_info.label,
                                "Populating tuner with driver label (includes antenna info)"
                            );
                        }

                        self.devices.insert(device_id, enriched_device);
                    }
                    return;
                }
                _ => {}
            }
        }

        for tuner in &device.tuners {
            let tuner_info = crate::ui::tui::model::TunerInfo {
                id: tuner.id.clone(),
                label: tuner.label.clone(),
            };
            if self.tuners.insert(tuner_info.clone()) {
                debug!(
                    tuner_id = ?tuner_info.id,
                    label = %tuner_info.label,
                    "Populating tuner from discovered device"
                );
            }
        }

        self.devices.insert(device_id, device);
    }

    /// Remove a device and all its tuners
    pub fn remove_device(&mut self, device_id: &crate::hardware::DeviceId) {
        if self.devices.remove(device_id).is_some() {
            self.tuners.retain(|t| &t.id.device_id != device_id);
            debug!(device_id = ?device_id, "Device and all its tuners removed from TUI model");
            self.mark_dirty();
        }
    }

    /// Get the state of a specific tuner based on pool status
    pub fn tuner_state(&self, tuner_id: &crate::hardware::pool::TunerId) -> TunerState {
        use tracing::debug;

        if let Some(tuner) = self.pool_info.get(tuner_id) {
            let result = match (&tuner.state, &tuner.activity) {
                (crate::hardware::pool::TunerState::Available, _) => TunerState::Available,
                (crate::hardware::pool::TunerState::Allocated, Some(activity)) => match activity {
                    crate::hardware::pool::TunerActivity::Scanning => TunerState::Scanning,
                    crate::hardware::pool::TunerActivity::Listening => TunerState::Listening,
                    crate::hardware::pool::TunerActivity::Other => TunerState::Available,
                },
                (crate::hardware::pool::TunerState::Allocated, None) => TunerState::Available,
            };

            debug!(
                tuner_id = ?tuner_id,
                pool_state = ?tuner.state,
                pool_activity = ?tuner.activity,
                result_state = ?result,
                "tuner_state: Found in pool_info"
            );

            result
        } else {
            debug!(
                tuner_id = ?tuner_id,
                pool_info_size = self.pool_info.len(),
                "tuner_state: NOT found in pool_info, defaulting to Available"
            );
            TunerState::Available
        }
    }

    /// Get count of discovered tuners
    pub fn device_count(&self) -> usize {
        self.tuners.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{
        DeviceId, DeviceInfo,
        pool::TunerId,
        types::{Backend, TunerInfo},
    };

    fn create_usb_device(serial: &str, label: &str) -> DeviceInfo {
        let device_id = DeviceId::Usb {
            vid: 0x0bda,
            pid: 0x2838,
            serial: serial.to_string(),
            bus_port: "1-2".to_string(),
        };

        DeviceInfo {
            id: device_id.clone(),
            label: label.to_string(),
            tuners: vec![TunerInfo {
                id: TunerId::new(device_id, 0),
                label: label.to_string(),
                mode: String::new(),
                antenna: None,
            }],
        }
    }

    fn create_driver_device(driver: &str, serial: &str, label: &str) -> DeviceInfo {
        let device_id = DeviceId::Driver {
            backend: Backend::Soapy,
            driver: driver.to_string(),
            serial: serial.to_string(),
        };

        DeviceInfo {
            id: device_id.clone(),
            label: label.to_string(),
            tuners: vec![TunerInfo {
                id: TunerId::new(device_id, 0),
                label: label.to_string(),
                mode: String::new(),
                antenna: None,
            }],
        }
    }

    #[test]
    fn test_usb_enrichment_usb_first() {
        let mut model = Model::default();

        let usb_device = create_usb_device("00000001", "RTLSDRBlog Blog V4 :: 00000001");
        model.add_device(usb_device);

        assert_eq!(model.tuners.len(), 1);
        assert_eq!(model.devices.len(), 1);

        let driver_device = create_driver_device(
            "rtlsdr",
            "00000001",
            "Generic RTL2832U OEM :: 00000001 (rtlsdr:00000001)",
        );
        model.add_device(driver_device);

        assert_eq!(
            model.tuners.len(),
            1,
            "Should have exactly one tuner after deduplication"
        );
        assert_eq!(
            model.devices.len(),
            1,
            "Should have exactly one device after deduplication"
        );

        let tuner = model.tuners.iter().next().unwrap();
        assert_eq!(
            tuner.label, "Generic RTL2832U OEM :: 00000001 (rtlsdr:00000001)",
            "Should use driver label (may include antenna info)"
        );
        assert!(
            matches!(tuner.id.device_id, DeviceId::Driver { .. }),
            "Should use driver device ID"
        );

        let device = model.devices.values().next().unwrap();
        assert_eq!(
            device.label, "RTLSDRBlog Blog V4 :: 00000001",
            "Device should use USB label"
        );
        assert!(
            matches!(device.id, DeviceId::Driver { .. }),
            "Device should use driver ID"
        );
    }

    #[test]
    fn test_usb_enrichment_driver_first() {
        let mut model = Model::default();

        let driver_device = create_driver_device(
            "rtlsdr",
            "00000001",
            "Generic RTL2832U OEM :: 00000001 (rtlsdr:00000001)",
        );
        model.add_device(driver_device);

        assert_eq!(model.tuners.len(), 1);
        assert_eq!(model.devices.len(), 1);

        let usb_device = create_usb_device("00000001", "RTLSDRBlog Blog V4 :: 00000001");
        model.add_device(usb_device);

        assert_eq!(
            model.tuners.len(),
            1,
            "Should have exactly one tuner after enrichment"
        );
        assert_eq!(
            model.devices.len(),
            1,
            "Should have exactly one device after enrichment"
        );

        let tuner = model.tuners.iter().next().unwrap();
        assert_eq!(
            tuner.label, "RTLSDRBlog Blog V4 :: 00000001",
            "Should use USB label"
        );

        let device = model.devices.values().next().unwrap();
        assert_eq!(
            device.label, "RTLSDRBlog Blog V4 :: 00000001",
            "Device label should be updated to USB label"
        );
    }

    #[test]
    fn test_no_enrichment_different_serials() {
        let mut model = Model::default();

        let usb_device = create_usb_device("00000001", "RTLSDRBlog Blog V4 :: 00000001");
        model.add_device(usb_device);

        let driver_device = create_driver_device(
            "rtlsdr",
            "00000002",
            "Generic RTL2832U OEM :: 00000002 (rtlsdr:00000002)",
        );
        model.add_device(driver_device);

        assert_eq!(
            model.tuners.len(),
            2,
            "Should have two separate tuners for different devices"
        );
        assert_eq!(model.devices.len(), 2, "Should have two separate devices");
    }

    #[test]
    fn test_multiple_tuners_enrichment() {
        let mut model = Model::default();

        let usb_id = DeviceId::Usb {
            vid: 0x0bda,
            pid: 0x2838,
            serial: "00000001".to_string(),
            bus_port: "1-2".to_string(),
        };

        let usb_device = DeviceInfo {
            id: usb_id.clone(),
            label: "RTLSDRBlog Blog V4 :: 00000001".to_string(),
            tuners: vec![
                TunerInfo {
                    id: TunerId::new(usb_id.clone(), 0),
                    label: "RTLSDRBlog Blog V4 Ch0 :: 00000001".to_string(),
                    mode: String::new(),
                    antenna: None,
                },
                TunerInfo {
                    id: TunerId::new(usb_id, 1),
                    label: "RTLSDRBlog Blog V4 Ch1 :: 00000001".to_string(),
                    mode: String::new(),
                    antenna: None,
                },
            ],
        };
        model.add_device(usb_device);

        let driver_id = DeviceId::Driver {
            backend: Backend::Soapy,
            driver: "rtlsdr".to_string(),
            serial: "00000001".to_string(),
        };

        let driver_device = DeviceInfo {
            id: driver_id.clone(),
            label: "Generic RTL2832U OEM :: 00000001".to_string(),
            tuners: vec![
                TunerInfo {
                    id: TunerId::new(driver_id.clone(), 0),
                    label: "Generic RTL2832U OEM Ch0 :: 00000001".to_string(),
                    mode: String::new(),
                    antenna: None,
                },
                TunerInfo {
                    id: TunerId::new(driver_id, 1),
                    label: "Generic RTL2832U OEM Ch1 :: 00000001".to_string(),
                    mode: String::new(),
                    antenna: None,
                },
            ],
        };
        model.add_device(driver_device);

        assert_eq!(
            model.tuners.len(),
            2,
            "Should have exactly two tuners after deduplication"
        );

        let labels: Vec<_> = model.tuners.iter().map(|t| t.label.as_str()).collect();
        assert!(
            labels.contains(&"Generic RTL2832U OEM Ch0 :: 00000001"),
            "Should use driver label for Ch0"
        );
        assert!(
            labels.contains(&"Generic RTL2832U OEM Ch1 :: 00000001"),
            "Should use driver label for Ch1"
        );
    }
}
