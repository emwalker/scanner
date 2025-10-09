use crate::sdr;
use std::collections::HashMap;
use tracing::debug;

pub trait DeviceEnumerator: Send {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>, Box<dyn std::error::Error>>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourcePriority {
    UsbInspection = 1,
    Backend = 2,
}

pub struct MultiEnumerator {
    pub enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)>,
}

impl MultiEnumerator {
    pub fn enumerate(&self) -> Vec<sdr::DeviceInfo> {
        let mut devices_by_id = HashMap::new();

        for (enumerator, priority) in &self.enumerators {
            match enumerator.enumerate() {
                Ok(devs) => {
                    debug!(
                        enumerator = enumerator.name(),
                        count = devs.len(),
                        "enumerated devices"
                    );
                    for device in devs {
                        let id = device.id.clone();
                        devices_by_id
                            .entry(id)
                            .and_modify(|(existing_dev, existing_priority)| {
                                if priority > existing_priority {
                                    *existing_dev = device.clone();
                                    *existing_priority = *priority;
                                }
                            })
                            .or_insert((device, *priority));
                    }
                }
                Err(e) => {
                    debug!(
                        enumerator = enumerator.name(),
                        error = %e,
                        "enumeration failed"
                    );
                }
            }
        }

        let mut devices: Vec<_> = devices_by_id
            .into_iter()
            .map(|(_, (device, _))| device)
            .collect();
        devices.sort_by(|a, b| a.id.cmp(&b.id));
        devices
    }
}

pub struct BackendEnumerator {
    pub backends: Vec<Box<dyn sdr::Backend>>,
}

impl DeviceEnumerator for BackendEnumerator {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>, Box<dyn std::error::Error>> {
        let mut devices = Vec::new();
        for backend in &self.backends {
            if let Ok(devs) = backend.enumerate_devices() {
                devices.extend(devs);
            }
        }
        Ok(devices)
    }

    fn name(&self) -> &str {
        "backend"
    }
}

#[cfg(target_os = "linux")]
pub struct UsbEnumerator {
    known_devices: HashMap<(u16, u16), &'static str>,
}

#[cfg(target_os = "linux")]
impl UsbEnumerator {
    pub fn new() -> Self {
        Self::with_database(Self::default_database())
    }

    pub fn with_database(known_devices: HashMap<(u16, u16), &'static str>) -> Self {
        Self { known_devices }
    }

    fn default_database() -> HashMap<(u16, u16), &'static str> {
        let mut db = HashMap::new();
        // RTL-SDR devices
        db.insert((0x0bda, 0x2838), "RTL-SDR");
        db.insert((0x0bda, 0x2832), "RTL-SDR");
        // HackRF
        db.insert((0x1d50, 0x6089), "HackRF One");
        // AirSpy
        db.insert((0x1d50, 0x60a1), "AirSpy");
        db.insert((0x1d50, 0x60a6), "AirSpy HF+");
        db.insert((0x03eb, 0x800c), "AirSpy Mini");
        // LimeSDR
        db.insert((0x1d50, 0x6108), "LimeSDR-USB");
        db.insert((0x0403, 0x601f), "LimeSDR-Mini");
        // PlutoSDR
        db.insert((0x0456, 0xb673), "PlutoSDR");
        // BladeRF
        db.insert((0x2cf0, 0x5246), "BladeRF");
        db.insert((0x1d50, 0x6066), "BladeRF 2.0");
        db
    }

    fn try_extract_device_info(&self, device: &udev::Device) -> Option<sdr::DeviceInfo> {
        let (vid_str, pid_str) = (
            device.attribute_value("idVendor")?,
            device.attribute_value("idProduct")?,
        );

        let vid = u16::from_str_radix(vid_str.to_str()?, 16).ok()?;
        let pid = u16::from_str_radix(pid_str.to_str()?, 16).ok()?;

        let model = self.known_devices.get(&(vid, pid))?;

        let serial = device
            .property_value("ID_SERIAL_SHORT")
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let bus = device
            .property_value("BUSNUM")
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let port = device
            .property_value("DEVNUM")
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        Some(sdr::DeviceInfo {
            id: sdr::DeviceId::Usb {
                vid,
                pid,
                serial: serial.to_string(),
                bus_port: format!("{}-{}", bus, port),
            },
            label: format!("{} (USB VID={:04x} PID={:04x})", model, vid, pid),
        })
    }
}

#[cfg(target_os = "linux")]
impl DeviceEnumerator for UsbEnumerator {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>, Box<dyn std::error::Error>> {
        let mut devices = Vec::new();
        let mut enumerator = udev::Enumerator::new()?;
        enumerator.match_subsystem("usb")?;

        for device in enumerator.scan_devices()? {
            if let Some(device_info) = self.try_extract_device_info(&device) {
                devices.push(device_info);
            }
        }

        Ok(devices)
    }

    fn name(&self) -> &str {
        "usb"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdr::Mock;

    #[test]
    fn test_backend_enumerator() {
        let backends: Vec<Box<dyn crate::sdr::Backend>> = vec![Box::new(Mock)];
        let enumerator = BackendEnumerator { backends };

        let devices = enumerator.enumerate().unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(enumerator.name(), "backend");
    }

    #[test]
    fn test_multi_enumerator_single_source() {
        let backends: Vec<Box<dyn crate::sdr::Backend>> = vec![Box::new(Mock)];
        let enumerator = MultiEnumerator {
            enumerators: vec![(
                Box::new(BackendEnumerator { backends }),
                SourcePriority::Backend,
            )],
        };

        let devices = enumerator.enumerate();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_multi_enumerator_priority() {
        let mock1: Vec<Box<dyn crate::sdr::Backend>> = vec![Box::new(Mock)];
        let mock2: Vec<Box<dyn crate::sdr::Backend>> = vec![Box::new(Mock)];

        let enumerator = MultiEnumerator {
            enumerators: vec![
                (
                    Box::new(BackendEnumerator { backends: mock1 }),
                    SourcePriority::UsbInspection,
                ),
                (
                    Box::new(BackendEnumerator { backends: mock2 }),
                    SourcePriority::Backend,
                ),
            ],
        };

        let devices = enumerator.enumerate();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_multi_enumerator_deterministic_ordering() {
        let backends: Vec<Box<dyn crate::sdr::Backend>> = vec![Box::new(Mock)];
        let enumerator = MultiEnumerator {
            enumerators: vec![(
                Box::new(BackendEnumerator { backends }),
                SourcePriority::Backend,
            )],
        };

        let devices1 = enumerator.enumerate();
        let devices2 = enumerator.enumerate();

        assert_eq!(devices1.len(), devices2.len());
        for (d1, d2) in devices1.iter().zip(devices2.iter()) {
            assert_eq!(d1.id, d2.id);
        }
    }

    #[test]
    fn test_source_priority_ordering() {
        assert!(SourcePriority::Backend > SourcePriority::UsbInspection);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_usb_enumerator_empty_database() {
        let enumerator = UsbEnumerator::with_database(HashMap::new());
        assert_eq!(enumerator.name(), "usb");

        let result = enumerator.enumerate();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_usb_enumerator_default_database() {
        let db = UsbEnumerator::default_database();

        assert!(db.contains_key(&(0x0bda, 0x2838)));
        assert!(db.contains_key(&(0x1d50, 0x6089)));
        assert!(db.contains_key(&(0x1d50, 0x60a1)));
        assert!(db.contains_key(&(0x1d50, 0x6108)));
        assert!(db.contains_key(&(0x0456, 0xb673)));
        assert!(db.contains_key(&(0x2cf0, 0x5246)));
    }
}
