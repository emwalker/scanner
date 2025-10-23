#![allow(dead_code)]

use std::collections::HashMap;

use tracing::{debug, info};

use crate::hardware;

pub trait DeviceEnumerator: Send {
    fn enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>>;
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
    pub fn enumerate(&self) -> Vec<hardware::DeviceInfo> {
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

pub struct DirectEnumerator {
    pub backends: Vec<Box<dyn hardware::Backend>>,
}

impl DeviceEnumerator for DirectEnumerator {
    fn enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
        let mut devices = Vec::new();
        for backend in &self.backends {
            if let Ok(devs) = backend.enumerate_devices() {
                devices.extend(devs);
            }
        }
        Ok(devices)
    }

    fn name(&self) -> &str {
        "direct"
    }
}

pub struct SubprocessEnumerator {
    backend_name: String,
    parent_log_file: Option<String>,
}

impl SubprocessEnumerator {
    pub fn new(backend_name: String, parent_log_file: Option<String>) -> Self {
        Self {
            backend_name,
            parent_log_file,
        }
    }

    fn spawn_and_enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
        use std::{
            env,
            os::unix::net::UnixStream,
            path::Path,
            process::{Command, Stdio},
            thread,
            time::{Duration, Instant, SystemTime, UNIX_EPOCH},
        };

        use crate::ipc::{ControlChannel, ControlMessage, UnixControlChannel};

        info!(backend = %self.backend_name, "Starting subprocess enumeration");

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let socket_path = format!(
            "/tmp/scanner-enum-{}-{}-{}.sock",
            self.backend_name,
            std::process::id(),
            timestamp
        );

        debug!(socket_path = %socket_path, "Generated socket path");

        use crate::cli::worker_logging::{WorkerContext, WorkerType, generate_worker_log_path};

        let worker_log_path = generate_worker_log_path(
            self.parent_log_file.as_deref(),
            WorkerType::Enumeration,
            &WorkerContext {
                device_id: None,
                timestamp: Some(timestamp),
                backend: Some(self.backend_name.clone()),
            },
        );

        let mut cmd = Command::new(env::current_exe()?);
        cmd.arg("worker")
            .arg("enumerate")
            .arg("--backend")
            .arg(&self.backend_name)
            .arg("--socket-path")
            .arg(&socket_path);

        if let Some(log_path) = worker_log_path {
            cmd.arg("--log-file").arg(&log_path);
        }

        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        debug!("Spawning worker subprocess");
        let mut child = cmd.spawn()?;
        debug!(pid = %child.id(), "Worker subprocess spawned");

        let start = Instant::now();
        while !Path::new(&socket_path).exists() {
            if start.elapsed() > Duration::from_secs(5) {
                let _ = child.kill();
                return Err(format!(
                    "Socket creation timeout for backend '{}'",
                    self.backend_name
                )
                .into());
            }
            thread::sleep(Duration::from_millis(10));
        }

        debug!("Connecting to worker socket");
        let stream = UnixStream::connect(&socket_path)?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut channel = UnixControlChannel::new(stream);

        debug!("Waiting for response from worker");
        let message = channel.recv()?;

        match message {
            ControlMessage::DeviceList { devices } => {
                debug!(device_count = devices.len(), "Received device list");
                let _ = child.wait();
                Ok(devices)
            }
            ControlMessage::Error { message, .. } => {
                debug!(error = %message, "Received error from worker");
                let _ = child.wait();
                Err(format!(
                    "Enumeration error from '{}': {}",
                    self.backend_name, message
                )
                .into())
            }
            _ => {
                debug!("Received unexpected message type");
                let _ = child.kill();
                Err(format!("Unexpected message from '{}' worker", self.backend_name).into())
            }
        }
    }
}

impl DeviceEnumerator for SubprocessEnumerator {
    fn enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
        self.spawn_and_enumerate()
    }

    fn name(&self) -> &str {
        &self.backend_name
    }
}

fn format_device_label(
    udev_vendor: Option<&str>,
    udev_product: Option<&str>,
    device_vendor: Option<&str>,
    device_product: Option<&str>,
    friendly_name: &str,
    serial: &str,
) -> String {
    let manufacturer = udev_vendor.or(device_vendor);
    let product = udev_product.or(device_product);

    match (manufacturer, product) {
        (Some(mfg), Some(prod)) => format!("{} {} :: {}", mfg, prod, serial),
        (None, Some(prod)) => format!("{} :: {}", prod, serial),
        (Some(mfg), None) => format!("{} :: {}", mfg, serial),
        (None, None) => format!("{} :: {}", friendly_name, serial),
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
        // SDRplay
        db.insert((0x1df7, 0x2500), "SDRplay RSP1");
        db.insert((0x1df7, 0x3000), "SDRplay RSP1A");
        db.insert((0x1df7, 0x3010), "SDRplay RSP2");
        db.insert((0x1df7, 0x3020), "SDRplay RSPduo");
        db.insert((0x1df7, 0x3030), "SDRplay RSPdx");
        db
    }

    fn try_extract_device_info(&self, device: &udev::Device) -> Option<hardware::DeviceInfo> {
        let (vid_str, pid_str) = (
            device.attribute_value("idVendor")?,
            device.attribute_value("idProduct")?,
        );

        let vid = u16::from_str_radix(vid_str.to_str()?, 16).ok()?;
        let pid = u16::from_str_radix(pid_str.to_str()?, 16).ok()?;

        let friendly_name = self.known_devices.get(&(vid, pid))?;

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

        let udev_vendor = device
            .property_value("ID_VENDOR_FROM_DATABASE")
            .and_then(|s| s.to_str());

        let udev_product = device
            .property_value("ID_MODEL_FROM_DATABASE")
            .and_then(|s| s.to_str());

        let device_vendor = device
            .attribute_value("manufacturer")
            .and_then(|s| s.to_str());

        let device_product = device.attribute_value("product").and_then(|s| s.to_str());

        let label = format_device_label(
            udev_vendor,
            udev_product,
            device_vendor,
            device_product,
            friendly_name,
            serial,
        );

        let device_id = hardware::DeviceId::Usb {
            vid,
            pid,
            serial: serial.to_string(),
            bus_port: format!("{}-{}", bus, port),
        };

        debug!(
            vid = format!("{:04x}", vid),
            pid = format!("{:04x}", pid),
            serial = serial,
            label = %label,
            "USB device detected"
        );

        Some(hardware::DeviceInfo {
            id: device_id.clone(),
            label: label.clone(),
            tuners: vec![hardware::types::TunerInfo {
                id: hardware::pool::TunerId::new(device_id, 0),
                label,
                mode: String::new(),
                antenna: None,
            }],
        })
    }
}

#[cfg(target_os = "linux")]
impl DeviceEnumerator for UsbEnumerator {
    fn enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
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
    use crate::hardware::Mock;

    #[test]
    fn test_direct_enumerator() {
        let backends: Vec<Box<dyn crate::hardware::Backend>> = vec![Box::new(Mock)];
        let enumerator = DirectEnumerator { backends };

        let devices = enumerator.enumerate().unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(enumerator.name(), "direct");
    }

    #[test]
    fn test_multi_enumerator_single_source() {
        let backends: Vec<Box<dyn crate::hardware::Backend>> = vec![Box::new(Mock)];
        let enumerator = MultiEnumerator {
            enumerators: vec![(
                Box::new(DirectEnumerator { backends }),
                SourcePriority::Backend,
            )],
        };

        let devices = enumerator.enumerate();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_multi_enumerator_priority() {
        let mock1: Vec<Box<dyn crate::hardware::Backend>> = vec![Box::new(Mock)];
        let mock2: Vec<Box<dyn crate::hardware::Backend>> = vec![Box::new(Mock)];

        let enumerator = MultiEnumerator {
            enumerators: vec![
                (
                    Box::new(DirectEnumerator { backends: mock1 }),
                    SourcePriority::UsbInspection,
                ),
                (
                    Box::new(DirectEnumerator { backends: mock2 }),
                    SourcePriority::Backend,
                ),
            ],
        };

        let devices = enumerator.enumerate();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_multi_enumerator_deterministic_ordering() {
        let backends: Vec<Box<dyn crate::hardware::Backend>> = vec![Box::new(Mock)];
        let enumerator = MultiEnumerator {
            enumerators: vec![(
                Box::new(DirectEnumerator { backends }),
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

    #[cfg(target_os = "linux")]
    #[test]
    fn test_usb_enumerator_includes_sdrplay_devices() {
        let db = UsbEnumerator::default_database();

        assert!(
            db.contains_key(&(0x1df7, 0x2500)),
            "SDRplay RSP1 should be in database"
        );
        assert!(
            db.contains_key(&(0x1df7, 0x3000)),
            "SDRplay RSP1A should be in database"
        );
        assert!(
            db.contains_key(&(0x1df7, 0x3010)),
            "SDRplay RSP2 should be in database"
        );
        assert!(
            db.contains_key(&(0x1df7, 0x3020)),
            "SDRplay RSPduo should be in database"
        );
        assert!(
            db.contains_key(&(0x1df7, 0x3030)),
            "SDRplay RSPdx should be in database"
        );

        assert_eq!(db.get(&(0x1df7, 0x2500)), Some(&"SDRplay RSP1"));
        assert_eq!(db.get(&(0x1df7, 0x3000)), Some(&"SDRplay RSP1A"));
        assert_eq!(db.get(&(0x1df7, 0x3010)), Some(&"SDRplay RSP2"));
        assert_eq!(db.get(&(0x1df7, 0x3020)), Some(&"SDRplay RSPduo"));
        assert_eq!(db.get(&(0x1df7, 0x3030)), Some(&"SDRplay RSPdx"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_usb_enumerator_sdrplay_vendor_id() {
        let db = UsbEnumerator::default_database();

        let sdrplay_devices: Vec<_> = db.iter().filter(|((vid, _), _)| *vid == 0x1df7).collect();

        assert!(
            sdrplay_devices.len() >= 5,
            "Should have at least 5 SDRplay devices with VID 0x1df7"
        );
    }

    #[test]
    fn test_format_device_label_udev_database_priority() {
        let label = super::format_device_label(
            Some("SDRplay"),
            Some("RSPduo"),
            Some("Ignored"),
            Some("Ignored"),
            "Fallback",
            "2301034E34",
        );
        assert_eq!(label, "SDRplay RSPduo :: 2301034E34");
    }

    #[test]
    fn test_format_device_label_fallback_to_device_strings() {
        let label = super::format_device_label(
            None,
            None,
            Some("Device Vendor"),
            Some("Device Product"),
            "Fallback",
            "ABC123",
        );
        assert_eq!(label, "Device Vendor Device Product :: ABC123");
    }

    #[test]
    fn test_format_device_label_fallback_to_friendly_name() {
        let label = super::format_device_label(None, None, None, None, "SDRplay RSPduo", "XYZ");
        assert_eq!(label, "SDRplay RSPduo :: XYZ");
    }

    #[test]
    fn test_format_device_label_mixed_sources() {
        let label = super::format_device_label(
            Some("Udev Vendor"),
            None,
            None,
            Some("Device Product"),
            "Fallback",
            "123",
        );
        assert_eq!(label, "Udev Vendor Device Product :: 123");
    }
}
