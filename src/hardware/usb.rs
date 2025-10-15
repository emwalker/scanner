//! USB device inspection backend
//!
//! This backend uses udev to inspect USB devices and extract manufacturer/product
//! labels. It provides device metadata but cannot open devices - actual device
//! access is handled by other backends (Soapy, etc).

use super::{
    Backend, DeviceId, DeviceInfo, DeviceTrait, pool::TunerId, streaming::StreamingDevice,
    types::TunerInfo,
};
use crate::core::types::{Result, ScannerError};
use std::collections::HashMap;
use tracing::debug;

pub struct Usb {
    known_devices: HashMap<(u16, u16), &'static str>,
}

impl Default for Usb {
    fn default() -> Self {
        Self::new()
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

impl Usb {
    pub fn new() -> Self {
        Self {
            known_devices: Self::default_database(),
        }
    }

    fn default_database() -> HashMap<(u16, u16), &'static str> {
        let mut db = HashMap::new();
        db.insert((0x0bda, 0x2838), "RTL-SDR");
        db.insert((0x0bda, 0x2832), "RTL-SDR");
        db.insert((0x1d50, 0x6089), "HackRF One");
        db.insert((0x1d50, 0x60a1), "AirSpy");
        db.insert((0x1d50, 0x60a6), "AirSpy HF+");
        db.insert((0x03eb, 0x800c), "AirSpy Mini");
        db.insert((0x1d50, 0x6108), "LimeSDR-USB");
        db.insert((0x0403, 0x601f), "LimeSDR-Mini");
        db.insert((0x0456, 0xb673), "PlutoSDR");
        db.insert((0x2cf0, 0x5246), "BladeRF");
        db.insert((0x1d50, 0x6066), "BladeRF 2.0");
        db.insert((0x1df7, 0x2500), "SDRplay RSP1");
        db.insert((0x1df7, 0x3000), "SDRplay RSP1A");
        db.insert((0x1df7, 0x3010), "SDRplay RSP2");
        db.insert((0x1df7, 0x3020), "SDRplay RSPduo");
        db.insert((0x1df7, 0x3030), "SDRplay RSPdx");
        db
    }

    #[cfg(target_os = "linux")]
    fn try_extract_device_info(&self, device: &udev::Device) -> Option<DeviceInfo> {
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

        let device_id = DeviceId::Usb {
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

        Some(DeviceInfo {
            id: device_id.clone(),
            label: label.clone(),
            tuners: vec![TunerInfo {
                id: TunerId::new(device_id, 0),
                label,
                mode: String::new(),
            }],
        })
    }
}

impl Backend for Usb {
    #[cfg(target_os = "linux")]
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        let mut devices = Vec::new();
        let mut enumerator = udev::Enumerator::new()
            .map_err(|e| ScannerError::Custom(format!("udev enumeration failed: {}", e)))?;

        enumerator
            .match_subsystem("usb")
            .map_err(|e| ScannerError::Custom(format!("udev subsystem match failed: {}", e)))?;

        for device in enumerator
            .scan_devices()
            .map_err(|e| ScannerError::Custom(format!("udev scan failed: {}", e)))?
        {
            if let Some(device_info) = self.try_extract_device_info(&device) {
                devices.push(device_info);
            }
        }

        debug!(device_count = devices.len(), "USB enumeration complete");
        Ok(devices)
    }

    #[cfg(not(target_os = "linux"))]
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    fn open_tuner(&self, _tuner_id: &TunerId) -> Result<Box<dyn DeviceTrait>> {
        Err(ScannerError::Custom(
            "USB backend cannot open devices - use Soapy or other hardware backend".to_string(),
        ))
    }

    fn open_streaming_tuner(&self, _tuner_id: &TunerId) -> Result<Box<dyn StreamingDevice>> {
        Err(ScannerError::Custom(
            "USB backend cannot open devices - use Soapy or other hardware backend".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "USB"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_database_includes_sdrplay_devices() {
        let db = Usb::default_database();

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
    }

    #[test]
    fn test_default_database_sdrplay_labels() {
        let db = Usb::default_database();

        assert_eq!(db.get(&(0x1df7, 0x2500)), Some(&"SDRplay RSP1"));
        assert_eq!(db.get(&(0x1df7, 0x3000)), Some(&"SDRplay RSP1A"));
        assert_eq!(db.get(&(0x1df7, 0x3010)), Some(&"SDRplay RSP2"));
        assert_eq!(db.get(&(0x1df7, 0x3020)), Some(&"SDRplay RSPduo"));
        assert_eq!(db.get(&(0x1df7, 0x3030)), Some(&"SDRplay RSPdx"));
    }

    #[test]
    fn test_default_database_sdrplay_vendor_id() {
        let db = Usb::default_database();

        let sdrplay_devices: Vec<_> = db.iter().filter(|((vid, _), _)| *vid == 0x1df7).collect();

        assert!(
            sdrplay_devices.len() >= 5,
            "Should have at least 5 SDRplay devices with VID 0x1df7"
        );
    }

    #[test]
    fn test_usb_backend_name() {
        let usb = Usb::new();
        assert_eq!(usb.name(), "USB");
    }

    #[test]
    fn test_default_database_includes_common_sdrs() {
        let db = Usb::default_database();

        assert!(
            db.contains_key(&(0x0bda, 0x2838)),
            "RTL-SDR should be present"
        );
        assert!(
            db.contains_key(&(0x1d50, 0x6089)),
            "HackRF One should be present"
        );
        assert!(
            db.contains_key(&(0x1d50, 0x60a1)),
            "AirSpy should be present"
        );
    }

    #[test]
    fn test_format_device_label_udev_database_priority() {
        let label = super::format_device_label(
            Some("SDRplay"),
            Some("RSPduo"),
            Some("Ignored Vendor"),
            Some("Ignored Product"),
            "Fallback Name",
            "2301034E34",
        );
        assert_eq!(label, "SDRplay RSPduo :: 2301034E34");
    }

    #[test]
    fn test_format_device_label_fallback_to_device_strings() {
        let label = super::format_device_label(
            None,
            None,
            Some("Device Manufacturer"),
            Some("Device Product"),
            "Fallback Name",
            "2301034E34",
        );
        assert_eq!(label, "Device Manufacturer Device Product :: 2301034E34");
    }

    #[test]
    fn test_format_device_label_fallback_to_friendly_name() {
        let label =
            super::format_device_label(None, None, None, None, "SDRplay RSPduo", "2301034E34");
        assert_eq!(label, "SDRplay RSPduo :: 2301034E34");
    }

    #[test]
    fn test_format_device_label_only_product() {
        let label =
            super::format_device_label(None, Some("RSPduo"), None, None, "Fallback", "ABC123");
        assert_eq!(label, "RSPduo :: ABC123");
    }

    #[test]
    fn test_format_device_label_only_manufacturer() {
        let label =
            super::format_device_label(Some("SDRplay"), None, None, None, "Fallback", "ABC123");
        assert_eq!(label, "SDRplay :: ABC123");
    }

    #[test]
    fn test_format_device_label_udev_overrides_device() {
        let label = super::format_device_label(
            Some("Udev Vendor"),
            None,
            Some("Device Vendor"),
            Some("Device Product"),
            "Fallback",
            "XYZ",
        );
        assert_eq!(label, "Udev Vendor Device Product :: XYZ");
    }
}
