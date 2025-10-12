//! Common types for SDR backend abstraction

use crate::core::types::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Tuner information within a device
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TunerInfo {
    /// Tuner identifier
    pub id: crate::hardware::pool::TunerId,
    /// Human-readable label for this tuner
    pub label: String,
    /// Mode identifier (e.g., "ST", "DT", "MA")
    pub mode: String,
}

/// Device information returned by enumeration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Stable device identifier
    pub id: DeviceId,
    /// Human-readable label
    pub label: String,
    /// List of tuners available on this device
    pub tuners: Vec<TunerInfo>,
}

impl DeviceInfo {
    /// Look up a tuner by TunerId
    pub fn tuner(&self, tuner_id: &crate::hardware::pool::TunerId) -> Option<&TunerInfo> {
        self.tuners.iter().find(|t| &t.id == tuner_id)
    }
}

/// Backend implementation type
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Backend {
    Soapy,
    Mock,
    Usb,
    Unknown(String),
}

impl Backend {
    pub fn as_str(&self) -> &str {
        match self {
            Backend::Soapy => "soapy",
            Backend::Mock => "mock",
            Backend::Usb => "usb",
            Backend::Unknown(s) => s.as_str(),
        }
    }
}

impl FromStr for Backend {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "soapy" => Backend::Soapy,
            "mock" => Backend::Mock,
            "usb" => Backend::Usb,
            other => Backend::Unknown(other.to_string()),
        })
    }
}

/// Stable device identifier
///
/// Identifies a physical SDR device. Multi-tuner devices (like SDRplay RSPduo)
/// have a single DeviceId but multiple tuners (channels) within that device.
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DeviceId {
    /// Driver-based identification (SoapySDR drivers)
    Driver {
        backend: Backend,
        driver: String,
        serial: String,
    },
    /// USB-based identification (VID/PID + serial + physical location)
    Usb {
        vid: u16,
        pid: u16,
        serial: String,
        bus_port: String,
    },
}

impl DeviceId {
    /// Normalize driver name to lowercase for consistent DeviceId creation
    ///
    /// SoapySDR drivers can return different capitalizations:
    /// - enumeration: "sdrplay" (lowercase)
    /// - driver_key(): "SDRplay" (mixed case)
    ///
    /// This ensures both discovery and pool create identical DeviceIds.
    fn normalize_driver(driver: &str) -> String {
        driver.to_lowercase()
    }

    /// Create a device ID from backend, driver, and serial number
    pub fn from_driver(backend: Backend, driver: &str, serial: &str) -> Self {
        Self::Driver {
            backend,
            driver: Self::normalize_driver(driver),
            serial: serial.to_string(),
        }
    }

    /// Create a device ID from driver name and serial number
    ///
    /// Infers backend from driver name. The driver name is normalized to lowercase.
    pub fn from_serial(driver: &str, serial: &str) -> Self {
        let backend = match driver.to_lowercase().as_str() {
            "sdrplay" | "rtlsdr" | "lime" | "hackrf" | "airspy" => Backend::Soapy,
            "mock" => Backend::Mock,
            other => Backend::Unknown(other.to_string()),
        };
        Self::from_driver(backend, driver, serial)
    }

    /// Get backend from device ID
    pub fn backend(&self) -> Backend {
        match self {
            DeviceId::Driver { backend, .. } => backend.clone(),
            DeviceId::Usb { .. } => Backend::Usb,
        }
    }

    /// Get driver name from device ID
    pub fn driver(&self) -> &str {
        match self {
            DeviceId::Driver { driver, .. } => driver.as_str(),
            DeviceId::Usb { .. } => "usb",
        }
    }

    /// Get a string representation suitable for logging
    pub fn as_str(&self) -> String {
        match self {
            DeviceId::Driver { driver, serial, .. } => format!("{}:{}", driver, serial),
            DeviceId::Usb {
                vid,
                pid,
                serial,
                bus_port,
            } => format!("usb:{:04x}:{:04x}:{}:{}", vid, pid, serial, bus_port),
        }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Comprehensive device capabilities
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Device identifier
    pub device_id: DeviceId,

    /// Frequency ranges (min, max) in Hz
    pub rx_frequency_ranges: Vec<(f64, f64)>,

    /// Sample rate ranges (min, max) in Hz
    pub rx_sample_rate_ranges: Vec<(f64, f64)>,

    /// Gain range (min, max) in dB
    pub gain_range: (f64, f64),

    /// Device has automatic gain control
    pub has_agc: bool,

    /// Available antenna options
    pub antenna_options: Vec<String>,

    /// Number of RX channels
    pub channels: usize,

    /// Maximum bandwidth in Hz
    pub max_bandwidth: f64,

    /// Typical latency in microseconds (for device pool allocation)
    pub typical_latency_us: u64,
}

impl Capabilities {
    /// Create capabilities with basic SDR defaults
    ///
    /// This is used by both SoapySDR (which queries actual hardware) and Mock backend.
    /// The driver and serial parameters ensure the DeviceId matches what enumeration returned.
    fn with_device_id(driver: &str, serial: &str) -> Self {
        Self {
            device_id: DeviceId::from_serial(driver, serial),
            rx_frequency_ranges: vec![(24e6, 1766e6)],
            rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
            gain_range: (0.0, 48.0),
            has_agc: true,
            antenna_options: vec!["RX".to_string()],
            channels: 1,
            max_bandwidth: 2_400_000.0,
            typical_latency_us: 50,
        }
    }

    /// Create capabilities for mock devices
    ///
    /// Uses default SDR capabilities for testing without hardware.
    pub fn for_mock(driver: &str, serial: &str) -> Self {
        Self::with_device_id(driver, serial)
    }

    /// Create capabilities from a DeviceId without opening the device
    ///
    /// Used in subprocess mode where we need to populate the pool with device
    /// metadata before the subprocess opens the actual device.
    /// Returns default capabilities - actual hardware capabilities will be
    /// queried by the subprocess when it opens the device.
    pub fn for_device(device_id: &DeviceId) -> Self {
        match device_id {
            DeviceId::Driver { driver, serial, .. } => Self::with_device_id(driver, serial),
            DeviceId::Usb { .. } => {
                // USB devices need driver-specific handling
                // For now, return defaults
                Self::with_device_id("unknown", "unknown")
            }
        }
    }

    /// Create capabilities from a SoapySDR device
    ///
    /// Uses the provided driver and serial for the DeviceId to ensure consistency
    /// with enumeration results.
    pub fn from_soapy_device(
        device: &soapysdr::Device,
        driver: &str,
        serial: &str,
    ) -> Result<Self> {
        let mut caps = Self::with_device_id(driver, serial);

        caps.rx_frequency_ranges = device
            .frequency_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        caps.rx_sample_rate_ranges = device
            .get_sample_rate_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        let gain_range = device.gain_range(soapysdr::Direction::Rx, 0)?;
        caps.gain_range = (gain_range.minimum, gain_range.maximum);

        caps.max_bandwidth = caps
            .rx_sample_rate_ranges
            .last()
            .map(|r| r.1)
            .unwrap_or(0.0);

        caps.typical_latency_us = match driver {
            "rtlsdr" => 50,   // Fast USB2.0 device
            "sdrplay" => 100, // Moderate latency
            "hackrf" => 75,   // Fast USB2.0
            "lime" => 150,    // Higher latency
            _ => 100,         // Default estimate
        };

        caps.has_agc = device
            .has_gain_mode(soapysdr::Direction::Rx, 0)
            .unwrap_or(false);

        caps.antenna_options = device
            .antennas(soapysdr::Direction::Rx, 0)
            .unwrap_or_default();

        caps.channels = device.num_channels(soapysdr::Direction::Rx).unwrap_or(1);

        Ok(caps)
    }

    /// Check if device supports a given frequency
    pub fn supports_frequency(&self, freq: f64) -> bool {
        self.rx_frequency_ranges
            .iter()
            .any(|(min, max)| freq >= *min && freq <= *max)
    }

    /// Check if device supports a given sample rate
    pub fn supports_sample_rate(&self, rate: f64) -> bool {
        self.rx_sample_rate_ranges
            .iter()
            .any(|(min, max)| rate >= *min && rate <= *max)
    }

    /// Check if this device can handle a specific task
    pub fn can_handle_task(&self, task: &crate::hardware::pool::TaskRequirements) -> bool {
        if !self.supports_frequency(task.frequency_hz) {
            return false;
        }

        if !self.supports_sample_rate(task.required_sample_rate) {
            return false;
        }

        true
    }
}

/// Backend-agnostic device error
#[derive(Debug)]
pub struct DeviceError {
    /// Error kind
    pub kind: DeviceErrorKind,
    /// Backend that generated the error
    pub backend: String,
    /// Detailed error message
    pub details: String,
}

/// Device error kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceErrorKind {
    /// Device not found
    NotFound,
    /// Device not available (in use, disconnected, etc.)
    NotAvailable,
    /// Invalid parameter
    InvalidParameter,
    /// Hardware error
    HardwareError,
    /// Operation timed out
    Timeout,
    /// Unsupported operation
    Unsupported,
}

impl DeviceError {
    /// Create a new device error
    pub fn new(kind: DeviceErrorKind, backend: &str, details: impl Into<String>) -> Self {
        Self {
            kind,
            backend: backend.to_string(),
            details: details.into(),
        }
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} error from {}: {}",
            match self.kind {
                DeviceErrorKind::NotFound => "Device not found",
                DeviceErrorKind::NotAvailable => "Device not available",
                DeviceErrorKind::InvalidParameter => "Invalid parameter",
                DeviceErrorKind::HardwareError => "Hardware error",
                DeviceErrorKind::Timeout => "Timeout",
                DeviceErrorKind::Unsupported => "Unsupported operation",
            },
            self.backend,
            self.details
        )
    }
}

impl std::error::Error for DeviceError {}

impl From<soapysdr::Error> for DeviceError {
    fn from(e: soapysdr::Error) -> Self {
        // Map SoapySDR errors to generic kinds
        let kind = match e.to_string().as_str() {
            s if s.contains("not found") => DeviceErrorKind::NotFound,
            s if s.contains("timeout") => DeviceErrorKind::Timeout,
            s if s.contains("not supported") => DeviceErrorKind::Unsupported,
            _ => DeviceErrorKind::HardwareError,
        };

        Self {
            kind,
            backend: "SoapySDR".to_string(),
            details: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_creation() {
        let id1 = DeviceId::from_serial("soapy", "12345");
        let id2 = DeviceId::from_serial("soapy", "12345");
        assert_eq!(id1, id2);

        match &id1 {
            DeviceId::Driver { driver, serial, .. } => {
                assert_eq!(driver, "soapy");
                assert_eq!(serial, "12345");
            }
            _ => panic!("Expected Driver variant"),
        }
    }

    #[test]
    fn test_device_id_backend() {
        let id = DeviceId::from_serial("sdrplay", "12345");
        assert_eq!(id.backend(), Backend::Soapy);

        let usb_id = DeviceId::Usb {
            vid: 0x0bda,
            pid: 0x2838,
            serial: "00000001".to_string(),
            bus_port: "1-2".to_string(),
        };
        assert_eq!(usb_id.backend(), Backend::Usb);
    }

    #[test]
    fn test_device_id_driver_normalization() {
        // Test that different capitalizations produce the same DeviceId
        let id_lowercase = DeviceId::from_serial("sdrplay", "12345");
        let id_mixedcase = DeviceId::from_serial("SDRplay", "12345");
        let id_uppercase = DeviceId::from_serial("SDRPLAY", "12345");

        assert_eq!(
            id_lowercase, id_mixedcase,
            "Lowercase and mixed case should match"
        );
        assert_eq!(
            id_lowercase, id_uppercase,
            "Lowercase and uppercase should match"
        );

        // Verify all are normalized to lowercase
        match &id_mixedcase {
            DeviceId::Driver { driver, .. } => {
                assert_eq!(
                    driver, "sdrplay",
                    "Driver should be normalized to lowercase"
                );
            }
            _ => panic!("Expected Driver variant"),
        }
    }

    #[test]
    fn test_device_id_display() {
        let id = DeviceId::from_serial("test", "001");
        assert_eq!(format!("{}", id), "test:001");
    }

    #[test]
    fn test_device_id_different_serials() {
        let id1 = DeviceId::from_serial("soapy", "12345");
        let id2 = DeviceId::from_serial("soapy", "67890");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_capabilities_frequency_check() {
        let caps = Capabilities {
            device_id: DeviceId::from_serial("test", "001"),
            rx_frequency_ranges: vec![(24e6, 1766e6)],
            rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
            gain_range: (0.0, 48.0),
            has_agc: true,
            antenna_options: vec![],
            channels: 1,
            max_bandwidth: 2_400_000.0,
            typical_latency_us: 50,
        };

        assert!(caps.supports_frequency(88.9e6));
        assert!(!caps.supports_frequency(10e6)); // Below range
        assert!(!caps.supports_frequency(2000e6)); // Above range
    }

    #[test]
    fn test_capabilities_sample_rate_check() {
        let caps = Capabilities {
            device_id: DeviceId::from_serial("test", "001"),
            rx_frequency_ranges: vec![(24e6, 1766e6)],
            rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
            gain_range: (0.0, 48.0),
            has_agc: true,
            antenna_options: vec![],
            channels: 1,
            max_bandwidth: 2_400_000.0,
            typical_latency_us: 50,
        };

        assert!(caps.supports_sample_rate(1_000_000.0));
        assert!(!caps.supports_sample_rate(100_000.0)); // Below range
        assert!(!caps.supports_sample_rate(10_000_000.0)); // Above range
    }

    #[test]
    fn test_device_error_display() {
        let err = DeviceError::new(
            DeviceErrorKind::NotFound,
            "TestBackend",
            "test device not found",
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Device not found"));
        assert!(msg.contains("TestBackend"));
        assert!(msg.contains("test device not found"));
    }
}
