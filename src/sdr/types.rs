//! Common types for SDR backend abstraction

use crate::types::Result;
use std::fmt;

/// Tuner information returned by enumeration
#[derive(Clone, Debug)]
pub struct TunerInfo {
    /// Stable tuner identifier
    pub id: TunerId,
    /// Human-readable label
    pub label: String,
}

/// Stable tuner identifier
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum TunerId {
    /// Backend-based identification (SoapySDR, etc.)
    Backend { backend: String, serial: String },
    /// USB-based identification (VID/PID + serial + physical location)
    Usb {
        vid: u16,
        pid: u16,
        serial: String,
        bus_port: String,
    },
}

impl TunerId {
    /// Create a tuner ID from backend name and serial number
    pub fn from_serial(backend: &str, serial: &str) -> Self {
        Self::Backend {
            backend: backend.to_string(),
            serial: serial.to_string(),
        }
    }

    /// Get a string representation suitable for logging
    pub fn as_str(&self) -> String {
        match self {
            TunerId::Backend { backend, serial } => format!("{}:{}", backend, serial),
            TunerId::Usb {
                vid,
                pid,
                serial,
                bus_port,
            } => format!("usb:{:04x}:{:04x}:{}:{}", vid, pid, serial, bus_port),
        }
    }
}

impl fmt::Display for TunerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Comprehensive device capabilities
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Tuner identifier
    pub tuner_id: TunerId,

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
    /// Create capabilities from a SoapySDR device
    pub fn from_soapy_device(device: &soapysdr::Device) -> Result<Self> {
        let driver = device.driver_key()?;
        let hardware_info = device.hardware_info()?;
        let hardware_info_str: String = (&hardware_info).into();

        let rx_freq_ranges = device
            .frequency_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        let rx_sample_rate_ranges: Vec<(f64, f64)> = device
            .get_sample_rate_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        let gain_range = {
            let r = device.gain_range(soapysdr::Direction::Rx, 0)?;
            (r.minimum, r.maximum)
        };

        let max_bandwidth = rx_sample_rate_ranges.last().map(|r| r.1).unwrap_or(0.0);

        // Estimate latency based on driver type
        let typical_latency_us = match driver.as_str() {
            "rtlsdr" => 50,   // Fast USB2.0 device
            "sdrplay" => 100, // Moderate latency
            "hackrf" => 75,   // Fast USB2.0
            "lime" => 150,    // Higher latency
            _ => 100,         // Default estimate
        };

        Ok(Self {
            tuner_id: TunerId::from_serial(&driver, &hardware_info_str),
            rx_frequency_ranges: rx_freq_ranges,
            rx_sample_rate_ranges,
            gain_range,
            has_agc: device
                .has_gain_mode(soapysdr::Direction::Rx, 0)
                .unwrap_or(false),
            antenna_options: device
                .antennas(soapysdr::Direction::Rx, 0)
                .unwrap_or_default(),
            channels: device.num_channels(soapysdr::Direction::Rx).unwrap_or(1),
            max_bandwidth,
            typical_latency_us,
        })
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
    fn test_tuner_id_creation() {
        let id1 = TunerId::from_serial("soapy", "12345");
        let id2 = TunerId::from_serial("soapy", "12345");
        assert_eq!(id1, id2);

        match &id1 {
            TunerId::Backend { backend, serial } => {
                assert_eq!(backend, "soapy");
                assert_eq!(serial, "12345");
            }
            _ => panic!("Expected Backend variant"),
        }
    }

    #[test]
    fn test_tuner_id_display() {
        let id = TunerId::from_serial("test", "001");
        assert_eq!(format!("{}", id), "test:001");
    }

    #[test]
    fn test_tuner_id_different_serials() {
        let id1 = TunerId::from_serial("soapy", "12345");
        let id2 = TunerId::from_serial("soapy", "67890");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_capabilities_frequency_check() {
        let caps = Capabilities {
            tuner_id: TunerId::from_serial("test", "001"),
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
            tuner_id: TunerId::from_serial("test", "001"),
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
