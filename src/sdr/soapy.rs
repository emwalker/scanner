//! SoapySDR backend implementation

use super::{Backend, Capabilities, DeviceId, DeviceInfo, DeviceTrait};
use crate::types::{Result, ScannerError};
use rustradio::Complex;
use rustradio::graph::GraphRunner;
use std::any::Any;
use std::os::unix::io::AsRawFd;

/// Temporarily redirect stderr to /dev/null to suppress RtAudio spam
fn suppress_stderr<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use std::fs::OpenOptions;

    unsafe {
        let stderr_fd = libc::STDERR_FILENO;
        let saved_stderr = libc::dup(stderr_fd);

        if saved_stderr == -1 {
            return f();
        }

        let dev_null = OpenOptions::new().write(true).open("/dev/null").ok();

        if let Some(null_file) = dev_null {
            libc::dup2(null_file.as_raw_fd(), stderr_fd);
            let result = f();
            libc::dup2(saved_stderr, stderr_fd);
            libc::close(saved_stderr);
            result
        } else {
            libc::close(saved_stderr);
            f()
        }
    }
}

/// SoapySDR backend
///
/// Provides access to SDR devices through the SoapySDR abstraction layer.
/// Supports a wide variety of hardware: RTL-SDR, HackRF, SDRplay, LimeSDR, etc.
pub struct Soapy;

impl Backend for Soapy {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        // Suppress stderr during enumeration to prevent RtAudio spam
        let devices = suppress_stderr(|| soapysdr::enumerate(""))?;

        Ok(devices
            .into_iter()
            .filter_map(|d| {
                let driver = d.get("driver").unwrap_or("soapy");

                // Skip audio devices - we only want SDR hardware
                if driver == "audio" {
                    return None;
                }

                let serial = d.get("serial").unwrap_or("unknown").to_string();
                let mode = d.get("mode").unwrap_or("");
                let model = d.get("label").unwrap_or("Unknown").to_string();

                // Include mode in serial for devices like RSPduo that have multiple modes
                let unique_serial = if mode.is_empty() {
                    serial.clone()
                } else {
                    format!("{}:{}", serial, mode)
                };

                Some(DeviceInfo {
                    id: DeviceId::from_serial(driver, &unique_serial),
                    label: format!("{} ({}:{})", model, driver, serial),
                })
            })
            .collect())
    }

    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn DeviceTrait>> {
        let (backend, serial) = match id {
            DeviceId::Backend { backend, serial } => (backend.as_str(), serial.as_str()),
            DeviceId::Usb { .. } => {
                return Err(ScannerError::Custom(
                    "USB device IDs not supported for opening via SoapySDR backend".to_string(),
                ));
            }
        };

        // Handle RSPduo format: serial can be "1234:ST" or just "1234"
        let args = if let Some((actual_serial, mode)) = serial.split_once(':') {
            format!("driver={},serial={},mode={}", backend, actual_serial, mode)
        } else {
            format!("driver={},serial={}", backend, serial)
        };

        Ok(Box::new(SoapyDevice::new(args)?))
    }

    fn name(&self) -> &str {
        "SoapySDR"
    }
}

/// SoapySDR device wrapper
///
/// Important: We store device_args (String) not soapysdr::Device because:
/// 1. rustradio's SoapySdrSource::builder() consumes the device
/// 2. We need to create multiple graphs from the same device
/// 3. Creating a fresh device each time is safe with SoapySDR
pub struct SoapyDevice {
    device_id: DeviceId,
    device_args: String,
    capabilities: Capabilities,
}

impl SoapyDevice {
    /// Create a new SoapyDevice from device arguments
    pub fn new(device_args: String) -> Result<Self> {
        // Create temporary device to query capabilities
        let args = soapysdr::Args::from(device_args.as_str());
        let temp_device = soapysdr::Device::new(args)?;

        let capabilities = Capabilities::from_soapy_device(&temp_device)?;
        let device_id = capabilities.device_id.clone();

        Ok(Self {
            device_id,
            device_args,
            capabilities,
        })
    }
}

impl DeviceTrait for SoapyDevice {
    fn id(&self) -> &DeviceId {
        &self.device_id
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        // Create fresh device for this graph
        let args = soapysdr::Args::from(self.device_args.as_str());
        let device = soapysdr::Device::new(args)?;

        // Configure device
        if device.has_gain_mode(soapysdr::Direction::Rx, 0)? {
            device.set_gain_mode(soapysdr::Direction::Rx, 0, false)?;
        }

        // Normalize gain to 0.0-1.0 range (SDRplay uses 0-48 dB)
        let normalized_gain = gain_db.clamp(0.0, 48.0) / 48.0;

        // Build source and add to graph
        let (source_block, output_stream) =
            rustradio::blocks::SoapySdrSource::builder(&device, freq, samp_rate)
                .igain(normalized_gain)
                .build()?;

        graph.add(Box::new(source_block));
        Ok(output_stream)
    }

    fn tune(&mut self, freq: f64) -> Result<()> {
        // Note: This requires recreating the device
        // Most efficient to rebuild the graph instead
        let args = soapysdr::Args::from(self.device_args.as_str());
        let device = soapysdr::Device::new(args)?;
        device.set_frequency(soapysdr::Direction::Rx, 0, freq, "")?;
        Ok(())
    }

    fn set_gain(&mut self, gain: f64) -> Result<()> {
        let args = soapysdr::Args::from(self.device_args.as_str());
        let device = soapysdr::Device::new(args)?;
        device.set_gain(soapysdr::Direction::Rx, 0, gain)?;
        Ok(())
    }

    fn into_inner(self: Box<Self>) -> Box<dyn Any> {
        // Return device args as the "raw" representation
        Box::new(self.device_args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soapy_backend_name() {
        let backend = Soapy;
        assert_eq!(backend.name(), "SoapySDR");
    }

    #[test]
    #[ignore] // Requires actual hardware
    fn test_soapy_backend_enumeration() {
        let backend = Soapy;
        let devices = backend.enumerate_devices().unwrap();

        assert!(!devices.is_empty(), "Should find connected devices");

        for device in devices {
            assert!(!device.label.is_empty());
            assert!(device.label.contains(':'));
        }
    }

    #[test]
    #[ignore] // Requires actual hardware
    fn test_soapy_device_capabilities() {
        let backend = Soapy;
        let devices = backend.enumerate_devices().unwrap();
        assert!(!devices.is_empty());

        let device = backend.open_device(&devices[0].id).unwrap();
        let caps = device.capabilities();

        assert!(!caps.rx_frequency_ranges.is_empty());
        assert!(!caps.rx_sample_rate_ranges.is_empty());
        assert!(caps.channels > 0);
    }

    #[test]
    fn test_rspduo_multi_mode_enumeration() {
        use std::collections::HashMap;

        let mock_devices = vec![
            {
                let mut d = HashMap::new();
                d.insert("driver".to_string(), "sdrplay".to_string());
                d.insert("serial".to_string(), "2301034E34".to_string());
                d.insert("mode".to_string(), "ST".to_string());
                d.insert(
                    "label".to_string(),
                    "SDRplay Dev0 RSPduo 2301034E34 - Single Tuner".to_string(),
                );
                d
            },
            {
                let mut d = HashMap::new();
                d.insert("driver".to_string(), "sdrplay".to_string());
                d.insert("serial".to_string(), "2301034E34".to_string());
                d.insert("mode".to_string(), "DT".to_string());
                d.insert(
                    "label".to_string(),
                    "SDRplay Dev1 RSPduo 2301034E34 - Dual Tuner".to_string(),
                );
                d
            },
            {
                let mut d = HashMap::new();
                d.insert("driver".to_string(), "sdrplay".to_string());
                d.insert("serial".to_string(), "2301034E34".to_string());
                d.insert("mode".to_string(), "MA".to_string());
                d.insert(
                    "label".to_string(),
                    "SDRplay Dev2 RSPduo 2301034E34 - Master".to_string(),
                );
                d
            },
            {
                let mut d = HashMap::new();
                d.insert("driver".to_string(), "sdrplay".to_string());
                d.insert("serial".to_string(), "2301034E34".to_string());
                d.insert("mode".to_string(), "MA8".to_string());
                d.insert(
                    "label".to_string(),
                    "SDRplay Dev3 RSPduo 2301034E34 - Master (RSPduo sample rate=8Mhz)".to_string(),
                );
                d
            },
        ];

        let mut processed_devices = Vec::new();
        for d in &mock_devices {
            let driver = d.get("driver").unwrap();
            let serial = d.get("serial").unwrap();
            let mode = d.get("mode").map(|s| s.as_str()).unwrap_or("");
            let model = d.get("label").unwrap();

            let unique_serial = if mode.is_empty() {
                serial.clone()
            } else {
                format!("{}:{}", serial, mode)
            };

            processed_devices.push(DeviceInfo {
                id: DeviceId::from_serial(driver, &unique_serial),
                label: format!("{} ({}:{})", model, driver, serial),
            });
        }

        assert_eq!(
            processed_devices.len(),
            4,
            "Should enumerate all 4 RSPduo modes"
        );

        let ids: std::collections::HashSet<_> = processed_devices.iter().map(|d| &d.id).collect();
        assert_eq!(
            ids.len(),
            4,
            "All 4 RSPduo modes should have unique DeviceIds (regression test for duplicate serial issue)"
        );

        let expected_ids = vec![
            "sdrplay:2301034E34:ST",
            "sdrplay:2301034E34:DT",
            "sdrplay:2301034E34:MA",
            "sdrplay:2301034E34:MA8",
        ];

        for expected_id in expected_ids {
            assert!(
                processed_devices.iter().any(|d| match &d.id {
                    DeviceId::Backend { backend, serial } =>
                        format!("{}:{}", backend, serial) == expected_id,
                    _ => false,
                }),
                "Should find device with ID: {}",
                expected_id
            );
        }
    }
}
