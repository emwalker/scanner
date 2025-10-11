//! SoapySDR backend implementation

use super::{Backend, Capabilities, DeviceId, DeviceInfo, DeviceTrait};
use crate::core::types::{Result, ScannerError};
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
                return Err(ScannerError::UnsupportedDeviceIdFormat {
                    backend: "soapysdr".to_string(),
                    device_format: "USB".to_string(),
                });
            }
        };

        // Handle RSPduo format: serial can be "1234:ST" or just "1234"
        let args = if let Some((actual_serial, mode)) = serial.split_once(':') {
            format!("driver={},serial={},mode={}", backend, actual_serial, mode)
        } else {
            format!("driver={},serial={}", backend, serial)
        };

        Ok(Box::new(SoapyDevice::new(args, backend, serial)?))
    }

    fn open_streaming_device(
        &self,
        id: &DeviceId,
    ) -> Result<Box<dyn super::streaming::StreamingDevice>> {
        let (backend, serial) = match id {
            DeviceId::Backend { backend, serial } => (backend.as_str(), serial.as_str()),
            DeviceId::Usb { .. } => {
                return Err(ScannerError::UnsupportedDeviceIdFormat {
                    backend: "soapysdr".to_string(),
                    device_format: "USB".to_string(),
                });
            }
        };

        let args = if let Some((actual_serial, mode)) = serial.split_once(':') {
            format!("driver={},serial={},mode={}", backend, actual_serial, mode)
        } else {
            format!("driver={},serial={}", backend, serial)
        };

        let device = suppress_stderr(|| soapysdr::Device::new(args.as_str()))?;
        let channels = device.num_channels(soapysdr::Direction::Rx).unwrap_or(1);

        Ok(Box::new(SoapyStreamingDevice {
            device,
            device_id: id.clone(),
            channels,
            active_streams: std::collections::HashMap::new(),
        }))
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
    pub fn new(device_args: String, expected_driver: &str, expected_serial: &str) -> Result<Self> {
        // Create temporary device to query capabilities
        let args = soapysdr::Args::from(device_args.as_str());
        let temp_device = soapysdr::Device::new(args)?;

        // Use the expected driver/serial from enumeration for DeviceId creation
        // This ensures the DeviceId matches what discovery created
        let capabilities =
            Capabilities::from_soapy_device(&temp_device, expected_driver, expected_serial)?;
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

/// Reset SoapySDR module state by unloading and reloading all modules.
/// This clears any stale mutex locks from previous runs.
pub fn reset_soapysdr_state() {
    unsafe {
        soapysdr_sys::SoapySDR_unloadModules();
        soapysdr_sys::SoapySDR_loadModules();
    }
}

/// Cleanup SoapySDR state on shutdown by unloading all modules.
pub fn cleanup_soapysdr_state() {
    unsafe {
        soapysdr_sys::SoapySDR_unloadModules();
    }
}

pub struct SoapyStreamingDevice {
    device: soapysdr::Device,
    device_id: DeviceId,
    channels: usize,
    active_streams: std::collections::HashMap<usize, soapysdr::RxStream<Complex>>,
}

impl super::streaming::StreamingDevice for SoapyStreamingDevice {
    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn configure_rx(
        &mut self,
        channel: usize,
        freq: f64,
        rate: f64,
        gain: f64,
    ) -> Result<super::streaming::ActualConfig> {
        self.device
            .set_sample_rate(soapysdr::Direction::Rx, channel, rate)?;
        self.device
            .set_frequency(soapysdr::Direction::Rx, channel, freq, "")?;
        self.device
            .set_gain(soapysdr::Direction::Rx, channel, gain)?;

        let actual_rate = self.device.sample_rate(soapysdr::Direction::Rx, channel)?;
        let actual_freq = self.device.frequency(soapysdr::Direction::Rx, channel)?;
        let actual_gain = self.device.gain(soapysdr::Direction::Rx, channel)?;

        Ok(super::streaming::ActualConfig {
            freq_hz: actual_freq,
            sample_rate: actual_rate,
            gain_db: actual_gain,
        })
    }

    fn start_stream(&mut self, channel: usize) -> Result<()> {
        let mut stream = self.device.rx_stream::<Complex>(&[channel])?;
        stream.activate(None)?;
        self.active_streams.insert(channel, stream);
        Ok(())
    }

    fn read_samples(
        &mut self,
        channel: usize,
        buffer: &mut [Complex],
        timeout_us: i64,
    ) -> Result<usize> {
        let stream = self
            .active_streams
            .get_mut(&channel)
            .ok_or_else(|| ScannerError::Custom(format!("Channel {} not streaming", channel)))?;

        let n = stream.read(&mut [buffer], timeout_us)?;
        Ok(n)
    }

    fn stop_stream(&mut self, channel: usize) -> Result<()> {
        if let Some(mut stream) = self.active_streams.remove(&channel) {
            stream.deactivate(None)?;
        }
        Ok(())
    }
}
