//! SoapySDR backend implementation

use super::{Backend, Capabilities, DeviceId, DeviceInfo, DeviceTrait};
use crate::types::Result;
use rustradio::Complex;
use rustradio::graph::GraphRunner;
use std::any::Any;

/// SoapySDR backend
///
/// Provides access to SDR devices through the SoapySDR abstraction layer.
/// Supports a wide variety of hardware: RTL-SDR, HackRF, SDRplay, LimeSDR, etc.
pub struct Soapy;

impl Backend for Soapy {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        let devices = soapysdr::enumerate("")?;

        Ok(devices
            .into_iter()
            .map(|d| {
                let serial = d.get("serial").unwrap_or("unknown").to_string();
                let model = d.get("label").unwrap_or("Unknown").to_string();
                let driver = d.get("driver").unwrap_or("soapy").to_string();

                DeviceInfo {
                    id: DeviceId::from_serial(&driver, &serial),
                    serial,
                    model,
                    backend: "SoapySDR".to_string(),
                }
            })
            .collect())
    }

    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn DeviceTrait>> {
        // Build device args from ID
        let backend = id.backend();
        let serial = id.serial();
        let args = format!("driver={},serial={}", backend, serial);

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
            assert!(!device.serial.is_empty());
            assert_eq!(device.backend, "SoapySDR");
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
}
