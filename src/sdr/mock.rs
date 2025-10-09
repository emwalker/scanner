//! Mock backend for testing without hardware

use super::{
    Backend, Capabilities, DeviceError, DeviceErrorKind, DeviceId, DeviceInfo, DeviceTrait,
};
use crate::types::Result;
use rustradio::Complex;
use rustradio::graph::GraphRunner;
use std::any::Any;
use std::f32::consts::PI;

/// Mock backend for testing without hardware
///
/// Returns a fixed set of mock devices that can be used for testing
/// the device pool, task scheduler, and other components without
/// requiring actual SDR hardware.
pub struct Mock;

impl Backend for Mock {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![
            DeviceInfo {
                id: DeviceId::from_serial("mock", "001"),
                label: "Mock RTL-SDR (mock:001)".to_string(),
            },
            DeviceInfo {
                id: DeviceId::from_serial("mock", "002"),
                label: "Mock SDRplay (mock:002)".to_string(),
            },
        ])
    }

    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn DeviceTrait>> {
        Ok(Box::new(MockDevice::new(id.clone(), false)))
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

/// Mock device implementation
///
/// Generates realistic test signals (sine waves) and supports
/// failure injection for robustness testing.
pub struct MockDevice {
    device_id: DeviceId,
    capabilities: Capabilities,
    fail_on_tune: bool,
}

impl MockDevice {
    /// Create a new mock device
    ///
    /// # Arguments
    ///
    /// * `device_id` - Tuner identifier
    /// * `fail_on_tune` - If true, `tune()` will return an error (for testing error handling)
    pub fn new(device_id: DeviceId, fail_on_tune: bool) -> Self {
        let capabilities = Capabilities {
            device_id: device_id.clone(),
            rx_frequency_ranges: vec![(24e6, 1766e6)], // Typical RTL-SDR range
            rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
            gain_range: (0.0, 48.0),
            has_agc: true,
            antenna_options: vec!["RX".to_string()],
            channels: 1,
            max_bandwidth: 2_400_000.0,
            typical_latency_us: 50,
        };

        Self {
            device_id,
            capabilities,
            fail_on_tune,
        }
    }
}

impl DeviceTrait for MockDevice {
    fn id(&self) -> &DeviceId {
        &self.device_id
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        _freq: f64,
        samp_rate: f64,
        _gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        // Generate test signal: 100 kHz tone at center frequency
        let tone_freq = 100_000.0;
        let samples_per_period = (samp_rate / tone_freq) as usize;
        let total_samples = samples_per_period * 10; // 10 periods

        let samples: Vec<Complex> = (0..total_samples)
            .map(|i| {
                let phase = 2.0 * PI * (tone_freq as f32) * (i as f32) / (samp_rate as f32);
                Complex::new(phase.cos() * 0.5, phase.sin() * 0.5)
            })
            .collect();

        let (source, stream) = rustradio::blocks::VectorSource::new(samples);
        graph.add(Box::new(source));
        Ok(stream)
    }

    fn tune(&mut self, _freq: f64) -> Result<()> {
        if self.fail_on_tune {
            Err(DeviceError::new(
                DeviceErrorKind::HardwareError,
                "Mock",
                "simulated tuning failure",
            )
            .into())
        } else {
            Ok(())
        }
    }

    fn set_gain(&mut self, _gain: f64) -> Result<()> {
        Ok(())
    }

    fn into_inner(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustradio::graph::Graph;

    #[test]
    fn test_mock_backend_enumeration() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();

        assert_eq!(devices.len(), 2, "Mock backend should return 2 devices");
        assert_eq!(devices[0].id, DeviceId::from_serial("mock", "001"));
        assert_eq!(devices[1].id, DeviceId::from_serial("mock", "002"));
        assert!(devices[0].label.contains("Mock"));
    }

    #[test]
    fn test_mock_device_open() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let device = backend.open_device(&devices[0].id).unwrap();

        assert_eq!(device.id(), &DeviceId::from_serial("mock", "001"));
    }

    #[test]
    fn test_mock_device_capabilities() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let device = backend.open_device(&devices[0].id).unwrap();

        let caps = device.capabilities();
        assert!(caps.supports_frequency(88.9e6));
        assert!(caps.supports_sample_rate(1_000_000.0));
    }

    #[test]
    fn test_mock_device_graph_integration() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let device = backend.open_device(&devices[0].id).unwrap();

        let mut graph = Graph::new();
        let _stream = device
            .add_source_to_graph(&mut graph, 88.9e6, 2.4e6, 20.0)
            .unwrap();

        // Stream created successfully
    }

    #[test]
    fn test_mock_device_tune_success() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let mut device = backend.open_device(&devices[0].id).unwrap();

        // Should succeed
        device.tune(100e6).unwrap();
    }

    #[test]
    fn test_mock_device_tune_failure() {
        let device_id = DeviceId::from_serial("mock", "999");
        let mut device = MockDevice::new(device_id, true); // fail_on_tune = true

        // Should fail
        let result = device.tune(100e6);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_device_into_inner() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let device = backend.open_device(&devices[0].id).unwrap();

        let raw = device.into_inner();
        let device_id = raw.downcast::<DeviceId>().unwrap();
        assert_eq!(*device_id, DeviceId::from_serial("mock", "001"));
    }
}
