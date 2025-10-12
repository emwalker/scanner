//! Mock backend for testing without hardware

use super::{
    Backend, Capabilities, DeviceError, DeviceErrorKind, DeviceId, DeviceInfo, DeviceTrait,
};
use crate::core::types::Result;
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
        use super::types::TunerInfo;

        let device_001 = DeviceId::from_serial("mock", "001");
        let device_002 = DeviceId::from_serial("mock", "002");

        Ok(vec![
            DeviceInfo {
                id: device_001.clone(),
                label: "Mock RTL-SDR (mock:001)".to_string(),
                tuners: vec![TunerInfo {
                    id: crate::hardware::pool::TunerId::new(device_001, 0),
                    label: "Mock RTL-SDR (mock:001)".to_string(),
                    mode: String::new(),
                }],
            },
            DeviceInfo {
                id: device_002.clone(),
                label: "Mock SDRplay (mock:002)".to_string(),
                tuners: vec![TunerInfo {
                    id: crate::hardware::pool::TunerId::new(device_002, 0),
                    label: "Mock SDRplay (mock:002)".to_string(),
                    mode: String::new(),
                }],
            },
        ])
    }

    fn open_tuner(
        &self,
        tuner_id: &crate::hardware::pool::TunerId,
    ) -> Result<Box<dyn DeviceTrait>> {
        let (driver, serial) = match &tuner_id.device_id {
            DeviceId::Driver { driver, serial, .. } => (driver.as_str(), serial.as_str()),
            DeviceId::Usb { .. } => {
                return Err(
                    crate::core::types::ScannerError::UnsupportedDeviceIdFormat {
                        backend: "mock".to_string(),
                        device_format: "USB".to_string(),
                    },
                );
            }
        };

        Ok(Box::new(MockDevice::new(driver, serial, false)))
    }

    fn open_streaming_tuner(
        &self,
        tuner_id: &crate::hardware::pool::TunerId,
    ) -> Result<Box<dyn super::streaming::StreamingDevice>> {
        let (driver, serial) = match &tuner_id.device_id {
            DeviceId::Driver { driver, serial, .. } => (driver.as_str(), serial.as_str()),
            DeviceId::Usb { .. } => {
                return Err(
                    crate::core::types::ScannerError::UnsupportedDeviceIdFormat {
                        backend: "mock".to_string(),
                        device_format: "USB".to_string(),
                    },
                );
            }
        };

        Ok(Box::new(MockStreamingDevice {
            device_id: DeviceId::from_serial(driver, serial),
            sample_rate: 2_400_000.0,
            phase: 0.0,
        }))
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
    /// Uses the same DeviceId creation logic as SoapyDevice to ensure
    /// enumeration and opening produce matching DeviceIds.
    ///
    /// # Arguments
    ///
    /// * `driver` - Backend driver name (e.g., "mock", "sdrplay")
    /// * `serial` - Device serial number
    /// * `fail_on_tune` - If true, `tune()` will return an error (for testing error handling)
    pub fn new(driver: &str, serial: &str, fail_on_tune: bool) -> Self {
        let capabilities = Capabilities::for_mock(driver, serial);
        let device_id = capabilities.device_id.clone();

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
        let tuner_id = crate::hardware::pool::TunerId::new(devices[0].id.clone(), 0);
        let device = backend.open_tuner(&tuner_id).unwrap();

        assert_eq!(device.id(), &DeviceId::from_serial("mock", "001"));
    }

    #[test]
    fn test_mock_device_capabilities() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let tuner_id = crate::hardware::pool::TunerId::new(devices[0].id.clone(), 0);
        let device = backend.open_tuner(&tuner_id).unwrap();

        let caps = device.capabilities();
        assert!(caps.supports_frequency(88.9e6));
        assert!(caps.supports_sample_rate(1_000_000.0));
    }

    #[test]
    fn test_mock_device_graph_integration() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let tuner_id = crate::hardware::pool::TunerId::new(devices[0].id.clone(), 0);
        let device = backend.open_tuner(&tuner_id).unwrap();

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
        let tuner_id = crate::hardware::pool::TunerId::new(devices[0].id.clone(), 0);
        let mut device = backend.open_tuner(&tuner_id).unwrap();

        // Should succeed
        device.tune(100e6).unwrap();
    }

    #[test]
    fn test_mock_device_tune_failure() {
        let mut device = MockDevice::new("mock", "999", true); // fail_on_tune = true

        // Should fail
        let result = device.tune(100e6);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_device_into_inner() {
        let backend = Mock;
        let devices = backend.enumerate_devices().unwrap();
        let tuner_id = crate::hardware::pool::TunerId::new(devices[0].id.clone(), 0);
        let device = backend.open_tuner(&tuner_id).unwrap();

        let raw = device.into_inner();
        let device_id = raw.downcast::<DeviceId>().unwrap();
        assert_eq!(*device_id, DeviceId::from_serial("mock", "001"));
    }
}

pub struct MockStreamingDevice {
    device_id: DeviceId,
    sample_rate: f64,
    phase: f32,
}

impl super::streaming::StreamingDevice for MockStreamingDevice {
    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn channels(&self) -> usize {
        1
    }

    fn configure_rx(
        &mut self,
        _channel: usize,
        freq: f64,
        rate: f64,
        gain: f64,
    ) -> Result<super::streaming::ActualConfig> {
        self.sample_rate = rate;
        Ok(super::streaming::ActualConfig {
            freq_hz: freq,
            sample_rate: rate,
            gain_db: gain,
        })
    }

    fn start_stream(&mut self, _channel: usize) -> Result<()> {
        self.phase = 0.0;
        Ok(())
    }

    fn read_samples(
        &mut self,
        _channel: usize,
        buffer: &mut [Complex],
        _timeout_us: i64,
    ) -> Result<usize> {
        let freq = 1000.0;
        let phase_inc = 2.0 * PI * freq / self.sample_rate as f32;

        for sample in buffer.iter_mut() {
            *sample = Complex::new(self.phase.cos(), self.phase.sin());
            self.phase += phase_inc;
            if self.phase > 2.0 * PI {
                self.phase -= 2.0 * PI;
            }
        }

        let duration_ms = (buffer.len() as f64 / self.sample_rate) * 1000.0;
        std::thread::sleep(std::time::Duration::from_millis(duration_ms as u64));

        Ok(buffer.len())
    }

    fn stop_stream(&mut self, _channel: usize) -> Result<()> {
        Ok(())
    }
}
