use crate::types::Result;
use rustradio::Complex;
use rustradio::blocks::SoapySdrSource;

/// Our abstraction over SoapySDR device access
/// Hides device instantiation details and provides a single point for device readiness checks
pub struct SdrSource {
    device_args: String,
}

impl SdrSource {
    /// Create a new SdrSource when the device is ready
    /// Polls for device availability with retries and delays
    pub fn when_ready(device_args: String) -> Result<Self> {
        use std::thread;
        use std::time::Duration;

        const MAX_ATTEMPTS: u32 = 10;
        const RETRY_DELAY: Duration = Duration::from_millis(500);
        const TOTAL_TIMEOUT: Duration = Duration::from_secs(10);

        let start_time = std::time::Instant::now();

        for attempt in 1..=MAX_ATTEMPTS {
            // Check for timeout
            if start_time.elapsed() >= TOTAL_TIMEOUT {
                return Err(crate::types::ScannerError::Custom(format!(
                    "Timeout waiting for SDR device '{}' after {:?}",
                    device_args, TOTAL_TIMEOUT
                )));
            }

            // Check if any devices matching our args are available
            match soapysdr::enumerate(&*device_args) {
                Ok(matching_devices) => {
                    if !matching_devices.is_empty() {
                        println!("SDR device ready after {} attempt(s)", attempt);
                        return Ok(Self { device_args });
                    }
                }
                Err(e) => {
                    println!("Device enumeration error on attempt {}: {:?}", attempt, e);
                }
            }

            if attempt < MAX_ATTEMPTS {
                println!(
                    "SDR device '{}' not ready, waiting {:?} before retry {} of {}",
                    device_args, RETRY_DELAY, attempt, MAX_ATTEMPTS
                );
                thread::sleep(RETRY_DELAY);
            }
        }

        Err(crate::types::ScannerError::Custom(format!(
            "No SDR device available matching args: '{}' after {} attempts",
            device_args, MAX_ATTEMPTS
        )))
    }

    /// Create a source block for streaming IQ data
    /// This encapsulates the SoapySdrSource::builder pattern
    pub fn create_source_block(
        &self,
        frequency_hz: f64,
        sample_rate: f64,
    ) -> Result<(SoapySdrSource, rustradio::stream::ReadStream<Complex>)> {
        // TODO: Add device readiness check here in the future

        let dev = soapysdr::Device::new(self.device_args.as_str())?;
        let (sdr_source_block, sdr_output_stream) =
            SoapySdrSource::builder(&dev, frequency_hz, sample_rate)
                .igain(1 as _)
                .build()?;

        Ok((sdr_source_block, sdr_output_stream))
    }

    /// Create a raw SoapySDR device and stream for direct FFT analysis
    /// This encapsulates the device creation and stream setup
    pub fn create_raw_stream(&self, frequency_hz: f64, sample_rate: f64) -> Result<RawSdrStream> {
        // TODO: Add device readiness check here in the future

        let dev = soapysdr::Device::new(self.device_args.as_str())?;

        // Configure device frequency and sample rate
        dev.set_sample_rate(soapysdr::Direction::Rx, 0, sample_rate)?;
        dev.set_frequency(soapysdr::Direction::Rx, 0, frequency_hz, ())?;

        let mut rxstream = dev.rx_stream::<Complex>(&[0])?;
        rxstream.activate(None)?;

        Ok(RawSdrStream {
            _device: dev, // Keep device alive
            stream: rxstream,
        })
    }
}

/// Wrapper for raw SoapySDR stream access
pub struct RawSdrStream {
    _device: soapysdr::Device, // Keep device alive
    stream: soapysdr::RxStream<Complex>,
}

impl RawSdrStream {
    /// Read samples from the stream
    pub fn read_stream(
        &mut self,
        buffers: &mut [&mut [Complex]],
        timeout_us: i64,
    ) -> Result<usize> {
        let samples_read = self.stream.read(buffers, timeout_us)?;
        Ok(samples_read)
    }

    /// Deactivate the stream when done
    pub fn deactivate(mut self) -> Result<()> {
        self.stream.deactivate(None)?;
        Ok(())
    }
}
