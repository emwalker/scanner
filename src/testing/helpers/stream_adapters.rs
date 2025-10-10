use super::trait_def::SampleSource;
use tokio::sync::broadcast::error::TryRecvError;

/// Adapter to make SDR broadcast receiver compatible with SampleSource trait
/// This allows the unified peak detection code to work with both testing sources and real SDR streams
pub struct SdrStreamSource {
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    sample_rate: f64,
    center_frequency: f64,
    peak_scan_duration: f64,
    timeout_us: u64,
}

impl SdrStreamSource {
    pub fn new(
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        sample_rate: f64,
        center_frequency: f64,
        peak_scan_duration: f64,
    ) -> Self {
        Self {
            sdr_rx,
            sample_rate,
            center_frequency,
            peak_scan_duration,
            timeout_us: 100, // 100μs timeout between read attempts
        }
    }
}

impl SampleSource for SdrStreamSource {
    fn read_samples(
        &mut self,
        buffer: &mut [rustradio::Complex],
    ) -> crate::core::types::Result<usize> {
        use std::thread;
        use std::time::Duration;

        let mut samples_read = 0;
        while samples_read < buffer.len() {
            match self.sdr_rx.try_recv() {
                Ok(packet) => {
                    let samples = packet.as_slice();
                    let to_copy = samples.len().min(buffer.len() - samples_read);
                    buffer[samples_read..samples_read + to_copy]
                        .copy_from_slice(&samples[..to_copy]);
                    samples_read += to_copy;
                }
                Err(TryRecvError::Empty) => {
                    if samples_read > 0 {
                        break;
                    }
                    thread::sleep(Duration::from_micros(self.timeout_us));
                    continue;
                }
                Err(TryRecvError::Lagged(_)) => {
                    continue;
                }
                Err(TryRecvError::Closed) => {
                    break;
                }
            }
        }
        Ok(samples_read)
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn peak_scan_duration(&self) -> f64 {
        self.peak_scan_duration
    }

    fn deactivate(&mut self) -> crate::core::types::Result<()> {
        // Nothing to deactivate for SDR stream source
        Ok(())
    }

    fn device_args(&self) -> &str {
        ""
    }
}
