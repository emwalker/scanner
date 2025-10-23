use tokio::sync::broadcast::error::TryRecvError;

use super::trait_def::SampleSource;

/// Adapter to make SDR broadcast receiver compatible with SampleSource trait
/// This allows the unified peak detection code to work with both testing sources and real SDR
/// streams
pub struct SdrStreamSource {
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    sample_rate: f64,
    center_frequency: f64,
    peak_scan_duration: f64,
    timeout_us: u64,
    pause_signal: Option<crate::pause_signal::PauseSignal>,
}

impl SdrStreamSource {
    pub fn new(
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        sample_rate: f64,
        center_frequency: f64,
        peak_scan_duration: f64,
        pause_signal: Option<crate::pause_signal::PauseSignal>,
    ) -> Self {
        Self {
            sdr_rx,
            sample_rate,
            center_frequency,
            peak_scan_duration,
            timeout_us: 100, // 100μs timeout between read attempts
            pause_signal,
        }
    }
}

impl SampleSource for SdrStreamSource {
    fn read_samples(
        &mut self,
        buffer: &mut [rustradio::Complex],
    ) -> crate::core::types::Result<usize> {
        use std::{thread, time::Duration};

        let mut samples_read = 0;
        let mut empty_count = 0;
        const MAX_EMPTY_RETRIES: usize = 10000; // 1 second at 100μs per retry

        while samples_read < buffer.len() {
            match self.sdr_rx.try_recv() {
                Ok(packet) => {
                    let samples = packet.as_slice();
                    let to_copy = samples.len().min(buffer.len() - samples_read);
                    buffer[samples_read..samples_read + to_copy]
                        .copy_from_slice(&samples[..to_copy]);
                    samples_read += to_copy;
                    empty_count = 0;
                }
                Err(TryRecvError::Empty) => {
                    if samples_read > 0 {
                        break;
                    }

                    // Check pause state before blocking (ECS Resource pattern)
                    if let Some(ref pause_signal) = self.pause_signal
                        && pause_signal.is_paused()
                    {
                        return Err(crate::core::types::ScannerError::Custom(
                            "Operation cancelled due to pause".to_string(),
                        ));
                    }

                    empty_count += 1;
                    if empty_count >= MAX_EMPTY_RETRIES {
                        return Err(crate::core::types::ScannerError::Custom(
                            "Timeout waiting for samples (possible pause or shutdown)".to_string(),
                        ));
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
