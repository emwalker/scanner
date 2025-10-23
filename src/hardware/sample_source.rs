//! Sample source abstraction for testable SDR operations
//!
//! This module provides the SampleSource trait that abstracts over different
//! sources of complex samples, enabling both real SDR hardware and mock data
//! for testing.

#[cfg(test)]
use std::f32::consts::PI;
use std::{
    fs::File,
    io::{BufReader, Read},
};

use rustradio::Complex;
use tokio::sync::broadcast::error::TryRecvError;
#[cfg(test)]
use tracing::debug;

use crate::core::types::Result;

/// Abstraction for sources of I/Q samples
pub trait SampleSource {
    /// Read samples into the provided buffer
    /// Returns the number of samples actually read
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize>;

    /// Get the configured sample rate
    fn sample_rate(&self) -> f64;

    /// Get the configured center frequency
    fn center_frequency(&self) -> f64;

    /// Clean up resources when done
    fn deactivate(&mut self) -> Result<()>;

    fn peak_scan_duration(&self) -> f64;

    fn device_args(&self) -> &str;
}

/// Mock sample source for testing that generates a simple sine wave
#[cfg(test)]
pub struct MockSampleSource {
    sample_rate: f64,
    center_frequency: f64,
    samples_generated: usize,
    max_samples: usize,
    phase: f32,
    frequency_offset: f32, // Hz offset from center frequency
    amplitude: f32,        // Signal amplitude
}

#[cfg(test)]
impl MockSampleSource {
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        max_samples: usize,
        signal_freq_offset: f32,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            samples_generated: 0,
            max_samples,
            phase: 0.0,
            frequency_offset: signal_freq_offset,
            amplitude: 0.5, // Default amplitude
        }
    }

    /// Create a MockSampleSource with custom amplitude
    pub fn with_amplitude(
        sample_rate: f64,
        center_frequency: f64,
        max_samples: usize,
        signal_freq_offset: f32,
        amplitude: f32,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            samples_generated: 0,
            max_samples,
            phase: 0.0,
            frequency_offset: signal_freq_offset,
            amplitude,
        }
    }
}

#[cfg(test)]
impl SampleSource for MockSampleSource {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        let samples_to_generate = buffer.len().min(self.max_samples - self.samples_generated);
        if samples_to_generate == 0 {
            return Ok(0);
        }

        let angular_freq = 2.0 * PI * self.frequency_offset / self.sample_rate as f32;
        debug!(
            "MockSampleSource: freq_offset={}, angular_freq={}",
            self.frequency_offset, angular_freq
        );

        for sample in buffer.iter_mut().take(samples_to_generate) {
            // Generate a single complex exponential at the specified frequency offset
            // This creates a pure tone at center_freq + frequency_offset
            // e^(j*phase) = cos(phase) + j*sin(phase)

            *sample = Complex::new(
                self.phase.cos() * self.amplitude, // I component
                self.phase.sin() * self.amplitude, // Q component
            );

            // Update phase
            self.phase += angular_freq;

            // Wrap phase to avoid accumulation errors
            if self.phase > 2.0 * PI {
                self.phase -= 2.0 * PI;
            }
        }

        self.samples_generated += samples_to_generate;
        Ok(samples_to_generate)
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn deactivate(&mut self) -> Result<()> {
        Ok(())
    }

    fn device_args(&self) -> &str {
        "test"
    }

    fn peak_scan_duration(&self) -> f64 {
        1.0
    }
}

/// File-based sample source for testing
pub struct FileSampleSource {
    reader: BufReader<File>,
    sample_rate: f64,
    center_frequency: f64,
    samples_remaining: usize,
}

impl FileSampleSource {
    pub fn new(file_path: &str, sample_rate: f64, center_frequency: f64) -> Result<Self> {
        let file = File::open(file_path)?;

        // Get file size to estimate number of samples (8 bytes per complex sample: f32 real + f32
        // imag)
        let file_size = file.metadata()?.len() as usize;
        let samples_remaining = file_size / 8; // 2 f32s per complex sample

        Ok(Self {
            reader: BufReader::new(file),
            sample_rate,
            center_frequency,
            samples_remaining,
        })
    }
}

impl SampleSource for FileSampleSource {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        let samples_to_read = buffer.len().min(self.samples_remaining);
        if samples_to_read == 0 {
            return Ok(0);
        }

        // Read raw bytes for f32 pairs
        let bytes_to_read = samples_to_read * 8; // 8 bytes per complex sample
        let mut byte_buffer = vec![0u8; bytes_to_read];

        match self.reader.read_exact(&mut byte_buffer) {
            Ok(_) => {
                // Convert bytes to Complex<f32> samples
                for (i, sample) in buffer.iter_mut().take(samples_to_read).enumerate() {
                    let real_bytes = &byte_buffer[i * 8..i * 8 + 4];
                    let imag_bytes = &byte_buffer[i * 8 + 4..i * 8 + 8];

                    let real = f32::from_le_bytes([
                        real_bytes[0],
                        real_bytes[1],
                        real_bytes[2],
                        real_bytes[3],
                    ]);
                    let imag = f32::from_le_bytes([
                        imag_bytes[0],
                        imag_bytes[1],
                        imag_bytes[2],
                        imag_bytes[3],
                    ]);

                    *sample = Complex::new(real, imag);
                }

                self.samples_remaining -= samples_to_read;
                Ok(samples_to_read)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn deactivate(&mut self) -> Result<()> {
        // Nothing to deactivate for file source
        Ok(())
    }

    fn device_args(&self) -> &str {
        ""
    }

    fn peak_scan_duration(&self) -> f64 {
        1.0
    }
}

/// Adapter to make SDR broadcast receiver compatible with SampleSource trait
/// This allows the unified peak detection code to work with both testing sources and real SDR
/// streams
pub struct SdrStreamSource {
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    sample_rate: f64,
    center_frequency: f64,
    peak_scan_duration: f64,
    timeout_us: u64,
}

impl SdrStreamSource {
    pub fn new(
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
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
        use std::{thread, time::Duration};

        let mut samples_read = 0;
        for slot in buffer.iter_mut() {
            match self.sdr_rx.try_recv() {
                Ok(sample) => {
                    *slot = sample;
                    samples_read += 1;
                }
                Err(TryRecvError::Empty) => {
                    // If we've read some samples, return what we have
                    if samples_read > 0 {
                        break;
                    }
                    // Otherwise wait a bit and try again
                    thread::sleep(Duration::from_micros(self.timeout_us));
                    continue;
                }
                Err(TryRecvError::Lagged(_)) => {
                    // Continue trying - lagged messages are not fatal
                    continue;
                }
                Err(TryRecvError::Closed) => {
                    // Channel closed - return what we have
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_source_interface() {
        let mut mock_source = MockSampleSource::new(
            1_000_000.0,  // 1 MHz sample rate
            88_900_000.0, // 88.9 MHz center frequency
            1000,         // 1000 samples max
            100_000.0,    // 100 kHz signal offset
        );

        // Test basic interface methods
        assert_eq!(mock_source.sample_rate(), 1_000_000.0);
        assert_eq!(mock_source.center_frequency(), 88_900_000.0);
        assert_eq!(mock_source.peak_scan_duration(), 1.0);

        // Test sample reading
        let mut buffer = vec![Complex::new(0.0, 0.0); 100];
        let samples_read = mock_source.read_samples(&mut buffer).unwrap();
        assert_eq!(samples_read, 100);

        // Verify samples are not all zero (signal + noise should be present)
        let non_zero_samples = buffer.iter().filter(|s| s.norm() > 0.01).count();
        assert!(
            non_zero_samples > 50,
            "Should have significant signal content"
        );

        // Test that we can read more samples
        let samples_read = mock_source.read_samples(&mut buffer).unwrap();
        assert_eq!(samples_read, 100);
    }

    #[test]
    fn test_mock_sample_source_signal_generation() {
        let mut mock_source = MockSampleSource::new(
            1_000_000.0,
            88_900_000.0,
            1000,
            100_000.0, // 100 kHz offset
        );

        let mut buffer = vec![Complex::new(0.0, 0.0); 1000];
        let samples_read = mock_source.read_samples(&mut buffer).unwrap();
        assert_eq!(samples_read, 1000);

        // Calculate average signal strength
        let avg_magnitude: f32 = buffer.iter().map(|s| s.norm()).sum::<f32>() / buffer.len() as f32;

        // Should have moderate signal strength
        assert!(avg_magnitude > 0.1, "Signal should be detectable");
        assert!(
            avg_magnitude < 1.0,
            "Signal should not exceed maximum expected amplitude"
        );
    }

    #[test]
    fn test_mock_sample_source_exhaustion() {
        let mut mock_source = MockSampleSource::new(
            1_000_000.0,
            88_900_000.0,
            500, // Only 500 samples available
            0.0,
        );

        let mut buffer = vec![Complex::new(0.0, 0.0); 1000];

        // First read should get 500 samples
        let samples_read = mock_source.read_samples(&mut buffer).unwrap();
        assert_eq!(samples_read, 500);

        // Second read should get 0 samples (exhausted)
        let samples_read = mock_source.read_samples(&mut buffer).unwrap();
        assert_eq!(samples_read, 0);
    }
}
