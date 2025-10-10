use super::trait_def::SampleSource;
use crate::core::types::Result;
use rustradio::Complex;
use std::f32::consts::PI;
use tracing::debug;

/// Mock sample source for testing that generates a simple sine wave
pub struct MockSampleSource {
    sample_rate: f64,
    center_frequency: f64,
    samples_generated: usize,
    max_samples: usize,
    phase: f32,
    frequency_offset: f32, // Hz offset from center frequency
}

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
        }
    }
}

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
                self.phase.cos() * 0.5, // I component
                self.phase.sin() * 0.5, // Q component
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
