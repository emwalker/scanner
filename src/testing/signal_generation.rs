//! Signal generation for testing peak detection algorithms

use crate::types::Result;
use rustradio::Complex;
use tracing::debug;

/// Enhanced test signal generator for peak detection validation
pub struct PeakTestSignalGenerator {
    sample_rate: f64,
    center_frequency: f64,
    samples_generated: usize,
    max_samples: usize,
    signals: Vec<TestSignal>,
    noise_level: f32,
    rng: rand::rngs::StdRng,
}

#[derive(Debug, Clone)]
pub struct TestSignal {
    pub frequency_hz: f64,
    pub amplitude: f32,
    pub phase_offset: f32,
    pub label: String,
}

impl TestSignal {
    pub fn new(frequency_hz: f64, amplitude: f32, label: &str) -> Self {
        Self {
            frequency_hz,
            amplitude,
            phase_offset: 0.0,
            label: label.to_string(),
        }
    }

    pub fn with_phase(mut self, phase_offset: f32) -> Self {
        self.phase_offset = phase_offset;
        self
    }
}

impl PeakTestSignalGenerator {
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        max_samples: usize,
        noise_level: f32,
    ) -> Self {
        use rand::SeedableRng;
        Self {
            sample_rate,
            center_frequency,
            samples_generated: 0,
            max_samples,
            signals: Vec::new(),
            noise_level,
            rng: rand::rngs::StdRng::seed_from_u64(42), // Deterministic seed for reproducibility
        }
    }

    pub fn add_signal(&mut self, signal: TestSignal) {
        debug!(
            "Adding test signal: {} at {:.3} MHz (offset: {:.1} kHz), amplitude: {:.3}",
            signal.label,
            signal.frequency_hz / 1e6,
            (signal.frequency_hz - self.center_frequency) / 1e3,
            signal.amplitude
        );
        self.signals.push(signal);
    }

    pub fn add_fm_stations(&mut self, frequencies: &[f64], amplitudes: &[f32]) {
        for (i, (&freq, &amp)) in frequencies.iter().zip(amplitudes.iter()).enumerate() {
            self.add_signal(TestSignal::new(freq, amp, &format!("FM_Station_{}", i + 1)));
        }
    }

    pub fn get_expected_peaks(&self) -> Vec<f64> {
        self.signals.iter().map(|s| s.frequency_hz).collect()
    }

    pub fn get_signal_labels(&self) -> Vec<String> {
        self.signals.iter().map(|s| s.label.clone()).collect()
    }
}

impl super::test_helpers::SampleSource for PeakTestSignalGenerator {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        use rand::Rng;

        let samples_to_generate = buffer.len().min(self.max_samples - self.samples_generated);
        if samples_to_generate == 0 {
            return Ok(0);
        }

        for sample in buffer.iter_mut().take(samples_to_generate) {
            let time = self.samples_generated as f32 / self.sample_rate as f32;
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            // Add all test signals
            for signal in &self.signals {
                let freq_offset = signal.frequency_hz - self.center_frequency;
                let angular_freq =
                    2.0 * std::f32::consts::PI * freq_offset as f32 / self.sample_rate as f32;
                let phase = angular_freq * time + signal.phase_offset;

                real += signal.amplitude * phase.cos();
                imag += signal.amplitude * phase.sin();
            }

            // Add white noise for realistic testing
            if self.noise_level > 0.0 {
                real += self.noise_level * self.rng.gen_range(-1.0..1.0);
                imag += self.noise_level * self.rng.gen_range(-1.0..1.0);
            }

            *sample = Complex::new(real, imag);
            self.samples_generated += 1;
        }

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
        "peak_test_generator"
    }

    fn peak_scan_duration(&self) -> f64 {
        1.5 // Use our optimized default
    }
}

/// Create a standard test scenario for peak detection validation
pub fn create_fm_band_test_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0, // 2 MHz sample rate
        90.0e6,      // 90.0 MHz center frequency
        3_000_000,   // 1.5 seconds at 2 MHz
        0.01,        // Low noise level
    );

    // Add realistic FM stations with varying strengths
    generator.add_signal(TestSignal::new(88.9e6, 0.8, "STRONG_Station_88.9"));
    generator.add_signal(TestSignal::new(90.1e6, 0.6, "MEDIUM_Station_90.1"));
    generator.add_signal(TestSignal::new(91.3e6, 0.3, "WEAK_Station_91.3"));
    generator.add_signal(TestSignal::new(89.5e6, 0.5, "MEDIUM_Station_89.5"));
    generator.add_signal(TestSignal::new(90.7e6, 0.4, "WEAK_Station_90.7"));

    generator
}

/// Create a test scenario with known weak signals for sensitivity testing
pub fn create_weak_signal_test_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0, // 2 MHz sample rate
        90.0e6,      // 90.0 MHz center frequency
        3_000_000,   // 1.5 seconds at 2 MHz
        0.05,        // Higher noise level
    );

    // Add very weak signals that might be missed without proper processing
    generator.add_signal(TestSignal::new(89.1e6, 0.15, "VERY_WEAK_Station_89.1"));
    generator.add_signal(TestSignal::new(90.3e6, 0.12, "VERY_WEAK_Station_90.3"));
    generator.add_signal(TestSignal::new(90.9e6, 0.18, "VERY_WEAK_Station_90.9"));

    generator
}

/// Create a test scenario with interference for robustness testing
pub fn create_interference_test_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0, // 2 MHz sample rate
        90.0e6,      // 90.0 MHz center frequency
        3_000_000,   // 1.5 seconds at 2 MHz
        0.08,        // Higher noise level
    );

    // Add legitimate signals
    generator.add_signal(TestSignal::new(89.5e6, 0.6, "GOOD_Station_89.5"));
    generator.add_signal(TestSignal::new(90.7e6, 0.5, "GOOD_Station_90.7"));

    // Add interference signals that might cause false positives
    generator.add_signal(TestSignal::new(89.75e6, 0.2, "INTERFERENCE_89.75"));
    generator.add_signal(TestSignal::new(90.45e6, 0.25, "INTERFERENCE_90.45"));

    generator
}
