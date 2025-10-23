// Example: Trait-based Demodulator Architecture
//
// This demonstrates an extensible architecture for supporting multiple
// demodulation modes (FM, AM, SSB, digital modes) through a common trait
// interface.
//
// Key patterns:
// - Demodulator trait for pluggable demodulation
// - Registry pattern for mode selection
// - Metrics reporting for signal quality
// - State reset for frequency changes

use num_complex::Complex;
use std::collections::HashMap;

// ============================================================================
// Core Demodulator Trait
// ============================================================================

#[derive(Debug, Clone)]
pub struct DemodMetrics {
    pub snr_db: f32,
    pub signal_strength: f32,
    pub frequency_offset: f32,
}

#[derive(Debug)]
pub enum DemodError {
    InvalidInput,
    BufferTooSmall,
    HardwareError(String),
}

pub trait Demodulator: Send + Sync {
    /// Human-readable name of the demodulation mode
    fn name(&self) -> &str;

    /// Expected bandwidth of this mode (Hz)
    fn bandwidth(&self) -> f32;

    /// Required sample rate for demodulator input (Hz)
    fn required_sample_rate(&self) -> f32;

    /// Demodulate complex samples to real audio
    ///
    /// Returns the number of audio samples produced
    fn demodulate(
        &mut self,
        input: &[Complex<f32>],
        output: &mut [f32],
    ) -> Result<usize, DemodError>;

    /// Reset internal state (e.g., when changing frequency)
    fn reset(&mut self);

    /// Get current demodulator metrics
    fn metrics(&self) -> DemodMetrics;
}

// ============================================================================
// Example Implementation: FM Demodulator
// ============================================================================

pub struct FmDemodulator {
    prev_sample: Complex<f32>,
    sample_rate: f32,
    signal_strength: f32,
}

impl FmDemodulator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            prev_sample: Complex::new(0.0, 0.0),
            sample_rate,
            signal_strength: 0.0,
        }
    }
}

impl Demodulator for FmDemodulator {
    fn name(&self) -> &str {
        "FM"
    }

    fn bandwidth(&self) -> f32 {
        200_000.0 // 200 kHz for FM broadcast
    }

    fn required_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn demodulate(
        &mut self,
        input: &[Complex<f32>],
        output: &mut [f32],
    ) -> Result<usize, DemodError> {
        if output.len() < input.len() {
            return Err(DemodError::BufferTooSmall);
        }

        let mut total_power = 0.0;

        for (i, &sample) in input.iter().enumerate() {
            // Quadrature demodulation: arg(sample * conj(prev_sample))
            let product = sample * self.prev_sample.conj();
            self.prev_sample = sample;

            // atan2 gives phase angle, normalize to ±1.0
            output[i] = product.im.atan2(product.re) / std::f32::consts::PI;

            // Track signal strength
            total_power += sample.norm_sqr();
        }

        self.signal_strength = (total_power / input.len() as f32).sqrt();

        Ok(input.len())
    }

    fn reset(&mut self) {
        self.prev_sample = Complex::new(0.0, 0.0);
        self.signal_strength = 0.0;
    }

    fn metrics(&self) -> DemodMetrics {
        DemodMetrics {
            snr_db: 20.0 * self.signal_strength.log10(), // Rough estimate
            signal_strength: self.signal_strength,
            frequency_offset: 0.0,
        }
    }
}

// ============================================================================
// Example Implementation: AM Demodulator
// ============================================================================

pub struct AmDemodulator {
    sample_rate: f32,
    signal_strength: f32,
    dc_offset: f32,
}

impl AmDemodulator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            signal_strength: 0.0,
            dc_offset: 0.0,
        }
    }
}

impl Demodulator for AmDemodulator {
    fn name(&self) -> &str {
        "AM"
    }

    fn bandwidth(&self) -> f32 {
        10_000.0 // 10 kHz for AM broadcast
    }

    fn required_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn demodulate(
        &mut self,
        input: &[Complex<f32>],
        output: &mut [f32],
    ) -> Result<usize, DemodError> {
        if output.len() < input.len() {
            return Err(DemodError::BufferTooSmall);
        }

        const DC_ALPHA: f32 = 0.001;
        let mut total_power = 0.0;

        for (i, &sample) in input.iter().enumerate() {
            // Envelope detection: magnitude of complex sample
            let envelope = sample.norm();

            // Track and remove DC offset
            self.dc_offset = DC_ALPHA * envelope + (1.0 - DC_ALPHA) * self.dc_offset;
            output[i] = envelope - self.dc_offset;

            total_power += sample.norm_sqr();
        }

        self.signal_strength = (total_power / input.len() as f32).sqrt();

        Ok(input.len())
    }

    fn reset(&mut self) {
        self.signal_strength = 0.0;
        self.dc_offset = 0.0;
    }

    fn metrics(&self) -> DemodMetrics {
        DemodMetrics {
            snr_db: 20.0 * self.signal_strength.log10(),
            signal_strength: self.signal_strength,
            frequency_offset: 0.0,
        }
    }
}

// ============================================================================
// Example Implementation: SSB Demodulator
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsbMode {
    Upper, // USB
    Lower, // LSB
}

pub struct SsbDemodulator {
    mode: SsbMode,
    sample_rate: f32,
    signal_strength: f32,
}

impl SsbDemodulator {
    pub fn new(mode: SsbMode, sample_rate: f32) -> Self {
        Self {
            mode,
            sample_rate,
            signal_strength: 0.0,
        }
    }
}

impl Demodulator for SsbDemodulator {
    fn name(&self) -> &str {
        match self.mode {
            SsbMode::Upper => "USB",
            SsbMode::Lower => "LSB",
        }
    }

    fn bandwidth(&self) -> f32 {
        2700.0 // 2.7 kHz for voice SSB
    }

    fn required_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn demodulate(
        &mut self,
        input: &[Complex<f32>],
        output: &mut [f32],
    ) -> Result<usize, DemodError> {
        if output.len() < input.len() {
            return Err(DemodError::BufferTooSmall);
        }

        let mut total_power = 0.0;

        for (i, &sample) in input.iter().enumerate() {
            // Simplified SSB demod: I ± Q
            output[i] = match self.mode {
                SsbMode::Upper => sample.re + sample.im,
                SsbMode::Lower => sample.re - sample.im,
            };

            total_power += sample.norm_sqr();
        }

        self.signal_strength = (total_power / input.len() as f32).sqrt();

        Ok(input.len())
    }

    fn reset(&mut self) {
        self.signal_strength = 0.0;
    }

    fn metrics(&self) -> DemodMetrics {
        DemodMetrics {
            snr_db: 20.0 * self.signal_strength.log10(),
            signal_strength: self.signal_strength,
            frequency_offset: 0.0,
        }
    }
}

// ============================================================================
// Demodulator Registry
// ============================================================================

pub struct DemodulatorRegistry {
    factories: HashMap<String, Box<dyn Fn() -> Box<dyn Demodulator>>>,
}

impl DemodulatorRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            factories: HashMap::new(),
        };

        // Register built-in demodulators
        registry.register("FM", || Box::new(FmDemodulator::new(256_000.0)));
        registry.register("AM", || Box::new(AmDemodulator::new(48_000.0)));
        registry.register("USB", || Box::new(SsbDemodulator::new(SsbMode::Upper, 48_000.0)));
        registry.register("LSB", || Box::new(SsbDemodulator::new(SsbMode::Lower, 48_000.0)));

        registry
    }

    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn Demodulator> + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Demodulator>> {
        self.factories.get(name).map(|factory| factory())
    }

    pub fn available_modes(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }
}

// ============================================================================
// Mode Selection Helper
// ============================================================================

pub fn determine_mode_for_frequency(freq_hz: f64) -> &'static str {
    match freq_hz {
        f if (87.5e6..=108.0e6).contains(&f) => "FM",     // FM broadcast
        f if (530e3..=1700e3).contains(&f) => "AM",       // AM broadcast
        f if f >= 14.0e6 => "USB",                        // HF upper sideband
        f if f <= 10.0e6 => "LSB",                        // HF lower sideband
        _ => "FM", // Default
    }
}

// ============================================================================
// Example Usage
// ============================================================================

fn main() {
    let registry = DemodulatorRegistry::new();

    println!("Available modes: {:?}", registry.available_modes());

    // Create FM demodulator
    let mut fm_demod = registry.create("FM").expect("FM demodulator not found");

    // Generate test signal
    let samples: Vec<Complex<f32>> = (0..1024)
        .map(|i| {
            let phase = i as f32 * 0.1;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect();

    // Demodulate
    let mut audio = vec![0.0f32; samples.len()];
    match fm_demod.demodulate(&samples, &mut audio) {
        Ok(n) => {
            println!("{} demodulated {} samples", fm_demod.name(), n);
            let metrics = fm_demod.metrics();
            println!("Signal strength: {:.2}", metrics.signal_strength);
            println!("SNR: {:.2} dB", metrics.snr_db);
        }
        Err(e) => {
            eprintln!("Demodulation failed: {:?}", e);
        }
    }

    // Try different mode
    let mut usb_demod = registry.create("USB").expect("USB demodulator not found");
    usb_demod.demodulate(&samples, &mut audio).unwrap();
    println!("{} demodulated", usb_demod.name());

    // Auto-select mode based on frequency
    let freq = 88.9e6; // 88.9 MHz
    let mode = determine_mode_for_frequency(freq);
    println!("Frequency {:.1} MHz -> mode: {}", freq / 1e6, mode);

    let mut auto_demod = registry.create(mode).expect("Mode not found");
    println!("Using {} demodulator for {:.1} MHz", auto_demod.name(), freq / 1e6);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fm_demodulator() {
        let mut demod = FmDemodulator::new(256_000.0);

        let samples = vec![Complex::new(1.0, 0.0); 100];
        let mut audio = vec![0.0; 100];

        let result = demod.demodulate(&samples, &mut audio);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn test_registry() {
        let registry = DemodulatorRegistry::new();

        assert!(registry.create("FM").is_some());
        assert!(registry.create("AM").is_some());
        assert!(registry.create("USB").is_some());
        assert!(registry.create("LSB").is_some());
        assert!(registry.create("INVALID").is_none());
    }

    #[test]
    fn test_mode_selection() {
        assert_eq!(determine_mode_for_frequency(88.9e6), "FM");
        assert_eq!(determine_mode_for_frequency(1000e3), "AM");
        assert_eq!(determine_mode_for_frequency(14.2e6), "USB");
        assert_eq!(determine_mode_for_frequency(7.2e6), "LSB");
    }

    #[test]
    fn test_demod_reset() {
        let mut demod = FmDemodulator::new(256_000.0);

        // Process some samples
        let samples = vec![Complex::new(1.0, 1.0); 10];
        let mut audio = vec![0.0; 10];
        demod.demodulate(&samples, &mut audio).unwrap();

        // Check state is non-zero
        assert_ne!(demod.prev_sample, Complex::new(0.0, 0.0));

        // Reset
        demod.reset();

        // Check state is cleared
        assert_eq!(demod.prev_sample, Complex::new(0.0, 0.0));
    }
}
