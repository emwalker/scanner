# Multi-Mode Demodulation Architecture

This reference covers demodulation of signals beyond FM, including SSB (USB/LSB), AM, and digital modes, with focus on extensible architecture for adding new demodulation schemes.

## Trait-Based Demodulator Architecture

### Core Demodulator Trait

```rust
pub trait Demodulator: Send + Sync {
    /// Human-readable name of the demodulation mode
    fn name(&self) -> &str;

    /// Expected bandwidth of this mode (Hz)
    fn bandwidth(&self) -> f32;

    /// Required sample rate for demodulator input
    fn required_sample_rate(&self) -> f32;

    /// Demodulate complex samples to real audio
    fn demodulate(&mut self, input: &[Complex<f32>], output: &mut [f32]) -> Result<usize, DemodError>;

    /// Reset internal state (e.g., when changing frequency)
    fn reset(&mut self);

    /// Get current demodulator metrics (SNR, signal strength, etc.)
    fn metrics(&self) -> DemodMetrics;
}

#[derive(Debug, Clone)]
pub struct DemodMetrics {
    pub snr_db: f32,
    pub signal_strength: f32,
    pub frequency_offset: f32,  // Estimated offset from center
}
```

### Demodulator Registry

```rust
pub struct DemodulatorRegistry {
    demodulators: HashMap<String, Box<dyn Fn() -> Box<dyn Demodulator>>>,
}

impl DemodulatorRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            demodulators: HashMap::new(),
        };

        // Register built-in demodulators
        registry.register("FM", || Box::new(FmDemodulator::new()));
        registry.register("AM", || Box::new(AmDemodulator::new()));
        registry.register("USB", || Box::new(SsbDemodulator::new(SsbMode::Upper)));
        registry.register("LSB", || Box::new(SsbDemodulator::new(SsbMode::Lower)));

        registry
    }

    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn Demodulator> + 'static,
    {
        self.demodulators.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Demodulator>> {
        self.demodulators.get(name).map(|factory| factory())
    }
}
```

## SSB (Single Sideband) Demodulation

### SSB Signal Characteristics

**Upper Sideband (USB)**:
- Used above 10 MHz for amateur radio voice
- Frequency inversion: higher audio = higher RF
- Bandwidth: ~2.7 kHz (300 Hz - 3 kHz audio)

**Lower Sideband (LSB)**:
- Used below 10 MHz for amateur radio voice
- Normal frequency relationship
- Same bandwidth as USB

### Phasing Method (Hilbert Transform)

The phasing method uses the Hilbert transform to create analytic signal (complex representation with no negative frequencies), then shifts to baseband and selects desired sideband.

**Algorithm**:
1. Convert RF signal to complex baseband (I/Q)
2. Apply Hilbert transform to create analytic signal
3. Shift to audio frequencies
4. Select USB or LSB by adding/subtracting I and Q

**Implementation**:
```rust
pub struct SsbDemodulator {
    mode: SsbMode,
    hilbert_filter: HilbertTransform,
    agc: Agc,
}

pub enum SsbMode {
    Upper,  // USB
    Lower,  // LSB
}

pub struct HilbertTransform {
    taps: Vec<f32>,
    delay_line: Vec<Complex<f32>>,
}

impl HilbertTransform {
    pub fn new(num_taps: usize) -> Self {
        // Design Hilbert transformer using Parks-McClellan
        // Approximates 90-degree phase shift
        let taps = design_hilbert_filter(num_taps);

        Self {
            taps,
            delay_line: vec![Complex::zero(); num_taps],
        }
    }

    pub fn process(&mut self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        for (i, &sample) in input.iter().enumerate() {
            // Shift delay line
            self.delay_line.rotate_left(1);
            self.delay_line[self.delay_line.len() - 1] = sample;

            // Convolve with Hilbert filter taps
            let mut i_sum = 0.0;
            let mut q_sum = 0.0;

            for (tap, &delayed) in self.taps.iter().zip(self.delay_line.iter()) {
                i_sum += tap * delayed.re;
                q_sum += tap * delayed.im;
            }

            output[i] = Complex::new(i_sum, q_sum);
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
        2700.0  // 2.7 kHz
    }

    fn required_sample_rate(&self) -> f32 {
        48000.0  // Audio rate sufficient for SSB
    }

    fn demodulate(&mut self, input: &[Complex<f32>], output: &mut [f32]) -> Result<usize, DemodError> {
        let mut analytic = vec![Complex::zero(); input.len()];
        self.hilbert_filter.process(input, &mut analytic);

        for (i, &sample) in analytic.iter().enumerate() {
            // USB: I + Q, LSB: I - Q
            let audio = match self.mode {
                SsbMode::Upper => sample.re + sample.im,
                SsbMode::Lower => sample.re - sample.im,
            };

            // Apply AGC
            output[i] = self.agc.process(audio);
        }

        Ok(output.len())
    }

    fn reset(&mut self) {
        self.hilbert_filter.delay_line.fill(Complex::zero());
        self.agc.reset();
    }

    fn metrics(&self) -> DemodMetrics {
        DemodMetrics {
            snr_db: self.agc.estimate_snr(),
            signal_strength: self.agc.current_gain(),
            frequency_offset: 0.0,  // SSB doesn't track frequency
        }
    }
}
```

### Hilbert Filter Design

```python
from scipy import signal
import numpy as np

def design_hilbert_filter(num_taps=65):
    """
    Design FIR Hilbert transformer.

    Args:
        num_taps: Number of filter taps (must be odd)

    Returns:
        taps: FIR filter coefficients
    """
    # Parks-McClellan optimal FIR design
    # Passband: 0.1 to 0.9 of Nyquist (avoid DC and Nyquist freq)
    bands = [0.1, 0.9]  # Normalized frequency (0 to 1 = 0 to Nyquist)
    desired = [1, 1]     # Desired amplitude response
    taps = signal.remez(num_taps, bands, desired, type='hilbert')

    return taps
```

### AGC for SSB

SSB signals have varying amplitude, requiring automatic gain control:

```rust
pub struct Agc {
    target_level: f32,
    attack_rate: f32,
    decay_rate: f32,
    current_gain: f32,
    max_gain: f32,
}

impl Agc {
    pub fn new(target_level: f32, attack_time_ms: f32, decay_time_ms: f32, sample_rate: f32) -> Self {
        let attack_rate = 1.0 - (-1000.0 / (attack_time_ms * sample_rate)).exp();
        let decay_rate = 1.0 - (-1000.0 / (decay_time_ms * sample_rate)).exp();

        Self {
            target_level,
            attack_rate,
            decay_rate,
            current_gain: 1.0,
            max_gain: 100.0,  // 40 dB max gain
        }
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        let abs_sample = sample.abs();

        // Compute desired gain
        let desired_gain = if abs_sample > 1e-6 {
            self.target_level / abs_sample
        } else {
            self.max_gain
        }.min(self.max_gain);

        // Smooth gain changes (fast attack, slow decay)
        if desired_gain < self.current_gain {
            // Attack (gain decreasing = signal getting louder)
            self.current_gain += (desired_gain - self.current_gain) * self.attack_rate;
        } else {
            // Decay (gain increasing = signal getting quieter)
            self.current_gain += (desired_gain - self.current_gain) * self.decay_rate;
        }

        sample * self.current_gain
    }

    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}
```

## AM (Amplitude Modulation) Demodulation

### AM Signal Characteristics

- Bandwidth: Typically 10 kHz (AM broadcast), 6 kHz (amateur radio)
- Simple envelope detection
- Carrier present at center frequency

### Envelope Detection

```rust
pub struct AmDemodulator {
    agc: Agc,
    dc_blocker: DcBlocker,
}

impl Demodulator for AmDemodulator {
    fn name(&self) -> &str {
        "AM"
    }

    fn bandwidth(&self) -> f32 {
        10000.0  // 10 kHz for AM broadcast
    }

    fn required_sample_rate(&self) -> f32 {
        48000.0
    }

    fn demodulate(&mut self, input: &[Complex<f32>], output: &mut [f32]) -> Result<usize, DemodError> {
        for (i, &sample) in input.iter().enumerate() {
            // Envelope detection: magnitude of complex sample
            let envelope = sample.norm();

            // Remove DC component
            let ac_signal = self.dc_blocker.process(envelope);

            // Apply AGC
            output[i] = self.agc.process(ac_signal);
        }

        Ok(output.len())
    }

    fn reset(&mut self) {
        self.agc.reset();
        self.dc_blocker.reset();
    }

    fn metrics(&self) -> DemodMetrics {
        DemodMetrics {
            snr_db: self.agc.estimate_snr(),
            signal_strength: self.agc.current_gain(),
            frequency_offset: 0.0,
        }
    }
}

pub struct DcBlocker {
    prev_input: f32,
    prev_output: f32,
    alpha: f32,
}

impl DcBlocker {
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        Self {
            prev_input: 0.0,
            prev_output: 0.0,
            alpha,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        // High-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        let output = self.alpha * (self.prev_output + input - self.prev_input);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    pub fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}
```

## Digital Mode Demodulation

### PSK31 (Phase Shift Keying)

PSK31 is a popular amateur radio digital mode for keyboard-to-keyboard text communication.

**Characteristics**:
- Bandwidth: 31.25 Hz (extremely narrow)
- Modulation: BPSK (binary phase shift keying)
- Baud rate: 31.25 baud
- Requires very accurate frequency and phase tracking

**Basic Architecture**:
```rust
pub struct Psk31Demodulator {
    pll: Pll,               // Phase-locked loop for carrier tracking
    matched_filter: FirFilter,
    symbol_sampler: SymbolSampler,
    decoder: VaricodeDecoder,
}

pub struct Pll {
    phase: f32,
    frequency: f32,
    phase_error: f32,
    loop_bandwidth: f32,
}

impl Pll {
    pub fn track(&mut self, sample: Complex<f32>) -> Complex<f32> {
        // Generate local oscillator
        let lo = Complex::from_polar(1.0, self.phase);

        // Mix down to baseband
        let baseband = sample * lo.conj();

        // Phase detector: atan2 of I/Q
        let phase_error = baseband.im.atan2(baseband.re);

        // Loop filter (PI controller)
        self.frequency += self.loop_bandwidth * phase_error;
        self.phase += self.frequency + phase_error * self.loop_bandwidth;

        // Wrap phase to [-pi, pi]
        while self.phase > std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }
        while self.phase < -std::f32::consts::PI {
            self.phase += 2.0 * std::f32::consts::PI;
        }

        baseband
    }
}
```

### FT8 (Frequency Shift Keying)

FT8 is a weak-signal digital mode for making contacts with very low signal levels.

**Characteristics**:
- Bandwidth: 50 Hz
- 15-second transmission cycle
- Uses 8-FSK modulation
- Decoded via FFT and correlation

**Note**: FT8 decoding is complex, typically use existing library (e.g., `ft8_lib` in Rust or WSJT-X)

```rust
pub struct Ft8Demodulator {
    fft_engine: rustfft::FftPlanner<f32>,
    decoder: Ft8Decoder,  // From ft8_lib crate
    buffer: Vec<Complex<f32>>,
}

// FT8 operates on 15-second blocks
// Collect samples, decode once per cycle
```

## Mode Auto-Detection

### Heuristic-Based Detection

```rust
pub struct ModeDetector {
    fft: FftPlanner<f32>,
}

impl ModeDetector {
    pub fn detect_mode(&self, samples: &[Complex<f32>]) -> Option<&str> {
        // Compute power spectrum
        let spectrum = self.compute_spectrum(samples);

        // Check for FM: wide spectrum (>100 kHz)
        if self.estimate_bandwidth(&spectrum) > 100e3 {
            return Some("FM");
        }

        // Check for AM: strong carrier at center
        if self.has_strong_carrier(&spectrum) {
            return Some("AM");
        }

        // Check for SSB: no carrier, ~3 kHz bandwidth
        if !self.has_strong_carrier(&spectrum) && self.estimate_bandwidth(&spectrum) < 5e3 {
            // Determine USB vs LSB based on spectrum asymmetry
            if self.spectrum_centroid(&spectrum) > 0.0 {
                return Some("USB");
            } else {
                return Some("LSB");
            }
        }

        // Check for digital modes: very narrow bandwidth
        if self.estimate_bandwidth(&spectrum) < 100.0 {
            return Some("PSK31");  // Or other narrow digital mode
        }

        None
    }

    fn has_strong_carrier(&self, spectrum: &[f32]) -> bool {
        let center_idx = spectrum.len() / 2;
        let center_power = spectrum[center_idx];
        let avg_power = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        center_power > avg_power * 10.0  // Carrier 10 dB above average
    }

    fn estimate_bandwidth(&self, spectrum: &[f32]) -> f32 {
        // Find -6 dB points
        let peak_power = spectrum.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = peak_power - 6.0;  // -6 dB

        let mut lower = 0;
        let mut upper = spectrum.len() - 1;

        for i in 0..spectrum.len() / 2 {
            if spectrum[spectrum.len() / 2 - i] < threshold {
                lower = spectrum.len() / 2 - i;
                break;
            }
        }

        for i in 0..spectrum.len() / 2 {
            if spectrum[spectrum.len() / 2 + i] < threshold {
                upper = spectrum.len() / 2 + i;
                break;
            }
        }

        // Convert index difference to frequency bandwidth
        let sample_rate = 48000.0;  // Assumed
        let bin_width = sample_rate / spectrum.len() as f32;
        (upper - lower) as f32 * bin_width
    }
}
```

## Integration with Scanner ECS

### Mode Selection Component

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemodMode {
    FM,
    AM,
    USB,
    LSB,
    PSK31,
    FT8,
}

pub struct DemodulatorComponent {
    pub mode: DemodMode,
    pub demod: Box<dyn Demodulator>,
}
```

### ECS System: Create Demodulator

```rust
fn create_demodulator_system(world: &mut World) {
    let registry = world.get_resource::<DemodulatorRegistry>();

    for (entity, station, _) in query_stations_needing_demod(world) {
        let mode = determine_mode_for_frequency(station.frequency);
        let demod = registry.create(mode.as_str())
            .expect("Unknown demodulator mode");

        world.add_component(entity, DemodulatorComponent {
            mode,
            demod,
        });
    }
}

fn determine_mode_for_frequency(freq: f64) -> DemodMode {
    match freq {
        f if (87.5e6..=108.0e6).contains(&f) => DemodMode::FM,  // FM broadcast
        f if (530e3..=1700e3).contains(&f) => DemodMode::AM,     // AM broadcast
        f if f > 14.0e6 => DemodMode::USB,                       // HF USB above 10 MHz
        f if f < 10.0e6 => DemodMode::LSB,                       // HF LSB below 10 MHz
        _ => DemodMode::FM,  // Default
    }
}
```

### ECS System: Process Demodulation

```rust
fn demodulate_system(world: &mut World) {
    for (entity, demod_comp, audio_buffer) in query_active_demodulators(world) {
        // Get IQ samples from broadcast channel
        let iq_samples = receive_iq_samples_for_station(entity, world);

        // Demodulate
        let mut audio = vec![0.0f32; iq_samples.len()];
        match demod_comp.demod.demodulate(&iq_samples, &mut audio) {
            Ok(n) => {
                audio.truncate(n);
                audio_buffer.extend(audio);
            }
            Err(e) => {
                error!("Demodulation failed: {}", e);
            }
        }
    }
}
```

## Performance Considerations

### Computational Complexity

| Mode | Complexity | CPU Usage (relative to FM) |
|------|------------|----------------------------|
| FM   | Low (atan2) | 1.0x baseline |
| AM   | Very low (magnitude) | 0.5x |
| SSB  | Medium (Hilbert filter) | 2-3x |
| PSK31 | Medium (PLL + matched filter) | 2x |
| FT8  | High (FFT + correlation) | 5-10x |

### Optimization Strategies

- **SIMD**: Vectorize Hilbert filter convolution
- **Lookup tables**: Approximate atan2 for FM demod
- **Decimation**: Reduce sample rate before demodulation (SSB can work at 8-12 kHz audio rate)
- **Batch processing**: Process larger chunks to amortize overhead

## Reference Implementations

- **GNU Radio**: `gr-analog` and `gr-digital` modules
- **Liquid DSP**: C library with many demodulators
- **csdr**: Command-line SDR toolkit with efficient implementations
- **RustRadio**: Growing collection of Rust demodulators
