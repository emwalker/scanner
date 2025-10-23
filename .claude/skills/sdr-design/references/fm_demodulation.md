# FM Demodulation: DSP Chain Design and Optimization

This reference provides comprehensive coverage of FM broadcast demodulation, from SDR sample ingestion to audio output, with focus on filter design, decimation, and CPU optimization.

## FM Broadcast Signal Characteristics

### Frequency Allocation
- **Channel spacing**: 200 kHz (US, most of world)
- **Maximum deviation**: ±75 kHz
- **Audio bandwidth**: 15 kHz (L+R mono signal)
- **Pilot tone**: 19 kHz (stereo indicator)
- **Stereo subcarrier**: 38 kHz (L-R signal)
- **RDS/RBDS**: 57 kHz subcarrier

### Bandwidth Requirements
- **Carson's Rule**: BW = 2(Δf + fm) = 2(75 + 15) = 180 kHz
- **Practical**: 200 kHz channel includes guard bands
- **After demodulation**: 0-53 kHz (mono), 0-15 kHz (audio only)

## SDR Sample Rate Considerations

### Typical SDR Sample Rates
- **RTL-SDR**: 225 kHz - 3.2 MHz (often use 1.024 MHz or 2.048 MHz)
- **HackRF**: 2 MHz - 20 MHz (often use 2 MHz)
- **SDRPlay**: 2 MHz - 10 MHz (often use 2 MHz)
- **Trade-offs**: Higher rates → more CPU, better antialiasing; Lower rates → less CPU, must be careful with adjacent channels

### Recommended Starting Point
- **Sample rate**: 1.024 MHz or 2.048 MHz (power of 2 for efficient decimation)
- **Rationale**:
  - Easily decimated to audio rates
  - Wide enough for clean FM channel capture
  - Not excessively high CPU load
  - 1.024 MHz → 200 kHz bandwidth covers FM channel plus guards

## Decimation Strategy

### Multi-stage Decimation (Recommended)

Decimating in multiple stages reduces total computational cost compared to single-stage decimation.

**Example Chain (2.048 MHz → 48 kHz)**:
```
2.048 MHz
  → [decim by 4, LPF 256 kHz] → 512 kHz
  → [decim by 2, LPF 128 kHz] → 256 kHz
  → [FM demod] → 256 kHz (baseband)
  → [decim by 2, LPF 64 kHz] → 128 kHz
  → [decim by 2, LPF 32 kHz] → 64 kHz
  → [decim by 2, LPF 16 kHz] → 32 kHz
  → [resample 2/3] → 48 kHz (audio rate)
```

**Why multi-stage?**
- First filter operates at highest rate but needs fewer taps (wide transition band)
- Later filters operate at lower rates, can afford more taps for sharper cutoff
- Total multiply-accumulate operations much lower than single-stage

### Filter Design for Each Stage

#### Stage 1: 2.048 MHz → 512 kHz (decim 4)
```python
# Passband: 0 - 100 kHz (FM channel)
# Stopband: 256 kHz - Nyquist (prevent aliasing after decimation)
# Transition: 100 - 256 kHz = 156 kHz (wide, few taps needed)

sample_rate = 2048000
decimation = 4
passband_edge = 100000  # Hz
stopband_edge = sample_rate / decimation / 2  # Nyquist after decimation = 256 kHz
transition_width = stopband_edge - passband_edge  # 156 kHz
num_taps = estimate_taps(sample_rate, transition_width)  # ~20-40 taps
```

**Computational cost**: 20-40 multiplies per input sample at 2.048 MHz = 40-80 million multiplies/sec

#### Stage 2: 512 kHz → 256 kHz (decim 2)
```python
sample_rate = 512000
decimation = 2
passband_edge = 100000  # Hz, still preserving FM channel
stopband_edge = 128000  # Nyquist after decimation
transition_width = 28000  # Narrower than stage 1
num_taps = estimate_taps(sample_rate, transition_width)  # ~30-50 taps
```

**Computational cost**: 30-50 multiplies per input sample at 512 kHz = 15-25 million multiplies/sec

#### Post-Demodulation Stages
After FM demodulation, signal is real-valued baseband (not complex), so filter operations are simpler:

```python
# 256 kHz → 128 kHz → 64 kHz → 32 kHz
# Each stage progressively sharper, but lower sample rate
# Final stage: brick-wall at 15 kHz (audio bandwidth)
```

### Single-stage vs Multi-stage Example

**Single-stage 2.048 MHz → 48 kHz (decim 42.67)**:
- Transition band: 15 kHz - 24 kHz = 9 kHz
- Required taps: ~1000-2000 (extremely sharp filter at high rate)
- Cost: 1000-2000 multiplies × 2.048 MHz = **2-4 billion multiplies/sec**

**Multi-stage (6 stages)**:
- Total cost: ~100-200 million multiplies/sec
- **Speedup: 10-20x**

## FIR Filter Design Parameters

### Filter Specification Components

**Passband**:
- Frequency range where signal should pass unattenuated
- Ripple tolerance: typically ±0.01 dB (±0.1% amplitude variation)

**Stopband**:
- Frequency range where signal must be attenuated
- Attenuation requirement: typically -60 dB to -80 dB (1/1000 to 1/10000 amplitude)

**Transition Band**:
- Frequency range between passband and stopband
- Wider transition → fewer taps → lower CPU
- Narrower transition → more taps → higher CPU

**Cutoff Frequency**:
- Often defined as -6 dB point (half-power, 0.707 amplitude)
- In windowed sinc filters, cutoff is at center of transition band

### Filter Tap Estimation

**Kaiser Window Method** (common for FIR design):

```python
def estimate_filter_taps(sample_rate, transition_width, stopband_attenuation_db=60):
    """
    Estimate number of taps required for FIR filter.

    Args:
        sample_rate: Sample rate in Hz
        transition_width: Transition bandwidth in Hz
        stopband_attenuation_db: Required stopband attenuation in dB

    Returns:
        Number of filter taps (odd number)
    """
    # Normalized transition width
    delta_f = transition_width / sample_rate

    # Kaiser formula
    num_taps = int((stopband_attenuation_db - 8) / (2.285 * 2 * np.pi * delta_f)) + 1

    # Round up to next odd number
    if num_taps % 2 == 0:
        num_taps += 1

    return num_taps

# Example: 2.048 MHz sample rate, 50 kHz transition, 60 dB stopband
# taps = (60 - 8) / (2.285 * 2 * pi * 50000/2048000) + 1
#      = 52 / (2.285 * 2 * pi * 0.0244) + 1
#      = 52 / 0.350 + 1
#      ≈ 149 taps
```

### Practical Filter Design (scipy)

```python
from scipy import signal
import numpy as np

def design_decimation_filter(sample_rate, decimation, passband_edge, stopband_atten_db=60):
    """
    Design FIR lowpass filter for decimation.

    Args:
        sample_rate: Input sample rate (Hz)
        decimation: Decimation factor
        passband_edge: Passband edge frequency (Hz)
        stopband_atten_db: Stopband attenuation (dB)

    Returns:
        taps: FIR filter coefficients
    """
    nyquist = sample_rate / 2
    stopband_edge = sample_rate / decimation / 2  # Prevent aliasing
    transition_width = stopband_edge - passband_edge

    # Estimate required taps
    num_taps = estimate_filter_taps(sample_rate, transition_width, stopband_atten_db)

    # Design filter using Kaiser window
    taps = signal.firwin(
        num_taps,
        cutoff=passband_edge,
        window=('kaiser', signal.kaiser_beta(stopband_atten_db)),
        fs=sample_rate
    )

    return taps
```

## High-pass Filtering for DC Offset

### Why DC Offset Occurs

SDR hardware imperfections cause DC bias in I/Q samples:
- LO leakage in direct conversion receivers
- ADC offset errors
- Appears as large spike at center frequency in FFT

### DC Removal Approaches

**Simple Highpass Filter** (recommended):
```python
# Very gentle highpass: -3dB at 50 Hz, preserves audio down to ~20 Hz
# 1-pole IIR filter is computationally cheap

def design_dc_blocker(sample_rate, cutoff_hz=50):
    """Design simple DC blocking highpass filter."""
    from scipy import signal

    # 1st order Butterworth highpass
    sos = signal.butter(1, cutoff_hz, btype='highpass', fs=sample_rate, output='sos')
    return sos

# Apply in real-time
# y[n] = x[n] - x[n-1] + 0.99 * y[n-1]  # Simple difference equation
```

**DC Subtraction** (alternative):
```python
# Track running average, subtract it
dc_estimate = 0.0
alpha = 0.001  # Slow tracking

for sample in samples:
    dc_estimate = alpha * sample + (1 - alpha) * dc_estimate
    output = sample - dc_estimate
```

### Placement in Pipeline

Apply DC removal **before** FM demodulation, at highest sample rate stage:
- Removes DC spike from FFT displays
- Prevents DC from affecting quadrature demod math
- Minimal CPU impact (1-pole filter very cheap)

## FM Demodulation Algorithm

### Quadrature Demodulation (Recommended)

FM signal: `s(t) = A * cos(2πf_c*t + φ(t))` where `φ(t)` contains audio

**Instantaneous frequency**: `f_inst = f_c + (1/2π) * dφ/dt`

**Discrete implementation**:
```rust
pub struct QuadratureDemod {
    prev_sample: Complex<f32>,
}

impl QuadratureDemod {
    pub fn demod(&mut self, sample: Complex<f32>) -> f32 {
        // Compute arg(sample * conj(prev_sample))
        // This gives phase difference between consecutive samples
        let product = sample * self.prev_sample.conj();
        self.prev_sample = sample;

        // atan2 gives phase angle
        // Divide by pi to normalize to ±1.0 for ±75 kHz deviation
        product.im.atan2(product.re) / std::f32::consts::PI
    }
}
```

**Alternative (avoid atan2)**:
```rust
// Fast approximation: arctan(im/re) ≈ im/re for small angles
// Only valid if deviation is small relative to sample rate
pub fn fast_demod(&mut self, sample: Complex<f32>) -> f32 {
    let product = sample * self.prev_sample.conj();
    self.prev_sample = sample;

    // This is only accurate for small phase changes
    // At 2 MHz sample rate, 75 kHz deviation → ~0.038 radians per sample
    product.im / product.re.max(1e-6)  // Avoid division by zero
}
```

### De-emphasis Filtering

FM broadcast uses pre-emphasis (boost high frequencies) at transmitter, requiring de-emphasis at receiver.

**US Standard**: 75 μs time constant (Europe uses 50 μs)

**Transfer function**: `H(s) = 1 / (1 + s*τ)` where τ = 75 μs

**Digital implementation**:
```python
def design_deemphasis_filter(sample_rate, time_constant_us=75):
    """
    Design de-emphasis filter for FM broadcast.

    Args:
        sample_rate: Audio sample rate (Hz)
        time_constant_us: Time constant in microseconds (75 for US, 50 for EU)

    Returns:
        sos: Second-order sections for IIR filter
    """
    from scipy import signal

    tau = time_constant_us * 1e-6  # Convert to seconds
    cutoff_freq = 1.0 / (2 * np.pi * tau)  # ~2122 Hz for 75us

    # 1-pole lowpass filter
    sos = signal.butter(1, cutoff_freq, btype='low', fs=sample_rate, output='sos')
    return sos

# Cutoff frequency for 75 μs: fc = 1/(2π*75e-6) ≈ 2122 Hz
# This attenuates high frequencies to compensate for transmitter pre-emphasis
```

**Apply after FM demod, before final audio output**

## CPU Optimization Techniques

### Computational Profiling

Identify bottlenecks:
```bash
# Linux perf profiling
perf record -g ./scanner scan --stations 88.9e6 --duration 10
perf report

# Look for:
# - FIR filter convolution (usually largest cost)
# - FFT operations
# - Resampling
# - Trigonometric functions (atan2, sin, cos)
```

### Filter Optimization

**Use SIMD intrinsics**:
```rust
// Example: AVX2 vectorized FIR filter (8 f32s at once)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub unsafe fn fir_filter_avx2(input: &[f32], taps: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        let mut acc = _mm256_setzero_ps();

        for j in (0..taps.len()).step_by(8) {
            let inp = _mm256_loadu_ps(&input[i + j]);
            let tap = _mm256_loadu_ps(&taps[j]);
            acc = _mm256_fmadd_ps(inp, tap, acc);
        }

        // Horizontal sum of 8 lanes
        output[i] = horizontal_sum_avx2(acc);
    }
}
```

**Or use optimized libraries**:
- `rustfft` for FFT operations
- `rustradio` FIR filter implementations
- Consider frequency-domain filtering (overlap-add) for very long filters (>1000 taps)

### Decimation Optimization

Only compute output samples that will be kept:
```rust
// Bad: filter all samples, then decimate
let filtered: Vec<f32> = input.iter()
    .map(|&x| fir_filter(x, &taps))
    .collect();
let decimated: Vec<f32> = filtered.iter()
    .step_by(decimation)
    .copied()
    .collect();

// Good: polyphase decimation (only compute kept samples)
let decimated: Vec<f32> = input.chunks_exact(decimation)
    .map(|chunk| polyphase_filter(chunk, &taps, decimation))
    .collect();
```

Polyphase filtering reduces computation by factor of decimation rate.

### Avoiding Underruns and Artifacts

**Buffer Sizing**: See `buffer_sizing.md` for details

**Symptoms**:
- Clicks/pops in audio → buffer underrun
- Distortion → clipping, insufficient filter stopband attenuation
- Garbled audio → incorrect sample rate, missing decimation stage
- Hiss → quantization noise, insufficient bit depth

**Solutions**:
- Increase buffer sizes (trade latency for robustness)
- Reduce filter tap counts (trade quality for speed)
- Enable SIMD optimizations
- Profile and optimize hot loops
- Consider reducing SDR sample rate if not needed

## Reference Decimation Chains

### Conservative (High Quality)
```
2.048 MHz → [decim 2, 80 taps] → 1.024 MHz
1.024 MHz → [decim 2, 80 taps] → 512 kHz
512 kHz → [decim 2, 60 taps] → 256 kHz
[FM Demod]
256 kHz → [decim 4, 120 taps] → 64 kHz
64 kHz → [decim 4, 100 taps] → 16 kHz (for mono only)
or
64 kHz → [resample] → 48 kHz (for audio output)
```
**CPU usage**: Moderate-High, excellent audio quality

### Balanced (Recommended)
```
2.048 MHz → [decim 4, 40 taps] → 512 kHz
512 kHz → [decim 2, 40 taps] → 256 kHz
[FM Demod]
256 kHz → [decim 2, 60 taps] → 128 kHz
128 kHz → [decim 2, 60 taps] → 64 kHz
64 kHz → [resample 4/3] → 48 kHz
```
**CPU usage**: Low-Moderate, good audio quality

### Aggressive (Low CPU)
```
2.048 MHz → [decim 8, 32 taps] → 256 kHz
[FM Demod]
256 kHz → [decim 4, 40 taps] → 64 kHz
64 kHz → [decim 4, 40 taps] → 16 kHz
or
64 kHz → [resample] → 48 kHz
```
**CPU usage**: Very low, acceptable audio quality (may have some aliasing)

## Testing and Validation

### Frequency Response Analysis
```python
import matplotlib.pyplot as plt
from scipy import signal

# Design filter
taps = design_decimation_filter(2048000, 4, 100000)

# Compute frequency response
w, h = signal.freqz(taps, fs=2048000)

plt.plot(w, 20 * np.log10(abs(h)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Filter Frequency Response')
plt.grid()
plt.show()
```

### Audio Quality Metrics
- **SNR**: Should be >50 dB for good quality FM
- **THD**: Total harmonic distortion <1%
- **Frequency response**: Flat ±1 dB from 50 Hz to 15 kHz
- **Stereo separation**: >30 dB

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Distortion | Clipping, deviation too high | Reduce gain, check max deviation assumption |
| Hiss | Weak signal, quantization noise | Check signal strength, increase bit depth |
| Muffled audio | De-emphasis too strong | Check time constant (75μs vs 50μs) |
| Sharp audio | Missing de-emphasis | Add de-emphasis filter |
| Aliasing artifacts | Insufficient filtering before decim | More taps, narrower transition band |
| High CPU usage | Too many taps | Multi-stage decimation, reduce taps |
| Clicks/pops | Buffer underruns | Increase buffer sizes, reduce CPU load |

## Further Reading

- GNU Radio FM demodulation flowgraphs
- RustRadio FIR filter implementations
- scipy.signal filter design documentation
- "Digital Signal Processing" by Lyons (decimation chapter)
