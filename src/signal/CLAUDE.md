# FM Signal Processing Research

This document provides technical analysis and findings about FM demodulation, band scanning, and filter design based on research and experimentation.

## FreqXlatingFir Filter Analysis

### Filter Bandwidth Requirements for Band Scanning

**Key Finding**: Band scanning requires wider filter bandwidths than initially expected due to frequency offset requirements.

**Research Results:**
- Standard FM filters (75 kHz cutoff) are too narrow for band scanning applications
- Band scanning with ±200 kHz frequency offsets requires filter support beyond ±60 kHz
- FreqXlatingFir has no gain compensation - signal retention depends entirely on filter frequency response

### Measured Filter Performance

#### Narrow Filter (75 kHz cutoff, 37.5 kHz transition):
```
Frequency Offset    Signal Retention    Quality
±0-50 kHz          >98%                EXCELLENT (>95%)
±60 kHz            83%                 GOOD (>80%)
±70-90 kHz         44% → 0.7%          POOR transition band
±100+ kHz          0%                  UNUSABLE (complete loss)
```

#### Wide Filter (300 kHz cutoff, 75 kHz transition):
```
Frequency Offset    Signal Retention    Quality
±0-200 kHz         100%+               EXCELLENT - perfect for band scanning
±300+ kHz          Variable            Beyond typical scanning requirements
```

### Usable Bandwidth Analysis

**Common Assumption**: 85% of sample rate is usable bandwidth
**Reality with Narrow Filters**: Only 10-12% (±50-60 kHz) is usable with standard FM filters
**Reality with Proper Filters**: Up to 90% usable bandwidth achievable with appropriate filter design

### Filter Design Trade-offs

**For band scanning applications requiring ±200 kHz coverage:**
- Narrow filter: 0% retention at ±200 kHz (complete failure)
- Wide filter: 100% retention at ±200 kHz (optimal performance)

**Technical Details:**
- FreqXlatingFir performs proper FIR filter frequency response (not square wave)
- Frequency translation works via complex exponential multiplication e^(-j2πft)
- Filter design uses standard windowing techniques (Hamming, etc.) with gradual rolloff

## CPU Load Analysis

### Counterintuitive Discovery: Wider Filters Reduce CPU Load

**Filter Design Trade-off**: For FIR filters, wider transition bands require fewer taps:
- **Narrow transition** (sharp cutoff): More taps needed for steep rolloff
- **Wide transition** (gradual cutoff): Fewer taps needed for gentle rolloff

**CPU Load Comparison**:
```
Filter Design                 Taps    MFLOPS    CPU vs Narrow    Memory
Narrow filter (75/37.5k)      65      65.0      Baseline         0.8 KB
Wide filter (400/100k)        25      25.0      0.4x (60% LESS)  0.3 KB
Very wide filter (480/120k)   21      21.0      0.3x (70% LESS)  0.2 KB
```

**Key Insight**: Narrow filters dominate processing pipeline with high FLOPs per sample, while wide filters require significantly fewer operations.

### Research Validation

Analysis confirmed by industry sources:
- *"The FIR filter has N coefficients and takes N x the sample rate operations per second"* (GNU Radio)
- *"Any time you drop the transition width by a factor of two, you can expect to need twice as many coefficients"* (DSP literature)
- *"An 800-tap filter at 2.4576 MHz results in nearly 8 billion floating-point operations per second"* (ZipCPU)

## SDR Bandwidth Scaling Analysis

### Maximum Usable Bandwidth Research

**Question**: How much of the available bandwidth is actually usable?
**Common assumption**: 85% usable
**Research findings**: Up to 90% usable with proper filter design

**SDR Capabilities by Bandwidth:**
- **1 MHz**: Standard implementation
- **2 MHz**: With automatic gain control (AGC)
- **10 MHz**: Without AGC (manual gain control required)

### Bandwidth Scaling Results

```
Configuration           CPU Load    Usable BW    Efficiency    AGC Support
1 MHz baseline          25 MFLOPS   0.8 MHz      Baseline      Yes
2 MHz + AGC            50 MFLOPS   1.6 MHz      1.00x         Yes
10 MHz no AGC          250 MFLOPS  8.0 MHz      1.00x         Manual
```

**Key Finding**: CPU scaling is perfectly linear - 2x bandwidth = 2x CPU, 10x bandwidth = 10x CPU.

### Strategic Recommendations

**Phase 1: Immediate (60% CPU Reduction)**
- Change filter from narrow to wide cutoff design
- **Result**: 60% less CPU usage, 90% usable bandwidth vs previous 12%

**Phase 2: 2 MHz + AGC Upgrade**
- **CPU cost**: 2x baseline (50 vs 25 MFLOPS)
- **Benefits**: 2x scanning range, retains AGC benefits
- **Web validation**: *"GPU acceleration advantageous for bandwidths ~2 MHz or higher"*

**Phase 3: 10 MHz Advanced (Optional)**
- **CPU cost**: 10x baseline (250 MFLOPS - still modest for modern CPUs)
- **Benefits**: 10x scanning range, entire band coverage capability
- **Trade-off**: Manual gain control required (no AGC)

### Industry Validation

Research confirms analysis aligns with established practices:
- *"At least third-generation Intel CPU required for 20MHz+ bandwidths"* (SDR industry)
- *"Processing gain occurs when noise outside the band is digitally removed"* (SDR literature)
- *"Narrower bandwidths provide better dynamic range through processing gain"* (AGC research)

### Alternative Optimization

For very high tap counts (>100), consider **FFT-based filtering**:
- *"FFT Filter implements decimating filter using fast convolution method via an FFT"* (GNU Radio)
- Useful when transition bandwidth requirements create excessive tap counts

## When Sharp Filter Cutoffs Justify CPU Cost

### Research Finding: Sharp Filters Rarely Worth It for FM Applications

**Test Results Summary**: Sharp filter transitions provide **identical performance** for most FM scenarios while consuming 5-10x more CPU.

### CPU vs Selectivity Trade-offs

```
Transition BW    Taps    CPU Cost    Stopband Rejection    FM Benefit
Wide (100k)      25      1.0x        20 dB                Sufficient for most use
Medium (50k)     49      2.0x        40 dB                Marginal improvement
Sharp (25k)      97      4.0x        60 dB                No additional benefit
Very Sharp (10k) 241     10.0x       80 dB                Wasted computation
```

**Key Finding**: For FM with 200 kHz channel spacing, all filter designs provide identical interference rejection because frequency separation already exceeds the transition band.

### Scenarios Where Sharp Filters ARE Worth the CPU Cost

#### 1. **Adjacent Channel Rejection (High-Density Areas)**
- **Use case**: Near broadcast transmitters with multiple strong FM stations
- **Benefit**: 40+ dB improvement rejecting signals exactly 200 kHz away
- **Worth it**: Professional monitoring in urban environments

#### 2. **Spurious Signal Suppression**
- **Use case**: Near industrial facilities, harmonics, intermodulation products
- **Benefit**: Clean stopband prevents false demodulation
- **Worth it**: Critical for reliable signal identification

#### 3. **Dynamic Range Preservation**
- **Use case**: Monitoring weak/distant stations in high-interference environment
- **Benefit**: Prevents out-of-band overload, maintains receiver sensitivity
- **Worth it**: Applications requiring maximum sensitivity

#### 4. **Multi-Channel Processing**
- **Use case**: Simultaneous processing of multiple channels
- **Benefit**: Reduces crosstalk between processed channels
- **Worth it**: Commercial monitoring systems, trunking applications

#### 5. **Regulatory Compliance**
- **Use case**: Professional/commercial SDR applications
- **Benefit**: Meeting spurious emission limits, spectral mask compliance
- **Worth it**: Legal requirement, no choice

### Strategic Filter Selection Guidelines

**For General FM Applications:**
- **Recommended**: Wide transition (100 kHz), 25 taps, 25 MFLOPS
- **Reasoning**: Sufficient performance, minimal CPU usage

**For Professional Monitoring:**
- **Consider**: Medium transition (50 kHz), 49 taps, 49 MFLOPS
- **Reasoning**: Better spurious rejection, still reasonable CPU cost

**For High-Interference Environments:**
- **May require**: Sharp transition (25 kHz), 97 taps, 97 MFLOPS
- **Reasoning**: Maximum adjacent channel rejection

### Modern CPU Reality Check

- **100 MFLOPS filter**: <10% of single CPU core (negligible impact)
- **500 MFLOPS filter**: ~50% of single CPU core (noticeable but acceptable)
- **1000+ MFLOPS filter**: May require dedicated core or GPU acceleration

## Multi-Channel Processing Analysis

### Single-Channel vs Multi-Channel Architecture

**Single-Channel Processing:**
```
SDR → FreqXlatingFir → Demodulator → Audio
Result: One signal at a time, sequential scanning
```

**Multi-Channel Processing:**
```
SDR → ┬─ FreqXlatingFir(-200k) → Demod → Channel 1
      ├─ FreqXlatingFir(-100k) → Demod → Channel 2
      ├─ FreqXlatingFir(+0k)   → Demod → Channel 3
      ├─ FreqXlatingFir(+100k) → Demod → Channel 4
      └─ FreqXlatingFir(+200k) → Demod → Channel 5

Result: 5 signals simultaneously, 5× scanning speed
```

### Filter Requirements Change Dramatically

**For multi-channel processing, filter requirements reverse:**
- **Wide filters**: Massive crosstalk between adjacent channels
- **Sharp filters**: Essential for clean channel separation

**This completely reverses the sharp filter recommendation** - multi-channel processing is exactly where expensive sharp filters become justified.

### Strategic Multi-Channel Recommendations

**For 1 MHz Bandwidth:**
- **Single-channel**: Use wide filters (25 taps, 25 MFLOPS)
- **Multi-channel**: Sharp filters become essential (5×97 = 485 taps, 485 MFLOPS)

**For 2 MHz Bandwidth:**
- **Recommended**: 5 channels at 400 kHz spacing with wide filters
- **CPU cost**: 125 taps total (5× single channel cost)
- **Performance**: 5× scanning speed, zero crosstalk

### Multi-Channel Applications

**Band Scanning:**
- Process multiple frequencies simultaneously
- Identify active signals across entire band
- 5-10× faster than sequential scanning

**Professional Monitoring:**
- Concurrent monitoring of multiple frequencies
- Monitor both control and voice channels
- Essential for trunking system analysis

## Window Overlap Optimization for Band Scanning

### Research-Based Findings on Optimal Overlap

Based on comprehensive research into SDR frequency scanning with overlapping windows:

#### **Optimal Overlap Percentages:**
- **75% overlap** is widely recommended as optimal for spectrum analysis applications
- **50% overlap** is the minimum for proper signal reconstruction with window functions
- **Higher overlap (75-97%)** provides better signal quality but increases computational load 4x vs 50%

#### **Why Window Overlap is Critical:**
- **Window function tapering**: Window functions (Hanning, Hamming, etc.) taper to zero at edges, attenuating signals
- **Signal loss prevention**: Without overlap, signals falling near window boundaries can be severely attenuated or missed
- **Artifact reduction**: Higher overlap reduces clicks, pops, and other processing artifacts
- **Transient capture**: Better detection of brief signals that might fall on window boundaries

#### **SDR-Specific Considerations:**
- **Filter rolloff effects**: Signals beyond certain offsets from center frequency start rolling off due to IF filter limitations
- **Usable bandwidth**: Research shows typically only 80% of total bandwidth is usable due to filter edge effects
- **DC spike avoidance**: Signals should be positioned away from center frequency to avoid DC offset issues

### **Computational Trade-offs**

**30% overlap approach:**
- Step size: 70% of window size
- Processing load: ~1.33x samples processed per frequency point

**75% overlap approach (optimal):**
- Step size: 25% of window size
- Processing load: ~4x samples processed per frequency point
- **Benefit**: Dramatically improved signal detection at frequency boundaries

### **AGC Settling Time Considerations**

Research reveals that **AGC settling time** is critical for overlapping window approaches:
- **Minimum settling time**: 1-3 seconds per frequency for proper AGC adaptation
- **Recommended**: 2-3 seconds per window for reliable signal strength measurements
- **Impact**: Proper AGC settling prevents false "weak signal" classifications

### **Industry Validation**

Research findings align with established SDR practices:
- *"75% overlap provides good approximation for capturing transient events"* (Tektronix FFT Analysis)
- *"High redundancy from overlap processing makes frequency domain processing more robust to noise"* (Signal Processing Research)
- *"For Von Hann or Hamming windows, 50% or 75% overlap is recommended"* (Spectrum Analysis Best Practices)

## FM Capture Effect and Filter Bandwidth Requirements

### Real-World RF Environment Analysis

**Research Finding**: Different frequencies can legitimately require different filter bandwidths based on local RF environment, not software bugs.

### FM Capture Effect Research

**FM Capture Effect Definition**: *"Only the stronger of two or more signals received within the bandwidth of the receiver passband will be demodulated"*

**Signal Strength Threshold**: *"The stronger signal needs to be only twice as powerful as the weaker signal"* for capture to occur

**Bandwidth Dependency**: Proper capture effect requires adequate receiver bandwidth to include the spectral interaction between competing signals

**Comparison with AM**: *"For FM, the stronger signal needs to be only twice as strong as the weaker one, while in case of AM, the stronger signal will have to be twenty times stronger to avoid objectionable interference"*

### Technical Implications

#### Filter Bandwidth Requirements Are Environment-Dependent:
- **Ideal conditions**: Standard Carson's Rule bandwidth (180-256 kHz) sufficient
- **Complex RF environment**: May require significantly wider bandwidth for proper FM demodulation when strong interfering signals are present
- **Local interference**: Adjacent strong signals require wider filters for proper capture effect

#### Frequency-Specific Requirements:
Different frequencies can legitimately require different filter bandwidths based on:
- **Local interference environment**: Adjacent strong signals
- **Propagation effects**: Multipath, fading, reflections
- **Transmitter characteristics**: Deviation, spurious emissions
- **Geographic factors**: Urban vs rural RF environment

### Recommendations

1. **For complex RF environments**: Use wide filter bandwidth (≥1600 kHz) or implement IF AGC
2. **For general scanning**: Consider adaptive filter bandwidth based on signal environment
3. **For interference analysis**: Wide filters reveal true RF environment rather than masking it
4. **For weak signal reception**: Combine IF AGC with appropriate filter bandwidth for target signal

This case study illustrates the importance of understanding real-world RF propagation effects when designing SDR signal processing systems.