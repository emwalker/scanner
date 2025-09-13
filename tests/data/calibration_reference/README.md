# FM Calibration Reference Dataset

This directory contains the comprehensive calibration results from testing 20+ FM frequencies to compare human perception vs analyzer classification.

## Calibration Session Details

- **Date**: 2025-09-14
- **Test Parameters**:
  - Duration: 5 seconds per frequency
  - Sample Rate: 2.0 MHz SDR, 48 kHz audio output
  - Gain: 24.0 dB
  - IF AGC: Enabled (0.1x gain reduction after frequency translation)
  - Squelch: Disabled for testing

## Summary of Findings

### Perfect Matches (12 frequencies)
Frequencies where Human perception matched Analyzer classification:
- **Static frequencies**: 88.1, 88.3, 88.5, 89.5, 89.9, 90.3, 90.5, 90.7, 91.3, 91.7 MHz
- **Good frequencies**: 88.9 MHz

### Mismatches (8 frequencies)
Critical frequencies where Human perception differed from Analyzer:

#### Signal Strength Threshold Issues (3 cases)
- **89.3 MHz**: Human=Poor, Analyzer=Static (signal_strength=0.161)
- **90.9 MHz**: Human=Poor, Analyzer=Static (signal_strength=0.130)
- **88.7 MHz**: Human=Poor, Analyzer=Static (signal_strength=0.834, but static_indicators=3)

#### Algorithm Conservatism Issues (5 cases)
- **89.1 MHz**: Human=Poor, Analyzer=Good (signal_strength=0.682, SNR=0.88dB)
- **89.7 MHz**: Human=Moderate, Analyzer=Good (signal_strength=0.406, SNR=3.6dB)
- **90.1 MHz**: Human=Moderate, Analyzer=Good (signal_strength=0.367, SNR=9.6dB)
- **91.1 MHz**: Human=Moderate, Analyzer=Good (signal_strength=0.282, SNR=5.8dB)
- **91.5 MHz**: Human=Moderate, Analyzer=Good (signal_strength=0.351, SNR=20.3dB) ⚠️

## Key Patterns Discovered

### 1. Signal Strength Threshold Too High
- Current threshold: 0.25
- Recommended: 0.15-0.18
- **Problem**: Audible poor-quality signals classified as static

### 2. Algorithm Too Optimistic
- **Problem**: Even excellent technical metrics (91.5MHz: SNR=20.3dB) don't guarantee human satisfaction
- **Root Cause**: Humans detect quality factors that spectral analysis misses
- **Solution**: More conservative "Good" classification, weight perceptual factors more heavily

### 3. SNR Correlation
- Low SNR (0.88-3.6 dB) often correlates with Human=Poor/Moderate but Analyzer=Good
- High SNR (9.6-20.3 dB) still doesn't guarantee human satisfaction

## Recommended Fixes

1. **Immediate**: Lower signal strength threshold from 0.25 to 0.17
2. **Short-term**: Implement more conservative Good rating (require higher standards)
3. **Long-term**: Research perceptual audio quality metrics (PESQ, STOI)

## Testing Protocol

All calibration tests used:
```bash
cargo run -- --stations <FREQ>e6 --duration 5 --disable-squelch --verbose --json
```

Where `<FREQ>` was tested across: 88.1, 88.3, 88.5, 88.7, 88.9, 89.1, 89.3, 89.5, 89.7, 89.9, 90.1, 90.3, 90.5, 90.7, 90.9, 91.1, 91.3, 91.5, 91.7 MHz.

This reference dataset should be preserved to ensure any algorithm changes maintain or improve these calibration results.