# Software-Defined Radio Testing Framework

This document summarizes research on testing methodologies for Software-Defined Radio (SDR) applications, based on industry best practices from GNU Radio, MATLAB/Simulink, and other SDR development frameworks.

## Overview

Testing SDR applications presents unique challenges due to their real-time, signal processing nature. Unlike traditional software, SDR applications deal with continuous data streams, complex mathematical operations, and hardware dependencies that require specialized testing approaches.

## Testing Methodologies for SDR Applications

### 1. File-Based Testing with I/Q Recordings

**Concept**: Record real I/Q data to files for reproducible, deterministic testing.

**Benefits**:
- **Reproducibility**: Same input data every time
- **No hardware dependencies**: Tests can run without SDR hardware
- **Regression testing**: Detect changes in signal processing behavior
- **Sharing**: Test datasets can be shared across teams and tools

**Standards**: 
- **SigMF (Signal Metadata Format)**: Open standard for RF signal metadata
- **IQEngine**: Web-based platform for sharing and analyzing RF I/Q recordings
- **SDRplay/SDRangel**: Provide collections of sample I/Q files

**Implementation in Our Framework**:
```rust
// Load I/Q test fixture with metadata
let (sample_source, metadata) = load_iq_fixture("test_data/88_9_signal.iq")?;

// Test peak detection with known signal
let result = test_peak_detection_isolated(
    "test_data/fm_signal.iq", 
    &[88.9e6], // Expected peaks
    &config, 
    true // Debug mode
)?;
```

### 2. Synthetic Signal Generation

**Concept**: Generate known test signals programmatically for controlled testing.

**Applications**:
- **Tone testing**: Pure sinusoids at known frequencies
- **Noise testing**: White/pink noise for dynamic range testing  
- **Modulated signals**: Generate AM/FM/PSK signals with known parameters
- **Multi-signal scenarios**: Complex RF environments with multiple signals

**GNU Radio Approach**:
- **Vector Sources**: Provide predetermined sample sequences
- **Signal Sources**: Generate tones, chirps, and modulated signals
- **Noise Sources**: Add controlled noise for robustness testing

**Implementation Pattern**:
```rust
// Create synthetic FM signal at 100 kHz offset
let mut mock_source = MockSampleSource::new(
    1_000_000.0,  // Sample rate
    88.9e6,       // Center frequency
    100_000,      // Sample count
    100_000.0,    // Signal offset frequency
);
```

### 3. Pipeline Isolation Testing

**Concept**: Test individual blocks and stages of the signal processing pipeline separately.

**GNU Radio Best Practice**: "Write QA code first" for every block.

**Our Implementation**:
- **Peak Detection Testing**: Isolated from frequency translation
- **Frequency Translation Testing**: Isolated from peak detection  
- **End-to-End Pipeline Testing**: Complete signal flow validation

**Benefits**:
- **Faster debugging**: Identify which stage has issues
- **Unit-level validation**: Each component works correctly
- **Modular development**: Blocks can be developed independently

### 4. Stimulus-Response Testing

**Concept**: Input known signals and verify expected outputs.

**Methods**:
- **Broadband noise input**: Measure frequency response
- **Swept frequency testing**: Characterize filter responses
- **Dynamic range testing**: Test with various signal levels

**Applications in Our Framework**:
- **Filter verification**: FreqXlatingFir bandwidth and response
- **Demodulator testing**: FM demodulation accuracy
- **AGC testing**: Automatic gain control behavior

### 5. Regression Testing with Captured Logs

**Concept**: Capture and analyze detailed logs to detect behavioral changes.

**Our Implementation**:
```rust
// Capture logs during testing
let (result, logs) = with_captured_logs(true, Format::Json, || {
    test_complete_pipeline_debug(iq_file, station_freq, mode, &config)
})?;

// Analyze captured JSON logs
assert!(logs.contains("Peak detection test started"));
assert!(logs.contains("Candidate created"));
```

**Benefits**:
- **Behavioral tracking**: Detect subtle changes in processing
- **Performance monitoring**: Track processing times and resource usage
- **Debug information**: Detailed trace of signal processing steps

## Testing Categories for Our Scanner Application

### Unit Tests
- **Individual blocks**: FreqXlatingFir, peak detection, candidate creation
- **Mathematical operations**: Frequency calculations, offset validation
- **Data structures**: Peak, Candidate, configuration validation

### Integration Tests  
- **Pipeline segments**: Peak detection → candidate creation → frequency translation
- **Multi-block interactions**: How blocks work together
- **Configuration scenarios**: Different scan modes and parameters

### System Tests
- **End-to-end scenarios**: Complete scanning workflows
- **Hardware integration**: SDR device interaction (when available)
- **Real-world signals**: Performance with actual RF environment

### Performance Tests
- **Real-time constraints**: Can we process signals fast enough?
- **Memory usage**: Buffer sizes and memory allocation
- **CPU utilization**: DSP processing overhead

### Regression Tests
- **Behavioral consistency**: Same inputs produce same outputs
- **Performance regression**: Processing times don't degrade  
- **Feature preservation**: New changes don't break existing functionality

## Industry Best Practices

### GNU Radio Testing Approach
1. **"Write QA code first"** - Test-driven development
2. **Vector sources and sinks** - Controlled input/output testing
3. **Incremental complexity** - Start simple, add complexity gradually
4. **Multiple verification methods** - Visual inspection, file dumps, automated checks

### MATLAB/Simulink SDR Testing
1. **Model-in-the-loop testing** - Simulate before hardware
2. **Hardware-in-the-loop testing** - Real SDR devices with simulated RF
3. **Comprehensive unit tests** - Every function and block tested
4. **Code coverage analysis** - Ensure all paths are tested

### SDR Testing Challenges and Solutions

#### Challenge: Real-time Constraints
**Solution**: Use file-based testing for development, real-time testing for validation

#### Challenge: Hardware Dependencies  
**Solution**: Mock interfaces and synthetic signal sources

#### Challenge: Complex Signal Processing
**Solution**: Pipeline isolation and known-good reference implementations

#### Challenge: Non-deterministic Behavior
**Solution**: Controlled test environments and statistical validation

## Our Testing Framework Features

### Implemented
- ✅ **File-based I/Q testing** with metadata support
- ✅ **Pipeline isolation** for peak detection and frequency translation
- ✅ **Structured debug logging** with JSON capture
- ✅ **Synthetic test scenarios** for frequency translation validation
- ✅ **Integration test framework** for complete pipeline testing

### Ready to Implement
- 📋 **I/Q test fixture creation** from live signals
- 📋 **Performance benchmarking** with timing measurements  
- 📋 **Statistical validation** for non-deterministic components
- 📋 **Hardware mock interfaces** for CI/CD environments

## Usage Examples

### Debug Pipeline Issues
```bash
# Compare stations vs band scanning modes
cargo run -- --stations 88.9e6 --debug-pipeline --verbose --format json
cargo run -- --band fm --debug-pipeline --verbose --format json
```

### Run Automated Tests
```bash
# Unit and integration tests
cargo test --test integration_test

# With captured output
cargo test test_frequency_translation_scenarios -- --nocapture
```

### Create Test Fixtures
```bash
# Record I/Q data for testing (when implemented)
cargo run -- --stations 88.9e6 --capture-iq test_data/88_9_signal.iq --capture-duration 5.0
```

## Future Enhancements

### Automated Test Data Generation
- **Signal generators**: Create test I/Q files with known characteristics
- **Noise injection**: Add controlled interference for robustness testing
- **Multi-signal scenarios**: Complex RF environments for stress testing

### Continuous Integration Testing
- **Headless testing**: Run tests without SDR hardware
- **Performance tracking**: Monitor processing times across builds
- **Behavioral regression detection**: Automated comparison of processing results

### Advanced Analysis
- **Spectral analysis**: Verify frequency domain characteristics
- **Phase noise measurement**: Characterize oscillator stability
- **Distortion analysis**: THD and spurious signal detection

## References

Based on research from:
- **GNU Radio Wiki**: Testing and debugging tutorials
- **MATLAB SDR Documentation**: Model-based testing approaches
- **IQEngine Platform**: Standards for I/Q file sharing and analysis
- **SDRplay/SDRangel**: Sample I/Q file collections
- **SigMF Standard**: Signal metadata format specification

This framework provides a solid foundation for systematic testing of our SDR scanner application, enabling both development efficiency and high reliability in production.