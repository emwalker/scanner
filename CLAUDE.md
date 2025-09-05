# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Building and Checking
- `cargo check` - Check for syntax errors and basic correctness
- `cargo build` - Build the project
- `cargo run -- --format json --exit-early --stations 88.9e6` - Run tuned to specific frequency (88.9 MHz)
- `make lint` - Format code and run clippy with fixes

# Committing to Git
- Let's use one-line commit messages
- Run `make lint` and fix all warnings. We should not create a commit with linter warnings.
- In commit messages, omit the "Generated with" line
- In commit messages, omit the "Co-Authored-By" line

### Testing the Application
- Test with known good signal: `cargo run -- --center-freq 88.9e6 --exit-early` (88.9 MHz has confirmed strong FM signal)
- The application will output DSP debug information showing signal processing stats

## Architecture Overview

This is a Software Defined Radio (SDR) scanner application written in Rust that performs real-time FM demodulation and audio playback. The architecture follows a multi-threaded pipeline design:

### Core Components

1. **SdrManager** - Handles SDR device configuration and stream management
   - Configures SDRPlay RSPduo device (default driver)
   - Sets sample rate (960 kHz), bandwidth, frequency, and gain parameters
   - Creates and manages RX streams for IQ sample data

2. **DspProcessor** - Implements professional-grade FM demodulation
   - Pre-demodulation IIR low-pass filtering for FM channel bandwidth (~100 kHz cutoff)
   - FM quadrature demodulation with phase unwrapping
   - Post-demodulation audio filtering (15 kHz bandwidth for FM broadcast)
   - Decimation from SDR sample rate to audio sample rate (960 kHz → 48 kHz)
   - Comprehensive debugging statistics (signal strength, phase, frequency deviation)

3. **AudioPlayer** - Real-time audio output using CPAL
   - Supports multiple sample formats (F32, I16, U16)
   - Configurable buffer sizes to minimize underruns
   - Non-blocking audio sample consumption with fallback to silence

4. **MainLoop** - Orchestrates the entire processing pipeline
   - Multi-threaded architecture: SDR I/O thread, DSP main thread, audio playback thread
   - Bounded channels for inter-thread communication with proper buffering
   - Graceful shutdown coordination between threads

### Threading Model
- **Reader Thread**: Dedicated SDR I/O, reads IQ samples into bounded channel
- **Main Thread**: DSP processing and FM demodulation
- **Audio Thread**: Real-time audio playback with CPAL

### Signal Processing Chain
1. SDR samples (Complex<f32> at 960 kHz) → IIR pre-filter → FM demodulation → IIR audio filter → decimation → audio output (48 kHz)

### Key Constants
- FM_MAX_DEVIATION_HZ: 75,000 (75 kHz max deviation for FM broadcast)
- FM_CHANNEL_BANDWIDTH_HZ: 200,000 (200 kHz FM channel bandwidth)
- AUDIO_BANDWIDTH_HZ: 15,000 (15 kHz audio bandwidth)

## Device and Signal Information

### SDR Hardware
- SDRPlay RSPduo is the target SDR device
- Default device args: "driver=sdrplay"
- Device is confirmed working

### Test Signal
- Strong FM signal available at 88.9 MHz for testing
- Signal confirmed to work with GNU Radio and produces clear audio
- Use `--tune-freq 88900000` for reliable testing

## Dependencies
- soapysdr: SDR device interface
- rustfft: FFT operations (upgraded from custom DFT)
- cpal: Cross-platform audio I/O
- clap: Command-line argument parsing
- thiserror: Error handling
- rustradio: SDR processing library with SIMD optimizations enabled

## Rust Toolchain
This project uses nightly Rust (via rust-toolchain.toml) to enable rustradio's SIMD feature for improved DSP performance.