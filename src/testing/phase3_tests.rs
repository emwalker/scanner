//! Phase 3 Tests: Spectral Preprocessing
//! Test-driven implementation for windowing, zero-padding, and overlap processing

use super::signal_generation::{PeakTestSignalGenerator, TestSignal};
use crate::types::ScanningConfig;

/// Test that windowing reduces spectral leakage by >10dB
#[test]
#[ignore] // TODO: Implement windowing functionality
fn test_windowing_reduces_spectral_leakage() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();

    // Test without windowing (baseline)
    config.enable_windowing = false;
    let mut no_window_generator = create_single_tone_scenario();
    let no_window_peaks = crate::fm::collect_peaks_from_source(&config, &mut no_window_generator)
        .expect("Failed to collect no-window peaks");

    // Test with Blackman-Harris windowing
    config.enable_windowing = true;
    config.window_type = crate::types::WindowType::BlackmanHarris;
    let mut windowed_generator = create_single_tone_scenario();
    let windowed_peaks = crate::fm::collect_peaks_from_source(&config, &mut windowed_generator)
        .expect("Failed to collect windowed peaks");

    // Measure spectral leakage around the main tone
    let main_frequency = 88_700_000.0;
    let leakage_reduction_db =
        measure_spectral_leakage_reduction(&no_window_peaks, &windowed_peaks, main_frequency);

    println!("Spectral leakage reduction: {:.1} dB", leakage_reduction_db);

    assert!(
        leakage_reduction_db > 10.0,
        "Windowing should reduce spectral leakage by >10dB, got {:.1}dB",
        leakage_reduction_db
    );
}

/// Test that zero-padding improves frequency resolution
#[test]
#[ignore] // TODO: Implement zero-padding functionality
fn test_zero_padding_improves_frequency_resolution() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();

    // Test without zero-padding (baseline)
    config.zero_padding_factor = 1; // No zero-padding
    let mut no_padding_generator = create_close_frequency_scenario();
    let no_padding_peaks = crate::fm::collect_peaks_from_source(&config, &mut no_padding_generator)
        .expect("Failed to collect no-padding peaks");

    // Test with 4x zero-padding
    config.zero_padding_factor = 4;
    let mut padded_generator = create_close_frequency_scenario();
    let padded_peaks = crate::fm::collect_peaks_from_source(&config, &mut padded_generator)
        .expect("Failed to collect zero-padded peaks");

    // Count how many of the close signals were resolved
    let target_frequencies = [88_700_000.0, 88_720_000.0]; // 20 kHz apart
    let tolerance = 5_000.0; // 5 kHz tolerance

    let no_padding_resolved =
        count_resolved_signals(&no_padding_peaks, &target_frequencies, tolerance);
    let padded_resolved = count_resolved_signals(&padded_peaks, &target_frequencies, tolerance);

    println!("No padding resolved: {}/2 signals", no_padding_resolved);
    println!("Zero-padded resolved: {}/2 signals", padded_resolved);

    assert!(
        padded_resolved > no_padding_resolved,
        "Zero-padding should improve frequency resolution and resolve more close signals"
    );
}

/// Test that 75% window overlap captures signals at bin edges
#[test]
#[ignore] // TODO: Implement window overlap processing
fn test_window_overlap_captures_edge_signals() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();

    // Test without overlap (baseline)
    config.window_overlap_percent = 0.0; // No overlap
    let mut no_overlap_generator = create_bin_edge_signal_scenario();
    let no_overlap_peaks = crate::fm::collect_peaks_from_source(&config, &mut no_overlap_generator)
        .expect("Failed to collect no-overlap peaks");

    // Test with 75% overlap
    config.window_overlap_percent = 75.0;
    let mut overlap_generator = create_bin_edge_signal_scenario();
    let overlap_peaks = crate::fm::collect_peaks_from_source(&config, &mut overlap_generator)
        .expect("Failed to collect overlap peaks");

    // Count signals detected (bin edge signals are harder to detect without overlap)
    let target_frequencies = [88_650_000.0, 88_750_000.0, 88_850_000.0]; // Signals at bin edges
    let tolerance = 10_000.0; // 10 kHz tolerance

    let no_overlap_detections =
        count_target_detections(&no_overlap_peaks, &target_frequencies, tolerance);
    let overlap_detections =
        count_target_detections(&overlap_peaks, &target_frequencies, tolerance);

    println!("No overlap detections: {}/3 signals", no_overlap_detections);
    println!("75% overlap detections: {}/3 signals", overlap_detections);

    assert!(
        overlap_detections > no_overlap_detections,
        "75% window overlap should capture more signals at bin edges"
    );
}

/// Test that FFT processing optimizations maintain speed
#[test]
fn test_fft_processing_maintains_speed() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();
    config.peak_scan_duration = 0.2; // Short for performance testing

    // Measure baseline performance (no spectral preprocessing)
    config.enable_windowing = false;
    config.zero_padding_factor = 1;
    config.window_overlap_percent = 0.0;

    let baseline_start = std::time::Instant::now();
    let mut baseline_generator = create_performance_test_scenario();
    let _baseline_peaks = crate::fm::collect_peaks_from_source(&config, &mut baseline_generator)
        .expect("Failed to collect baseline peaks");
    let baseline_duration = baseline_start.elapsed();

    // Measure performance with all spectral preprocessing enabled
    config.enable_windowing = true;
    config.window_type = crate::types::WindowType::BlackmanHarris;
    config.zero_padding_factor = 2; // Modest zero-padding for performance
    config.window_overlap_percent = 50.0; // Modest overlap for performance

    let preprocessed_start = std::time::Instant::now();
    let mut preprocessed_generator = create_performance_test_scenario();
    let _preprocessed_peaks =
        crate::fm::collect_peaks_from_source(&config, &mut preprocessed_generator)
            .expect("Failed to collect preprocessed peaks");
    let preprocessed_duration = preprocessed_start.elapsed();

    let slowdown_factor = preprocessed_duration.as_secs_f32() / baseline_duration.as_secs_f32();

    println!(
        "Baseline processing time: {:.3}s",
        baseline_duration.as_secs_f32()
    );
    println!(
        "Preprocessed processing time: {:.3}s",
        preprocessed_duration.as_secs_f32()
    );
    println!("Slowdown factor: {:.2}x", slowdown_factor);

    // Spectral preprocessing should not slow down processing by more than 3x
    assert!(
        slowdown_factor < 3.0,
        "Spectral preprocessing should not slow down processing by more than 3x, got {:.2}x",
        slowdown_factor
    );
}

// Helper functions for Phase 3 tests

fn create_test_config() -> ScanningConfig {
    ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(2),
        fft_size: 1024,
        peak_scan_duration: 0.3, // Fast for testing
        audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),
        // Spectral preprocessing defaults (will be overridden in tests)
        enable_windowing: false,
        window_type: crate::types::WindowType::Rectangular,
        zero_padding_factor: 1,
        window_overlap_percent: 0.0,
        ..Default::default()
    }
}

fn create_single_tone_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        600_000,      // max_samples (0.3 seconds)
        0.1,          // Low noise level
    );

    // Single strong tone for spectral leakage testing
    generator.add_signal(TestSignal::new(88_700_000.0, 0.5, "MainTone"));

    generator
}

fn create_close_frequency_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        600_000,      // max_samples (0.3 seconds)
        0.1,          // Low noise level
    );

    // Two signals very close in frequency (20 kHz apart)
    generator.add_signal(TestSignal::new(88_700_000.0, 0.3, "Signal1"));
    generator.add_signal(TestSignal::new(88_720_000.0, 0.3, "Signal2"));

    generator
}

fn create_bin_edge_signal_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        600_000,      // max_samples (0.3 seconds)
        0.2,          // Low noise level
    );

    // Signals positioned at FFT bin edges (harder to detect without overlap)
    generator.add_signal(TestSignal::new(88_650_000.0, 0.2, "BinEdge1"));
    generator.add_signal(TestSignal::new(88_750_000.0, 0.2, "BinEdge2"));
    generator.add_signal(TestSignal::new(88_850_000.0, 0.2, "BinEdge3"));

    generator
}

fn create_performance_test_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        400_000,      // max_samples (0.2 seconds)
        0.2,          // Moderate noise level
    );

    // Multiple signals for performance testing
    generator.add_signal(TestSignal::new(88_600_000.0, 0.3, "PerfSignal1"));
    generator.add_signal(TestSignal::new(88_750_000.0, 0.25, "PerfSignal2"));
    generator.add_signal(TestSignal::new(88_900_000.0, 0.2, "PerfSignal3"));

    generator
}

fn measure_spectral_leakage_reduction(
    no_window_peaks: &[crate::types::Peak],
    windowed_peaks: &[crate::types::Peak],
    main_frequency: f64,
) -> f32 {
    // Find the main peak magnitude in both cases
    let main_tolerance = 10_000.0; // 10 kHz tolerance for main peak

    let _no_window_main_mag = no_window_peaks
        .iter()
        .filter(|p| (p.frequency_hz - main_frequency).abs() < main_tolerance)
        .map(|p| p.magnitude)
        .fold(0.0f32, f32::max);

    let _windowed_main_mag = windowed_peaks
        .iter()
        .filter(|p| (p.frequency_hz - main_frequency).abs() < main_tolerance)
        .map(|p| p.magnitude)
        .fold(0.0f32, f32::max);

    // Find the peak leakage magnitude (excluding main peak area)
    let leakage_tolerance = 50_000.0; // Exclude 50 kHz around main peak

    let no_window_leakage_mag = no_window_peaks
        .iter()
        .filter(|p| (p.frequency_hz - main_frequency).abs() > leakage_tolerance)
        .map(|p| p.magnitude)
        .fold(0.0f32, f32::max);

    let windowed_leakage_mag = windowed_peaks
        .iter()
        .filter(|p| (p.frequency_hz - main_frequency).abs() > leakage_tolerance)
        .map(|p| p.magnitude)
        .fold(0.0f32, f32::max);

    // Calculate leakage reduction in dB
    if no_window_leakage_mag > 0.0 && windowed_leakage_mag > 0.0 {
        20.0 * (no_window_leakage_mag / windowed_leakage_mag).log10()
    } else {
        0.0 // No measurable leakage reduction
    }
}

fn count_resolved_signals(
    peaks: &[crate::types::Peak],
    target_frequencies: &[f64],
    tolerance: f64,
) -> usize {
    target_frequencies
        .iter()
        .filter(|&&freq| {
            peaks
                .iter()
                .any(|p| (p.frequency_hz - freq).abs() < tolerance)
        })
        .count()
}

fn count_target_detections(
    peaks: &[crate::types::Peak],
    target_frequencies: &[f64],
    tolerance: f64,
) -> usize {
    target_frequencies
        .iter()
        .filter(|&&freq| {
            peaks
                .iter()
                .any(|p| (p.frequency_hz - freq).abs() < tolerance)
        })
        .count()
}
