//! Detection Regression Tests
//! Tests to prevent regressions in station detection capabilities

use crate::audio::quality::AudioAnalyzer;
use crate::core::types::ScanningConfig;
use crate::testing::signal_generation::{PeakTestSignalGenerator, TestSignal};

/// Regression test: Signal averaging should not reduce detection count
#[test]
fn test_signal_averaging_does_not_reduce_detection_count() {
    // Create scenario with multiple weak signals that should all be detectable
    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: AudioAnalyzer::mock(),

        // Baseline: All signal averaging and CFAR features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::signal::peaks::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with signal averaging enabled
    let mut averaging_generator = create_multi_signal_detection_scenario();
    let averaging_config = ScanningConfig {
        // Signal averaging: Enable features
        enable_exponential_smoothing: true,
        enable_multi_frame_averaging: true,
        enable_coherent_integration: true,
        enable_moving_average_filter: true,

        // CFAR: Keep disabled to isolate signal averaging impact
        enable_cfar_detection: false,

        ..baseline_config.clone()
    };

    let averaging_peaks = crate::signal::peaks::collect_peaks_from_source(
        &averaging_config,
        &mut averaging_generator,
    )
    .expect("Failed to collect signal averaging peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!("Signal averaging detections: {}", averaging_peaks.len());

    // Signal averaging should not significantly reduce detection count
    let detection_ratio = averaging_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!(
        "Detection ratio (Signal Averaging / Baseline): {:.2}",
        detection_ratio
    );

    assert!(
        detection_ratio >= 0.8, // Allow up to 20% reduction, but we expect improvement
        "Signal averaging should not reduce detection count by more than 20%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );

    // Ideally, signal averaging should improve detection
    if detection_ratio > 1.0 {
        println!(
            "✅ Signal averaging improved detection by {:.1}%",
            (detection_ratio - 1.0) * 100.0
        );
    } else {
        println!(
            "⚠️  Signal averaging reduced detection by {:.1}%",
            (1.0 - detection_ratio) * 100.0
        );
    }
}

/// Regression test: CFAR should not reduce detection count
#[test]
fn test_cfar_does_not_reduce_detection_count() {
    // Test CFAR impact without signal averaging interference
    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: AudioAnalyzer::mock(),

        // Baseline: All features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,
        enable_windowing: false,
        enable_multi_frame_integration: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::signal::peaks::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with CFAR enabled
    let mut cfar_generator = create_multi_signal_detection_scenario();
    let cfar_config = ScanningConfig {
        // Signal averaging: Keep disabled to isolate CFAR impact
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,

        // CFAR: Enable for testing
        enable_cfar_detection: true,
        cfar_threshold_factor: 3.0, // Lower threshold for better detection
        cfar_guard_cells: 5,
        cfar_reference_cells: 20,

        ..baseline_config.clone()
    };

    let cfar_peaks =
        crate::signal::peaks::collect_peaks_from_source(&cfar_config, &mut cfar_generator)
            .expect("Failed to collect CFAR peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!("CFAR detections: {}", cfar_peaks.len());

    let detection_ratio = cfar_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!("Detection ratio (CFAR / Baseline): {:.2}", detection_ratio);

    assert!(
        detection_ratio >= 0.8, // Allow up to 20% reduction
        "CFAR should not reduce detection count by more than 20%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );
}

/// Regression test: Combined signal averaging + CFAR should not drastically reduce detection
#[test]
fn test_combined_phases_do_not_drastically_reduce_detection() {
    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: AudioAnalyzer::mock(),

        // Baseline: All features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,
        enable_windowing: false,
        enable_multi_frame_integration: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::signal::peaks::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with both signal averaging and CFAR enabled (exclude newer features)
    let mut combined_generator = create_multi_signal_detection_scenario();
    let combined_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: AudioAnalyzer::mock(),

        // Test combination: Signal averaging + CFAR enabled, newer features disabled
        enable_exponential_smoothing: true,
        enable_multi_frame_averaging: true,
        enable_coherent_integration: true,
        enable_moving_average_filter: true,
        enable_cfar_detection: true,
        enable_windowing: false,
        enable_multi_frame_integration: false,

        ..Default::default()
    };

    let combined_peaks =
        crate::signal::peaks::collect_peaks_from_source(&combined_config, &mut combined_generator)
            .expect("Failed to collect combined peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!(
        "Combined (Signal Averaging + CFAR) detections: {}",
        combined_peaks.len()
    );

    let detection_ratio = combined_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!(
        "Detection ratio (Combined / Baseline): {:.2}",
        detection_ratio
    );

    // This is the critical regression test - combined phases should not cause massive detection loss
    assert!(
        detection_ratio >= 0.5, // Allow up to 50% reduction, but this indicates a serious problem
        "Combined signal averaging + CFAR should not reduce detection count by more than 50%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );

    // Warn if we see significant reduction
    if detection_ratio < 0.8 {
        println!(
            "🚨 WARNING: Combined features reduced detection by {:.1}% - investigate signal averaging/CFAR interaction",
            (1.0 - detection_ratio) * 100.0
        );
    }
}

fn create_multi_signal_detection_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        89_000_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.3,          // Moderate noise level
    );

    // Add multiple signals at different strengths to test detection sensitivity
    generator.add_signal(TestSignal::new(88_700_000.0, 0.25, "Signal1")); // Strong
    generator.add_signal(TestSignal::new(88_900_000.0, 0.15, "Signal2")); // Medium
    generator.add_signal(TestSignal::new(89_100_000.0, 0.10, "Signal3")); // Weak
    generator.add_signal(TestSignal::new(89_300_000.0, 0.08, "Signal4")); // Very weak
    generator.add_signal(TestSignal::new(89_500_000.0, 0.05, "Signal5")); // Marginal

    generator
}
