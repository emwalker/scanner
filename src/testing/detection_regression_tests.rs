//! Detection Regression Tests
//! Tests to prevent regressions in station detection capabilities

use crate::testing::signal_generation::{PeakTestSignalGenerator, TestSignal};
use crate::types::ScanningConfig;

/// Regression test: Phase 1 signal averaging should not reduce detection count
#[test]
fn test_phase1_does_not_reduce_detection_count() {
    let _ = tracing_subscriber::fmt::try_init();

    // Create scenario with multiple weak signals that should all be detectable
    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),

        // Baseline: All Phase 1 and Phase 2 features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::fm::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with Phase 1 enabled
    let mut phase1_generator = create_multi_signal_detection_scenario();
    let phase1_config = ScanningConfig {
        // Phase 1: Enable signal averaging features
        enable_exponential_smoothing: true,
        enable_multi_frame_averaging: true,
        enable_coherent_integration: true,
        enable_moving_average_filter: true,

        // Phase 2: Keep CFAR disabled to isolate Phase 1 impact
        enable_cfar_detection: false,

        ..baseline_config.clone()
    };

    let phase1_peaks = crate::fm::collect_peaks_from_source(&phase1_config, &mut phase1_generator)
        .expect("Failed to collect Phase 1 peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!("Phase 1 detections: {}", phase1_peaks.len());

    // Phase 1 should not significantly reduce detection count
    let detection_ratio = phase1_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!(
        "Detection ratio (Phase 1 / Baseline): {:.2}",
        detection_ratio
    );

    assert!(
        detection_ratio >= 0.8, // Allow up to 20% reduction, but we expect improvement
        "Phase 1 signal averaging should not reduce detection count by more than 20%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );

    // Ideally, Phase 1 should improve detection
    if detection_ratio > 1.0 {
        println!(
            "✅ Phase 1 improved detection by {:.1}%",
            (detection_ratio - 1.0) * 100.0
        );
    } else {
        println!(
            "⚠️  Phase 1 reduced detection by {:.1}%",
            (1.0 - detection_ratio) * 100.0
        );
    }
}

/// Regression test: Phase 2 CFAR should not reduce detection count
#[test]
fn test_phase2_does_not_reduce_detection_count() {
    let _ = tracing_subscriber::fmt::try_init();

    // Test CFAR impact without Phase 1 interference
    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),

        // Baseline: All features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::fm::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with Phase 2 enabled
    let mut phase2_generator = create_multi_signal_detection_scenario();
    let phase2_config = ScanningConfig {
        // Phase 1: Keep disabled to isolate Phase 2 impact
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,

        // Phase 2: Enable CFAR
        enable_cfar_detection: true,
        cfar_threshold_factor: 3.0, // Lower threshold for better detection
        cfar_guard_cells: 5,
        cfar_reference_cells: 20,

        ..baseline_config.clone()
    };

    let phase2_peaks = crate::fm::collect_peaks_from_source(&phase2_config, &mut phase2_generator)
        .expect("Failed to collect Phase 2 peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!("Phase 2 detections: {}", phase2_peaks.len());

    let detection_ratio = phase2_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!(
        "Detection ratio (Phase 2 / Baseline): {:.2}",
        detection_ratio
    );

    assert!(
        detection_ratio >= 0.8, // Allow up to 20% reduction
        "Phase 2 CFAR should not reduce detection count by more than 20%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );
}

/// Regression test: Combined Phase 1+2 should not drastically reduce detection
#[test]
fn test_combined_phases_do_not_drastically_reduce_detection() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut baseline_generator = create_multi_signal_detection_scenario();
    let baseline_config = ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(3),
        fft_size: 1024,
        peak_scan_duration: 0.5,
        audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),

        // Baseline: All features disabled
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        enable_cfar_detection: false,

        ..Default::default()
    };

    let baseline_peaks =
        crate::fm::collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

    // Test with both Phase 1 and Phase 2 enabled (current defaults)
    let mut combined_generator = create_multi_signal_detection_scenario();
    let combined_config = ScanningConfig {
        // Use current defaults (both phases enabled)
        ..Default::default()
    };

    let combined_peaks =
        crate::fm::collect_peaks_from_source(&combined_config, &mut combined_generator)
            .expect("Failed to collect combined peaks");

    println!("Baseline detections: {}", baseline_peaks.len());
    println!("Combined (Phase 1+2) detections: {}", combined_peaks.len());

    let detection_ratio = combined_peaks.len() as f32 / baseline_peaks.len() as f32;
    println!(
        "Detection ratio (Combined / Baseline): {:.2}",
        detection_ratio
    );

    // This is the critical regression test - combined phases should not cause massive detection loss
    assert!(
        detection_ratio >= 0.5, // Allow up to 50% reduction, but this indicates a serious problem
        "Combined Phase 1+2 should not reduce detection count by more than 50%. Got {:.1}% reduction (ratio: {:.2})",
        (1.0 - detection_ratio) * 100.0,
        detection_ratio
    );

    // Warn if we see significant reduction
    if detection_ratio < 0.8 {
        println!(
            "🚨 WARNING: Combined phases reduced detection by {:.1}% - investigate Phase 1/2 interaction",
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
