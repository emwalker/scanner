//! Phase 2 Tests: CFAR Detection
//! Test-driven implementation for Constant False Alarm Rate detection

use super::signal_generation::{PeakTestSignalGenerator, TestSignal};
use crate::types::ScanningConfig;

/// Test that CFAR threshold adapts correctly to noise level changes
#[test]
fn test_cfar_threshold_adapts_to_noise_level() {
    let _ = tracing_subscriber::fmt::try_init();

    let config = create_test_config();

    // Test with low noise scenario
    let mut low_noise_generator = create_low_noise_scenario();
    let low_noise_peaks = crate::fm::collect_peaks_from_source(&config, &mut low_noise_generator)
        .expect("Failed to collect low noise peaks");

    // Test with high noise scenario
    let mut high_noise_generator = create_high_noise_scenario();
    let high_noise_peaks = crate::fm::collect_peaks_from_source(&config, &mut high_noise_generator)
        .expect("Failed to collect high noise peaks");

    // CFAR should maintain consistent detection rates despite noise level changes
    let low_noise_count = low_noise_peaks.len();
    let high_noise_count = high_noise_peaks.len();

    // With CFAR, detection counts should be more consistent across noise levels
    let detection_variance = ((low_noise_count as f32 - high_noise_count as f32).abs()
        / (low_noise_count + high_noise_count) as f32)
        * 100.0;

    println!("Low noise detections: {}", low_noise_count);
    println!("High noise detections: {}", high_noise_count);
    println!("Detection variance: {:.1}%", detection_variance);

    // CFAR should keep detection variance under 20%
    assert!(
        detection_variance < 20.0,
        "CFAR should maintain consistent detection across noise levels, got {:.1}% variance",
        detection_variance
    );
}

/// Test that guard bands prevent signal leakage into noise estimation
#[test]
fn test_cfar_guard_bands_prevent_signal_leakage() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();
    config.enable_cfar_detection = true;
    config.cfar_guard_cells = 10; // Guard cells around target

    // Create scenario with strong interfering signal adjacent to target
    let mut generator = create_adjacent_interference_scenario();
    let peaks = crate::fm::collect_peaks_from_source(&config, &mut generator)
        .expect("Failed to collect peaks with interference");

    // Find the target signal (should still be detected despite adjacent interference)
    let target_frequency = 88_700_000.0;
    let target_peak = peaks
        .iter()
        .find(|p| (p.frequency_hz - target_frequency).abs() < 10_000.0);

    assert!(
        target_peak.is_some(),
        "Guard bands should prevent interference from affecting target signal detection"
    );

    // The interfering signal should not bias the noise floor estimate
    if let Some(peak) = target_peak {
        // Target should have reasonable magnitude (not suppressed by interference)
        assert!(
            peak.magnitude > 100.0, // Reasonable threshold
            "Guard bands should preserve target signal strength"
        );
    }
}

/// Test that false alarm rate stays constant across noise levels
#[test]
// TODO: Tune CFAR parameters for consistent false alarm rate
fn test_cfar_constant_false_alarm_rate() {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_test_config();
    config.enable_cfar_detection = true;
    config.cfar_false_alarm_rate = 0.01; // 1% false alarm rate
    config.peak_scan_duration = 0.3; // Even faster for this test

    let num_trials = 3; // Reduced from 10 to 3 for faster testing
    let mut false_alarm_rates = Vec::new();

    // Test across fewer noise levels for faster execution
    for noise_level in [0.2, 0.5, 0.8] {
        let mut false_alarms = 0;
        let mut total_detections = 0;

        for _ in 0..num_trials {
            let mut generator = create_noise_only_scenario(noise_level);
            let peaks = crate::fm::collect_peaks_from_source(&config, &mut generator)
                .expect("Failed to collect noise-only peaks");

            // In noise-only scenario, all detections are false alarms
            false_alarms += peaks.len();
            total_detections += peaks.len();
        }

        let observed_far = if total_detections > 0 {
            false_alarms as f32 / (num_trials * 1000) as f32 // Approximate bins scanned
        } else {
            0.0
        };

        false_alarm_rates.push(observed_far);
        println!("Noise level {:.1}: FAR = {:.4}", noise_level, observed_far);
    }

    // CFAR should maintain consistent false alarm rate across noise levels
    let far_variance = calculate_variance(&false_alarm_rates);
    assert!(
        far_variance < 0.001, // Low variance in false alarm rate
        "CFAR should maintain constant false alarm rate, variance = {:.6}",
        far_variance
    );
}

/// Test that CFAR maintains detection performance despite interference (masking resistance)
#[test]
fn test_cfar_resists_strong_signal_masking() {
    let _ = tracing_subscriber::fmt::try_init();

    // Test cell-averaging CFAR vs fixed threshold in presence of strong interferer
    let mut baseline_config = create_test_config();
    baseline_config.enable_cfar_detection = false;
    baseline_config.peak_detection_threshold = 3.0; // Fixed threshold optimized for no interference

    // Test scenario: Clean environment without strong interferer (baseline should work well)
    let mut clean_generator = create_clean_weak_signals_scenario();
    let baseline_clean_peaks =
        crate::fm::collect_peaks_from_source(&baseline_config, &mut clean_generator)
            .expect("Failed to collect baseline clean peaks");

    // Test scenario: Environment with strong interferer (baseline should suffer masking)
    let mut interfered_generator = create_interfered_weak_signals_scenario();
    let baseline_interfered_peaks =
        crate::fm::collect_peaks_from_source(&baseline_config, &mut interfered_generator)
            .expect("Failed to collect baseline interfered peaks");

    // Test with CFAR enabled - should be more resistant to interference
    let mut cfar_config = baseline_config.clone();
    cfar_config.enable_cfar_detection = true;
    cfar_config.cfar_threshold_factor = 2.5; // Reasonable threshold factor
    cfar_config.cfar_guard_cells = 5; // Standard guard cells to prevent signal leakage
    cfar_config.cfar_reference_cells = 20; // Standard reference cells for noise estimation

    let mut cfar_interfered_generator = create_interfered_weak_signals_scenario();
    let cfar_interfered_peaks =
        crate::fm::collect_peaks_from_source(&cfar_config, &mut cfar_interfered_generator)
            .expect("Failed to collect CFAR interfered peaks");

    // Count weak signal detections (excluding the strong interferer)
    let weak_signal_frequencies = [88_750_000.0, 88_900_000.0, 89_100_000.0]; // Weak signals only
    let tolerance = 25_000.0; // 25 kHz tolerance

    let baseline_clean_detections =
        count_target_detections(&baseline_clean_peaks, &weak_signal_frequencies, tolerance);
    let baseline_interfered_detections = count_target_detections(
        &baseline_interfered_peaks,
        &weak_signal_frequencies,
        tolerance,
    );
    let cfar_interfered_detections =
        count_target_detections(&cfar_interfered_peaks, &weak_signal_frequencies, tolerance);

    println!(
        "Baseline (clean environment): {}/3 weak signals",
        baseline_clean_detections
    );
    println!(
        "Baseline (with interference): {}/3 weak signals",
        baseline_interfered_detections
    );
    println!(
        "CFAR (with interference): {}/3 weak signals",
        cfar_interfered_detections
    );

    // CFAR should maintain detection performance despite interference
    assert!(
        cfar_interfered_detections >= baseline_interfered_detections,
        "CFAR should not perform worse than fixed threshold with interference"
    );

    // CFAR should resist masking better than fixed threshold when interference is present
    let fixed_threshold_degradation =
        baseline_clean_detections as f32 - baseline_interfered_detections as f32;
    let cfar_vs_baseline_clean =
        baseline_clean_detections as f32 - cfar_interfered_detections as f32;

    assert!(
        cfar_vs_baseline_clean <= fixed_threshold_degradation,
        "CFAR should show less performance degradation due to interference than fixed threshold"
    );
}

// Helper functions for CFAR tests

fn create_test_config() -> ScanningConfig {
    ScanningConfig {
        audio_buffer_size: 8192,
        scanning_windows: Some(2),
        fft_size: 1024,
        peak_scan_duration: 0.5, // Fast for testing
        audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),

        // Phase 1: Disable signal averaging features to isolate Phase 2 testing
        enable_exponential_smoothing: false,
        enable_multi_frame_averaging: false,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,

        // Phase 2: Enable CFAR for testing
        enable_cfar_detection: true, // Enable CFAR for testing
        cfar_threshold_factor: 3.0,  // Lower factor for testing (3 dB above noise)
        cfar_guard_cells: 5,         // Smaller guard cells for fast testing
        cfar_reference_cells: 20,    // Smaller reference cells for fast testing
        ..Default::default()
    }
}

fn create_low_noise_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.1,          // Low noise level
    );

    // Add test signals at known frequencies
    generator.add_signal(TestSignal::new(88_700_000.0, 0.3, "Signal1"));
    generator.add_signal(TestSignal::new(88_900_000.0, 0.25, "Signal2"));
    generator.add_signal(TestSignal::new(89_100_000.0, 0.2, "Signal3"));

    generator
}

fn create_high_noise_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.6,          // High noise level
    );

    // Same signals as low noise, but in high noise environment
    generator.add_signal(TestSignal::new(88_700_000.0, 0.3, "Signal1"));
    generator.add_signal(TestSignal::new(88_900_000.0, 0.25, "Signal2"));
    generator.add_signal(TestSignal::new(89_100_000.0, 0.2, "Signal3"));

    generator
}

fn create_adjacent_interference_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.2,          // Moderate noise
    );

    // Target signal
    generator.add_signal(TestSignal::new(88_700_000.0, 0.2, "Target"));
    // Strong adjacent interferer
    generator.add_signal(TestSignal::new(88_750_000.0, 0.8, "Interferer"));

    generator
}

fn create_noise_only_scenario(noise_level: f32) -> PeakTestSignalGenerator {
    PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        noise_level,  // Variable noise level, no signals
    )
}

fn create_clean_weak_signals_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.2,          // Low noise level
    );

    // Add weak signals without any strong interferer
    generator.add_signal(TestSignal::new(88_750_000.0, 0.15, "WeakSignal1"));
    generator.add_signal(TestSignal::new(88_900_000.0, 0.12, "WeakSignal2"));
    generator.add_signal(TestSignal::new(89_100_000.0, 0.10, "WeakSignal3"));

    generator
}

fn create_interfered_weak_signals_scenario() -> PeakTestSignalGenerator {
    let mut generator = PeakTestSignalGenerator::new(
        2_000_000.0,  // sample_rate
        88_500_000.0, // center_frequency
        1_000_000,    // max_samples (0.5 seconds)
        0.2,          // Low base noise level
    );

    // Add strong interfering signal that may cause masking
    generator.add_signal(TestSignal::new(88_650_000.0, 0.8, "StrongInterferer"));

    // Add weak signals that should be detectable despite interference
    generator.add_signal(TestSignal::new(88_750_000.0, 0.15, "WeakSignal1")); // Near interferer
    generator.add_signal(TestSignal::new(88_900_000.0, 0.12, "WeakSignal2")); // Medium distance
    generator.add_signal(TestSignal::new(89_100_000.0, 0.10, "WeakSignal3")); // Far from interferer

    generator
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

fn calculate_variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;

    values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32
}
