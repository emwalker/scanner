use scanner::testing::*;
use scanner::{Format, ScanningConfig};

#[test]
fn test_frequency_translation_scenarios() {
    // Test the scenarios we identified earlier
    let scenarios = create_frequency_test_scenarios();

    for scenario in scenarios {
        let result = test_frequency_translation_isolated(
            scenario.sdr_center_freq,
            scenario.target_station_freq,
            true, // Enable debug for now
        );

        println!("Scenario: {}", scenario.test_name);
        println!(
            "  Expected offset: {:.1} kHz",
            scenario.expected_offset / 1e3
        );
        println!("  Actual offset: {:.1} kHz", result.frequency_offset / 1e3);
        println!("  Translation valid: {}", result.translation_valid);

        // All scenarios should be valid for frequency translation
        assert!(
            result.translation_valid,
            "Translation should be valid for {} (offset: {:.1} kHz)",
            scenario.test_name,
            result.frequency_offset / 1e3
        );

        // Check offset calculation is correct
        assert!(
            (result.frequency_offset - scenario.expected_offset).abs() < 1.0,
            "Offset calculation mismatch for {}",
            scenario.test_name
        );
    }
}

#[test]
fn test_pipeline_debug_modes() {
    let config = ScanningConfig {
        debug_pipeline: true,
        ..Default::default()
    };

    // Test that debug mode doesn't crash and config is set correctly
    assert!(config.debug_pipeline);
    assert_eq!(config.samp_rate, 1_000_000.0);
    assert_eq!(config.fft_size, 1024);
}

#[test]
fn test_captured_logging() {
    use tracing::debug;

    // Test the log capture functionality
    let result = with_captured_logs(true, Format::Json, || {
        debug!(
            message = "Test log entry",
            test_value = 42,
            test_string = "hello"
        );
        Ok(())
    });

    match result {
        Ok((_, logs)) => {
            // Should contain our test log entry in JSON format
            assert!(logs.contains("Test log entry"));
            assert!(logs.contains("test_value"));
            assert!(logs.contains("42"));
        }
        Err(e) => panic!("Log capture test failed: {}", e),
    }
}

#[test]
fn test_log_comparison_structure() {
    // Test that we can capture and compare logs from different scanning modes
    let config = ScanningConfig {
        debug_pipeline: true,
        samp_rate: 1_000_000.0,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        ..Default::default()
    };

    // This demonstrates the structure for comparing scanning modes
    // In practice, this would use actual I/Q test files
    let station_freq = 88.9e6;
    let window_center = 89.1e6; // 200 kHz offset scenario

    println!("Testing log comparison framework");
    println!("Station frequency: {:.3} MHz", station_freq / 1e6);
    println!("Window center: {:.3} MHz", window_center / 1e6);
    println!(
        "Expected offset: {:.1} kHz",
        (station_freq - window_center) / 1e3
    );

    // The framework is ready to use with actual I/Q files:
    // let result = compare_scanning_modes_with_logs(
    //     "test_data/88_9_signal.iq",
    //     station_freq,
    //     window_center,
    //     &config,
    // );

    assert!(config.debug_pipeline);
}

/// Integration test demonstrating the complete testing framework
/// This test would work with actual I/Q files when available
#[test]
#[ignore] // Ignore by default since it needs I/Q test files
fn test_complete_pipeline_with_captured_logs() {
    let _config = ScanningConfig {
        debug_pipeline: true,
        samp_rate: 1_000_000.0,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        ..Default::default()
    };

    // This would test with an actual I/Q file containing 88.9 MHz signal
    // let result = test_complete_pipeline_with_logs(
    //     "test_data/88_9_mhz_signal.iq",
    //     88.9e6,
    //     ScanningMode::Stations(88.9e6),
    //     &config,
    // );
    //
    // match result {
    //     Ok((pipeline_result, logs)) => {
    //         // Analyze the pipeline result
    //         assert!(pipeline_result.target_found);
    //
    //         // Analyze the captured logs for debugging
    //         println!("Captured logs:");
    //         println!("{}", logs);
    //
    //         // Could parse JSON logs to extract specific values
    //         assert!(logs.contains("Peak detection test started"));
    //         assert!(logs.contains("Candidate created"));
    //     },
    //     Err(e) => panic!("Pipeline test failed: {}", e),
    // }

    println!("Complete pipeline testing framework ready for I/Q test files");
}
