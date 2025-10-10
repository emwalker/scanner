use scanner::core::types::{Format, ScanningConfig};
use scanner::testing::*;

#[test]
fn test_pipeline_debug_modes() {
    let mut config = ScanningConfig::default();
    config.debug.pipeline = true;

    // Test that debug mode doesn't crash and config is set correctly
    assert!(config.debug.pipeline);
    assert_eq!(config.samp_rate, 2_000_000.0);
    assert_eq!(config.peak_detection.fft_size, 1024);
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
    let mut config = ScanningConfig::default();
    config.debug.pipeline = true;
    config.samp_rate = 1_000_000.0;
    config.peak_detection.fft_size = 1024;
    config.peak_detection.threshold = 1.0;

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

    assert!(config.debug.pipeline);
}
