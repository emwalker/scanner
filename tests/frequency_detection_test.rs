use scanner::types::ScanningConfig;

/// Test frequency detection accuracy with band scanning
#[test]
fn test_band_scan_frequency_detection() {
    let iq_filename = "tests/data/iq/scan/88.9_MHz_band_scan-1s-test1.iq";

    // Skip test if I/Q file doesn't exist yet
    if !std::path::Path::new(iq_filename).exists() {
        eprintln!("Skipping test - I/Q file not found: {}", iq_filename);
        return;
    }

    let (mut file_source, metadata) =
        scanner::testing::load_iq_fixture(iq_filename).expect("Failed to load I/Q fixture");

    let mut config = ScanningConfig::default();
    config.fft_size = metadata.fft_size;
    config.peak_detection_threshold = metadata.peak_detection_threshold;
    config.samp_rate = metadata.sample_rate;

    let peaks = scanner::fm::collect_peaks_from_source(&config, &mut file_source)
        .expect("Failed to collect peaks from I/Q file");

    let candidates = scanner::fm::find_candidates(&peaks, &config, metadata.center_frequency);

    // Should find 88.9 MHz station within ±200 kHz tolerance
    let target_freq = 88.9e6;
    let tolerance = 200_000.0;

    let found_target = candidates
        .iter()
        .any(|c| (c.frequency_hz() - target_freq).abs() <= tolerance);

    assert!(
        found_target,
        "Expected to find 88.9 MHz station within ±200 kHz, found frequencies: {:?}",
        candidates
            .iter()
            .map(|c| c.frequency_hz() / 1e6)
            .collect::<Vec<_>>()
    );
}
