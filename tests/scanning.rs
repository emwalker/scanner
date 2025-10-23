//! Scanning and signal processing tests

use rustradio::Complex;
use scanner::{
    core::types::{Band, ScanningConfig},
    testing::{MockSampleSource, SampleSource},
};
use tracing::debug;

#[test]
fn test_band_scanning_window_calculation() {
    let config = ScanningConfig {
        samp_rate: 1_000_000.0,
        ..Default::default()
    };

    let band = Band::Fm;
    let windows = band.windows(config.samp_rate, config.signal_processing.window_overlap);

    debug!("=== Band Scanning Window Analysis ===");
    debug!("Sample rate: {:.1} MHz", config.samp_rate / 1e6);
    debug!("Band: FM (88-108 MHz)");
    debug!("Number of windows: {}", windows.len());

    let target_freq = 88.9e6;
    let mut target_windows = Vec::new();

    for (i, window_center) in windows.iter().enumerate() {
        let usable_bandwidth = config.samp_rate * 0.8;
        let window_start = window_center - (usable_bandwidth / 2.0);
        let window_end = window_center + (usable_bandwidth / 2.0);

        debug!(
            "Window {}: Center {:.3} MHz, Range [{:.3} - {:.3}] MHz",
            i + 1,
            window_center / 1e6,
            window_start / 1e6,
            window_end / 1e6
        );

        if target_freq >= window_start && target_freq <= window_end {
            let offset = target_freq - window_center;
            target_windows.push((i + 1, *window_center, offset));
            debug!("  🎯 Contains 88.9 MHz (offset: {:.1} kHz)", offset / 1e3);
        }
    }

    assert!(
        !target_windows.is_empty(),
        "88.9 MHz should appear in at least one window"
    );

    debug!("✅ 88.9 MHz appears in {} window(s)", target_windows.len());
    for (window_num, center, offset) in &target_windows {
        debug!(
            "   Window {} (center: {:.3} MHz, offset: {:.1} kHz)",
            window_num,
            center / 1e6,
            offset / 1e3
        );
    }
}

#[test]
fn test_mock_sample_source_determinism() {
    let mut source1 = MockSampleSource::new(1_000_000.0, 88_900_000.0, 10000, 100_000.0);

    let mut source2 = MockSampleSource::new(1_000_000.0, 88_900_000.0, 10000, 100_000.0);

    let mut buffer1 = vec![Complex::new(0.0, 0.0); 1000];
    let mut buffer2 = vec![Complex::new(0.0, 0.0); 1000];

    let samples1 = source1
        .read_samples(&mut buffer1)
        .expect("Failed to read from source1");
    let samples2 = source2
        .read_samples(&mut buffer2)
        .expect("Failed to read from source2");

    assert_eq!(
        samples1, samples2,
        "Both sources should return same number of samples"
    );

    for i in 0..std::cmp::min(10, samples1) {
        let diff_real = (buffer1[i].re - buffer2[i].re).abs();
        let diff_imag = (buffer1[i].im - buffer2[i].im).abs();
        assert!(
            diff_real < 1e-10,
            "Sample {} real part differs: {} vs {}",
            i,
            buffer1[i].re,
            buffer2[i].re
        );
        assert!(
            diff_imag < 1e-10,
            "Sample {} imag part differs: {} vs {}",
            i,
            buffer1[i].im,
            buffer2[i].im
        );
    }

    println!("✅ MockSampleSource produces deterministic output");
    println!("   Generated {} samples per source", samples1);
    println!(
        "   First sample: {:.6} + {:.6}j",
        buffer1[0].re, buffer1[0].im
    );
}

#[test]
fn test_freq_xlating_fir_dc_signal_retention() {
    use rustradio::{block::Block, fir, stream::new_stream, window::WindowType};

    let samp_rate = 1_000_000.0;
    let taps = fir::low_pass(samp_rate, 400_000.0, 50_000.0, &WindowType::Hamming);

    let dc_signal = vec![Complex::new(0.5, 0.0); 2000];
    let input_power = 0.25;

    let (input, stream) = new_stream();
    let (mut filter, output) = scanner::signal::freq_xlating_fir::FreqXlatingFir::with_real_taps(
        stream, &taps, -200_000.0, samp_rate, 1,
    );

    {
        let mut input_buf = input.write_buf().unwrap();
        input_buf.slice()[..dc_signal.len()].copy_from_slice(&dc_signal);
        input_buf.produce(dc_signal.len(), &[]);
    }
    drop(input);

    let mut iterations = 0;
    loop {
        iterations += 1;
        if iterations > 10000 {
            println!(
                "Breaking after {} iterations to prevent infinite loop",
                iterations
            );
            break;
        }

        match filter.work() {
            Ok(rustradio::block::BlockRet::Again) => continue,
            Ok(rustradio::block::BlockRet::WaitForStream(..)) => continue,
            _ => break,
        }
    }

    let mut output_samples = Vec::new();
    while let Ok((buf, _)) = output.read_buf() {
        if buf.is_empty() {
            break;
        }
        let slice = buf.slice();
        output_samples.extend_from_slice(slice);
        let len = slice.len();
        buf.consume(len);
    }

    if output_samples.len() > 100 {
        let output_power = output_samples
            .iter()
            .skip(50)
            .take(100)
            .map(|s| s.re * s.re + s.im * s.im)
            .sum::<f32>()
            / 100.0;

        let retention = output_power / input_power;
        println!(
            "DC signal retention: {:.1}% (input: {:.3}, output: {:.3})",
            retention * 100.0,
            input_power,
            output_power
        );

        assert!(
            retention > 0.95,
            "FreqXlatingFir should preserve DC signals, got {:.1}%",
            retention * 100.0
        );
    }
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_peak_detection_with_synthetic_signal() {
    let mut sample_source = MockSampleSource::new(1_000_000.0, 89_000_000.0, 10000, 100_000.0);

    let mut config = ScanningConfig::default();
    config.samp_rate = sample_source.sample_rate();
    config.peak_detection.fft_size = 1024;
    config.peak_detection.threshold = 0.3;
    config
        .peak_detection
        .averaging
        .exponential_smoothing
        .enabled = false;
    config
        .peak_detection
        .averaging
        .multi_frame_averaging
        .enabled = false;
    config.peak_detection.averaging.coherent_integration_enabled = false;
    config.peak_detection.averaging.moving_average.enabled = false;
    config.peak_detection.cfar.enabled = false;
    config.peak_detection.windowing.enabled = false;
    config.peak_detection.windowing.zero_padding_factor = 1;
    config.peak_detection.multi_frame.enabled = false;

    let peaks = scanner::signal::peaks::collect_peaks_from_source(&config, &mut sample_source)
        .expect("Failed to collect peaks");

    println!("Found {} peaks from synthetic signal", peaks.len());
    for peak in &peaks {
        println!(
            "Peak: {:.1} MHz, magnitude: {:.3}",
            peak.frequency_hz / 1e6,
            peak.magnitude
        );
    }

    let expected_freq = 89_100_000.0;
    let found_expected = peaks.iter().any(|p| {
        let freq_diff = (p.frequency_hz - expected_freq).abs();
        freq_diff < 50_000.0
    });

    assert!(found_expected, "Should find peak near 89.1 MHz");
    assert!(!peaks.is_empty(), "Should find at least one peak");
}

#[test]
fn test_band_scan_frequency_detection() {
    let iq_filename = "tests/data/iq/scan/88.9_MHz_band_scan-1s-test1.iq";

    if !std::path::Path::new(iq_filename).exists() {
        eprintln!("Skipping test - I/Q file not found: {}", iq_filename);
        return;
    }

    let (mut file_source, metadata) =
        scanner::testing::load_iq_fixture(iq_filename).expect("Failed to load I/Q fixture");

    let mut config = ScanningConfig::default();
    config.peak_detection.fft_size = metadata.fft_size;
    config.peak_detection.threshold = metadata.peak_detection_threshold;
    config.samp_rate = metadata.sample_rate;

    let peaks = scanner::signal::peaks::collect_peaks_from_source(&config, &mut file_source)
        .expect("Failed to collect peaks from I/Q file");

    let signals = scanner::signal::find_signals(&peaks, &config, metadata.center_frequency);

    let target_freq = 88.9e6;
    let tolerance = 200_000.0;

    let found_target = signals
        .iter()
        .any(|c| (c.frequency_hz() - target_freq).abs() <= tolerance);

    assert!(
        found_target,
        "Expected to find 88.9 MHz station within ±200 kHz, found frequencies: {:?}",
        signals
            .iter()
            .map(|c| c.frequency_hz() / 1e6)
            .collect::<Vec<_>>()
    );
}
