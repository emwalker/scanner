use rustradio::Complex;
use scanner::testing::MockSampleSource;
use scanner::types::SampleSource;
use scanner::{Band, ScanningConfig};
use tracing::info;

/// Test the band scanning window calculation logic in isolation
#[test]
fn test_band_scanning_window_calculation() {
    let config = ScanningConfig {
        samp_rate: 1_000_000.0, // 1 MHz
        ..Default::default()
    };

    let band = Band::Fm;
    let windows = band.windows(config.samp_rate);

    info!("=== Band Scanning Window Analysis ===");
    info!("Sample rate: {:.1} MHz", config.samp_rate / 1e6);
    info!("Band: FM (88-108 MHz)");
    info!("Number of windows: {}", windows.len());

    // Test our known station frequency
    let target_freq = 88.9e6;
    let mut target_windows = Vec::new();

    for (i, window_center) in windows.iter().enumerate() {
        let usable_bandwidth = config.samp_rate * 0.8; // 800 kHz usable
        let window_start = window_center - (usable_bandwidth / 2.0);
        let window_end = window_center + (usable_bandwidth / 2.0);

        info!(
            "Window {}: Center {:.3} MHz, Range [{:.3} - {:.3}] MHz",
            i + 1,
            window_center / 1e6,
            window_start / 1e6,
            window_end / 1e6
        );

        // Check if target frequency falls in this window
        if target_freq >= window_start && target_freq <= window_end {
            let offset = target_freq - window_center;
            target_windows.push((i + 1, *window_center, offset));
            info!("  🎯 Contains 88.9 MHz (offset: {:.1} kHz)", offset / 1e3);
        }
    }

    // This is our reference implementation expectation
    assert!(
        !target_windows.is_empty(),
        "88.9 MHz should appear in at least one window"
    );

    // Verify the fundamental expectation that 88.9 MHz can be processed
    info!("✅ 88.9 MHz appears in {} window(s)", target_windows.len());
    for (window_num, center, offset) in &target_windows {
        info!(
            "   Window {} (center: {:.3} MHz, offset: {:.1} kHz)",
            window_num,
            center / 1e6,
            offset / 1e3
        );
    }
}

/// Test that MockSampleSource produces deterministic output
#[test]
fn test_mock_sample_source_determinism() {
    let mut source1 = MockSampleSource::new(
        1_000_000.0,  // 1 MHz sample rate
        88_900_000.0, // 88.9 MHz center freq
        10000,        // max samples
        100_000.0,    // 100 kHz signal frequency offset
    );

    let mut source2 = MockSampleSource::new(
        1_000_000.0,  // 1 MHz sample rate
        88_900_000.0, // 88.9 MHz center freq
        10000,        // max samples
        100_000.0,    // 100 kHz signal frequency offset
    );

    // Read samples from both sources
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

    // Compare first few samples
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

/// Test that FreqXlatingFir preserves DC signals (fundamental component test)
#[test]
fn test_freq_xlating_fir_dc_signal_retention() {
    use rustradio::{block::Block, fir, stream::new_stream, window::WindowType};

    // Create a wide filter that should preserve DC
    let samp_rate = 1_000_000.0;
    let taps = fir::low_pass(samp_rate, 400_000.0, 50_000.0, &WindowType::Hamming);

    // Create DC signal (constant value)
    let dc_signal = vec![Complex::new(0.5, 0.0); 2000];
    let input_power = 0.25; // 0.5^2

    let (input, stream) = new_stream();
    let (mut filter, output) = scanner::freq_xlating_fir::FreqXlatingFir::with_real_taps(
        stream, &taps, -200_000.0, samp_rate, 1,
    );

    // Send samples
    {
        let mut input_buf = input.write_buf().unwrap();
        input_buf.slice()[..dc_signal.len()].copy_from_slice(&dc_signal);
        input_buf.produce(dc_signal.len(), &[]);
    }
    drop(input); // Close input to signal end-of-stream

    // Process samples with iteration limit to prevent infinite loop
    let mut output_samples = Vec::new();
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

    // Collect output
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
            .skip(50) // Skip initial transient
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

/// Test peak detection with synthetic signal
#[test]
fn test_peak_detection_with_synthetic_signal() {
    // Create synthetic signal source with known frequency
    let mut sample_source = MockSampleSource::new(
        1_000_000.0,  // 1 MHz sample rate
        89_000_000.0, // 89.0 MHz center frequency
        10000,        // max samples
        100_000.0,    // 100 kHz offset (signal at 89.1 MHz)
    );

    let config = ScanningConfig {
        samp_rate: sample_source.sample_rate(),
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        ..Default::default()
    };

    // Process samples and find peaks
    let peaks = scanner::fm::collect_peaks_from_source(&config, &mut sample_source)
        .expect("Failed to collect peaks");

    println!("Found {} peaks from synthetic signal", peaks.len());
    for peak in &peaks {
        println!(
            "Peak: {:.1} MHz, magnitude: {:.3}",
            peak.frequency_hz / 1e6,
            peak.magnitude
        );
    }

    // Should find at least one peak near 89.1 MHz (89.0 + 0.1 offset)
    let expected_freq = 89_100_000.0;
    let found_expected = peaks.iter().any(|p| {
        let freq_diff = (p.frequency_hz - expected_freq).abs();
        freq_diff < 50_000.0 // Within 50 kHz tolerance
    });

    assert!(found_expected, "Should find peak near 89.1 MHz");
    assert!(!peaks.is_empty(), "Should find at least one peak");
}
