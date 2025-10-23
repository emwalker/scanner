use super::*;
use crate::{audio::quality::AudioAnalyzer, core::types::Peak, testing::*};

#[test]
fn test_band_scanning_windows() {
    use types::Band;

    let config = ScanningConfig::default();
    let band = Band::Fm;
    let windows = band.windows(config.samp_rate, config.signal_processing.window_overlap);

    println!("\n=== FM Band Window Analysis ===");
    println!("Sample rate: {} MHz", config.samp_rate / 1e6);
    println!("Number of windows: {}", windows.len());

    let target_station = 88.9e6;

    for (i, window_center) in windows.iter().enumerate() {
        let window_start = window_center - (config.samp_rate * 0.8 / 2.0);
        let window_end = window_center + (config.samp_rate * 0.8 / 2.0);

        if target_station >= window_start && target_station <= window_end {
            let offset = target_station - window_center;
            println!(
                "🎯 Window {}: Center {:.1} MHz covers 88.9 MHz (offset: {:.1} kHz)",
                i + 1,
                window_center / 1e6,
                offset / 1e3
            );

            if offset.abs() > 75_000.0 {
                println!("⚠️  This offset exceeds our filter bandwidth!");
            }
        }
    }
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_collect_peaks_from_mock_source() {
    let mut config = ScanningConfig::default();
    config.duration = 1;
    config.peak_detection.fft_size = 1024;
    config.peak_detection.threshold = 0.01;
    config.peak_detection.scan_duration = 0.1;
    config.samp_rate = 1000000.0;
    config.signal_processing.frequency_tracking.disabled = true;
    config.audio.analyzer = AudioAnalyzer::mock();

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

    let mut mock_source = MockSampleSource::new(1000000.0, 88900000.0, 100000, 100000.0);

    let peaks = crate::signal::peaks::collect_peaks_from_source(&config, &mut mock_source).unwrap();

    assert!(!peaks.is_empty(), "Should detect at least one peak");

    let target_freq = 89000000.0;
    let found_peak = peaks
        .iter()
        .find(|p| (p.frequency_hz - target_freq).abs() < 50000.0);
    assert!(
        found_peak.is_some(),
        "Should find peak near 89.0 MHz, found peaks at: {:?}",
        peaks
            .iter()
            .map(|p| p.frequency_hz / 1e6)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_sidelobe_discrimination_rejects_legitimate_fm_signal() {
    let peaks = vec![
        Peak {
            frequency_hz: 88_700_000.0,
            magnitude: 7.672,
        },
        Peak {
            frequency_hz: 88_702_000.0,
            magnitude: 7.749,
        },
        Peak {
            frequency_hz: 88_704_000.0,
            magnitude: 7.496,
        },
        Peak {
            frequency_hz: 88_706_000.0,
            magnitude: 7.334,
        },
        Peak {
            frequency_hz: 88_708_000.0,
            magnitude: 4.706,
        },
        Peak {
            frequency_hz: 88_710_000.0,
            magnitude: 6.123,
        },
        Peak {
            frequency_hz: 88_712_000.0,
            magnitude: 5.892,
        },
        Peak {
            frequency_hz: 88_714_000.0,
            magnitude: 5.234,
        },
        Peak {
            frequency_hz: 88_716_000.0,
            magnitude: 4.987,
        },
        Peak {
            frequency_hz: 88_718_000.0,
            magnitude: 4.123,
        },
        Peak {
            frequency_hz: 88_720_000.0,
            magnitude: 3.856,
        },
    ];

    let target_freq_mhz = 88.9;
    let sample_rate = 2_000_000.0;
    let center_freq = 89.2e6;

    let (score, analysis_summary) = candidates::analysis::analyze_spectral_characteristics(
        &peaks,
        target_freq_mhz,
        sample_rate,
        center_freq,
    );

    assert!(
        score > 0.0,
        "Fixed algorithm should accept legitimate FM signal with 20 kHz span. Score: {:.3}, \
         Analysis: '{}'",
        score,
        analysis_summary
    );
    assert!(
        !analysis_summary.contains("Narrow spectral width (sidelobe?)"),
        "Should not classify 20 kHz span as narrow/sidelobe. Analysis: '{}'",
        analysis_summary
    );
}

#[test]
fn test_frequency_rounding_100khz() {
    let test_cases = vec![
        (87_700_000.0, 87_700_000.0),
        (87_749_999.0, 87_700_000.0),
        (87_750_000.0, 87_800_000.0),
        (87_750_001.0, 87_800_000.0),
        (87_799_999.0, 87_800_000.0),
        (87_800_000.0, 87_800_000.0),
        (93_125_000.0, 93_100_000.0),
        (93_175_000.0, 93_200_000.0),
        (93_149_999.0, 93_100_000.0),
        (93_150_000.0, 93_200_000.0),
    ];

    for (input_hz, expected_hz) in test_cases {
        let rounded = (input_hz / 100000.0f64).round() * 100000.0f64;
        assert_eq!(
            rounded, expected_hz,
            "Failed rounding {:.0} Hz to nearest 100 kHz. Expected {:.0}, got {:.0}",
            input_hz, expected_hz, rounded
        );

        assert_eq!(
            (rounded as u64) % 100_000,
            0,
            "Rounded frequency {:.0} Hz is not aligned to 100 kHz boundary",
            rounded
        );
    }
}

#[test]
fn test_clear_processed_frequencies_with_concurrent_reads() {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

    state::clear_processed_frequencies();

    let should_stop = Arc::new(AtomicBool::new(false));
    let mut handles = vec![];

    for _ in 0..5 {
        let stop_flag = should_stop.clone();
        let handle = thread::spawn(move || {
            while !stop_flag.load(Ordering::SeqCst) {
                if let Ok(processed) = state::PROCESSED_FREQUENCIES.read() {
                    let _ = processed.contains(&88900);
                }
                thread::sleep(Duration::from_micros(10));
            }
        });
        handles.push(handle);
    }

    thread::sleep(Duration::from_millis(20));

    for _ in 0..10 {
        state::clear_processed_frequencies();
        thread::sleep(Duration::from_micros(100));
    }

    should_stop.store(true, Ordering::SeqCst);

    for handle in handles {
        handle.join().expect("Reader thread panicked");
    }
}
