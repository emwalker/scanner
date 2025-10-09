//! Basic Peak Extraction
//!
//! This module implements basic peak detection from FFT magnitude spectra
//! using simple threshold-based detection.

use crate::types::Peak;

/// Extract peaks from magnitude spectrum using simple threshold detection
///
/// Detects local maxima in the magnitude spectrum that exceed the given threshold.
/// This is the baseline peak detection method used when CFAR is disabled.
pub fn extract_peaks_from_magnitudes(
    magnitudes: &[f32],
    threshold: f32,
    fft_size: usize,
    sample_rate: f64,
    center_freq: f64,
) -> Vec<Peak> {
    let mut peaks = Vec::new();

    // Detect peaks: local maxima above threshold
    for i in 1..magnitudes.len() - 1 {
        if magnitudes[i] > threshold
            && magnitudes[i] > magnitudes[i - 1]
            && magnitudes[i] > magnitudes[i + 1]
        {
            // Convert FFT bin to frequency
            let freq_offset = (i as f64 / fft_size as f64) * sample_rate;
            let freq_hz = center_freq - (sample_rate / 2.0) + freq_offset;

            // Round to nearest 100 kHz to eliminate floating point precision errors
            let freq_hz_rounded = (freq_hz / 100000.0).round() * 100000.0;

            peaks.push(Peak {
                frequency_hz: freq_hz_rounded,
                magnitude: magnitudes[i],
            });
        }
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_quality::AudioAnalyzer;
    use crate::testing::signal_generation::{PeakTestSignalGenerator, TestSignal};
    use crate::types::ScanningConfig;

    /// Test basic peak extraction functionality
    #[test]
    fn test_extract_peaks_from_magnitudes() {
        // Create a simple magnitude spectrum with known peaks
        let magnitudes = vec![
            1.0, 2.0, 10.0, 5.0, 1.0, // Peak at index 2
            2.0, 3.0, 15.0, 6.0, 2.0, // Peak at index 7
            1.0, 1.0, 1.0, 1.0, 1.0, // No peaks
        ];

        let peaks = extract_peaks_from_magnitudes(
            &magnitudes,
            5.0,          // threshold
            1024,         // fft_size
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_freq
        );

        // Should detect 2 peaks above threshold
        assert_eq!(peaks.len(), 2);

        // Check that peaks are correctly identified
        assert!(peaks[0].magnitude >= 10.0);
        assert!(peaks[1].magnitude >= 15.0);

        // Check that frequencies are reasonable (within expected range)
        for peak in &peaks {
            assert!(peak.frequency_hz > 87_000_000.0);
            assert!(peak.frequency_hz < 91_000_000.0);
        }
    }

    /// Test that threshold filtering works correctly
    #[test]
    fn test_threshold_filtering() {
        let magnitudes = vec![
            1.0, 2.0, 3.0, 2.0, 1.0, // Peak magnitude 3.0
            1.0, 2.0, 6.0, 2.0, 1.0, // Peak magnitude 6.0
            1.0, 2.0, 12.0, 2.0, 1.0, // Peak magnitude 12.0
        ];

        // Test with low threshold - should detect all peaks
        let peaks_low = extract_peaks_from_magnitudes(
            &magnitudes,
            2.5,          // threshold
            1024,         // fft_size
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_freq
        );
        assert_eq!(peaks_low.len(), 3);

        // Test with medium threshold - should detect 2 peaks
        let peaks_med = extract_peaks_from_magnitudes(
            &magnitudes,
            5.0,          // threshold
            1024,         // fft_size
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_freq
        );
        assert_eq!(peaks_med.len(), 2);

        // Test with high threshold - should detect 1 peak
        let peaks_high = extract_peaks_from_magnitudes(
            &magnitudes,
            10.0,         // threshold
            1024,         // fft_size
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_freq
        );
        assert_eq!(peaks_high.len(), 1);
        assert!(peaks_high[0].magnitude >= 12.0);
    }

    /// Test peak extraction with real signal generator
    #[test]
    fn test_extract_peaks_with_signal_generator() {
        let config = ScanningConfig {
            audio_buffer_size: 8192,
            scanning_windows: Some(2),
            fft_size: 1024,
            peak_scan_duration: 0.5,
            peak_detection_threshold: 1.0,
            audio_analyzer: AudioAnalyzer::mock(),

            // Use basic extraction (no CFAR, no averaging)
            enable_exponential_smoothing: false,
            enable_multi_frame_averaging: false,
            enable_coherent_integration: false,
            enable_moving_average_filter: false,
            enable_cfar_detection: false,

            ..Default::default()
        };

        let mut generator = create_test_signal_scenario();
        let peaks = crate::peaks::collect_peaks_from_source(&config, &mut generator)
            .expect("Failed to collect peaks");

        // Should detect the test signals
        assert!(!peaks.is_empty(), "Should detect at least one peak");

        // Check that detected peaks are in reasonable frequency range
        for peak in &peaks {
            assert!(peak.frequency_hz > 87_000_000.0);
            assert!(peak.frequency_hz < 91_000_000.0);
            assert!(peak.magnitude > 0.0);
        }
    }

    fn create_test_signal_scenario() -> PeakTestSignalGenerator {
        let mut generator = PeakTestSignalGenerator::new(
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_frequency
            1_000_000,    // max_samples (0.5 seconds)
            0.2,          // Low noise level
        );

        // Add test signals
        generator.add_signal(TestSignal::new(88_700_000.0, 0.3, "Signal1"));
        generator.add_signal(TestSignal::new(89_100_000.0, 0.25, "Signal2"));

        generator
    }
}
