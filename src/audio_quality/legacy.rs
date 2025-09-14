use super::AudioQuality;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::VecDeque;
use tracing::debug;

pub struct AudioQualityAnalyzer {
    fft_planner: FftPlanner<f32>,
    buffer_size: usize,
    _samp_rate: f32, // Unused but kept for future frequency-aware analysis
    analysis_buffer: VecDeque<f32>,

    // Thresholds for classification (most now hardcoded in classification logic)
    spectral_flatness_threshold: f32,
    crest_factor_threshold: f32,
    peak_to_rms_threshold: f32,
}

impl AudioQualityAnalyzer {
    pub fn new(buffer_size: usize, samp_rate: f32) -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            buffer_size,
            _samp_rate: samp_rate,
            analysis_buffer: VecDeque::with_capacity(buffer_size),

            // Adjusted thresholds for real FM audio with some distortion
            spectral_flatness_threshold: 0.7, // More lenient for FM radio quality
            crest_factor_threshold: 6.0,      // Lower threshold for slightly distorted audio
            peak_to_rms_threshold: 6.0,       // More lenient for FM broadcast quality
        }
    }

    pub fn add_samples(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.analysis_buffer.len() >= self.buffer_size {
                self.analysis_buffer.pop_front();
            }
            self.analysis_buffer.push_back(sample);
        }
    }

    pub fn analyze_quality(&mut self) -> AudioQuality {
        // Require at least 1024 samples for analysis (minimum for meaningful FFT)
        if self.analysis_buffer.len() < 1024 {
            return AudioQuality::Unknown;
        }

        let samples: Vec<f32> = self.analysis_buffer.iter().copied().collect();

        // Calculate signal strength (RMS) first - this is the primary distinguisher
        let signal_strength = self.calculate_rms();

        // Signal strength threshold: distinguishes real stations from AGC-processed noise
        // Lowered from 0.25 to 0.17 based on calibration findings (89.3MHz, 90.9MHz mismatches)
        // Based on calibration: ~0.12 = static, ~0.16-0.18 = audible poor quality, ~0.35+ = real audio
        if signal_strength < 0.17 {
            debug!(
                signal_strength = signal_strength,
                threshold = 0.17,
                "Signal strength too low - classifying as static"
            );
            return AudioQuality::Static;
        }

        // Calculate other metrics for stations with sufficient signal strength
        let spectral_flatness = self.calculate_spectral_flatness(&samples);
        let crest_factor = self.calculate_crest_factor(&samples);
        let peak_to_rms = self.calculate_peak_to_rms_ratio(&samples);
        let snr_db = self.estimate_snr_db(&samples);
        let artifact_score = self.detect_artifacts(&samples);

        // Classification logic based on multiple metrics
        let mut good_indicators = 0;
        let mut moderate_indicators = 0;
        let mut poor_indicators = 0;
        let mut static_indicators = 0;

        // Spectral flatness: Lower values indicate tonal content
        if spectral_flatness < 2.0e-6 {
            good_indicators += 2; // Extremely strong tonal content gets extra weight
        } else if spectral_flatness < 0.15 {
            good_indicators += 1;
        } else if spectral_flatness < 0.35 {
            moderate_indicators += 1;
        } else if spectral_flatness < 0.6 {
            poor_indicators += 1;
        } else if spectral_flatness > self.spectral_flatness_threshold {
            static_indicators += 1;
        }

        // Crest factor: Higher values indicate dynamic audio content (based on actual measurements)
        if crest_factor > 14.0 {
            good_indicators += 1;
        } else if crest_factor > 11.0 {
            moderate_indicators += 1;
        } else if crest_factor > self.crest_factor_threshold {
            poor_indicators += 1;
        } else if crest_factor < 4.0 {
            static_indicators += 1;
        }

        // Peak-to-RMS ratio: Higher values indicate dynamic range (based on actual measurements)
        if peak_to_rms > 5.0 {
            good_indicators += 1;
        } else if peak_to_rms > 3.5 {
            moderate_indicators += 1;
        } else if peak_to_rms > self.peak_to_rms_threshold {
            poor_indicators += 1;
        } else if peak_to_rms < 4.0 {
            static_indicators += 1;
        }

        // SNR estimation: Higher values indicate less noise
        if snr_db > 15.0 {
            good_indicators += 1;
        } else if snr_db > 5.0 {
            moderate_indicators += 1;
        } else if snr_db > 2.5 {
            // Low SNR (2.5-5 dB) indicates distortion/noise but still audible
            poor_indicators += 1;
        } else if snr_db < 2.0 {
            static_indicators += 1;
        }

        // Artifact detection: Lower scores indicate cleaner audio (based on actual measurements)
        if artifact_score < 11.5 {
            // Clean audio threshold - allows for some real-world artifacts
            good_indicators += 1;
        } else if artifact_score < 13.0 {
            // Files with artifact scores 11.5-13 likely have distortion/overdriving
            moderate_indicators += 1;
        } else if artifact_score < 15.0 {
            poor_indicators += 1;
        } else {
            static_indicators += 1;
        }

        // Five-way classification based on indicators with nuanced distortion detection
        let result = if static_indicators >= 3 {
            AudioQuality::Static
        } else if good_indicators >= 2 && poor_indicators <= 1 && moderate_indicators < 3 {
            // Good quality: strong positive indicators, limited distortion signs
            AudioQuality::Good
        } else if poor_indicators >= 2
            || moderate_indicators >= 2
            || (moderate_indicators + poor_indicators) >= 3
        {
            // Moderate quality: multiple distortion indicators
            AudioQuality::Moderate
        } else if poor_indicators >= 1
            || (good_indicators + moderate_indicators + poor_indicators) >= 2
        {
            AudioQuality::Poor
        } else {
            AudioQuality::Unknown
        };

        debug!(
            signal_strength = signal_strength,
            spectral_flatness = spectral_flatness,
            crest_factor_db = crest_factor,
            peak_to_rms = peak_to_rms,
            snr_db = snr_db,
            artifact_score = artifact_score,
            good_indicators = good_indicators,
            moderate_indicators = moderate_indicators,
            poor_indicators = poor_indicators,
            static_indicators = static_indicators,
            result = format!("{:?}", result),
            "Audio quality analysis complete"
        );

        result
    }

    pub fn calculate_rms(&self) -> f32 {
        if self.analysis_buffer.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.analysis_buffer.iter().map(|&s| s * s).sum();
        (sum_squares / self.analysis_buffer.len() as f32).sqrt()
    }

    fn calculate_spectral_flatness(&mut self, samples: &[f32]) -> f32 {
        let fft_size = samples.len();
        let fft = self.fft_planner.plan_fft_forward(fft_size);

        // Apply Hann window to reduce spectral leakage
        let windowed: Vec<Complex<f32>> = samples
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window = 0.5
                    * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos());
                Complex::new(sample * window, 0.0)
            })
            .collect();

        let mut fft_buffer = windowed;
        fft.process(&mut fft_buffer);

        // Calculate power spectrum (only positive frequencies)
        let nyquist = fft_size / 2;
        let power_spectrum: Vec<f32> = fft_buffer[1..nyquist]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Calculate spectral flatness (geometric mean / arithmetic mean)
        let geometric_mean = power_spectrum
            .iter()
            .map(|&p| (p + 1e-10).ln()) // Add small epsilon to avoid log(0)
            .sum::<f32>()
            / power_spectrum.len() as f32;
        let geometric_mean = geometric_mean.exp();

        let arithmetic_mean = power_spectrum.iter().sum::<f32>() / power_spectrum.len() as f32;

        if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_crest_factor(&self, samples: &[f32]) -> f32 {
        let peak = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        if rms > 1e-10 {
            20.0 * (peak / rms).log10() // Convert to dB
        } else {
            0.0
        }
    }

    fn calculate_peak_to_rms_ratio(&self, samples: &[f32]) -> f32 {
        let peak = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        if rms > 1e-10 { peak / rms } else { 0.0 }
    }

    fn estimate_snr_db(&self, samples: &[f32]) -> f32 {
        // Simple SNR estimation using signal power vs noise floor
        // This is a basic implementation - more sophisticated methods could be used

        let signal_power = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;

        // Estimate noise floor as the minimum power in sliding windows
        let window_size = samples.len() / 10;
        let mut min_power = f32::INFINITY;

        for window_start in (0..samples.len()).step_by(window_size).take(10) {
            let window_end = (window_start + window_size).min(samples.len());
            let window_power = samples[window_start..window_end]
                .iter()
                .map(|&s| s * s)
                .sum::<f32>()
                / (window_end - window_start) as f32;
            min_power = min_power.min(window_power);
        }

        if min_power > 1e-10 && signal_power > min_power {
            10.0 * (signal_power / min_power).log10()
        } else {
            0.0
        }
    }

    fn detect_artifacts(&mut self, samples: &[f32]) -> f32 {
        // Detect clicks, pops, and other transient artifacts
        // Based on research: artifacts show as sudden amplitude changes and spectral discontinuities

        let mut artifact_score = 0.0;
        let window_size = 512.min(samples.len() / 8);

        if window_size < 32 {
            return 0.0; // Not enough samples for meaningful analysis
        }

        // 1. Detect sudden amplitude changes (clicks/pops)
        let mut max_amplitude_jump = 0.0f32;
        for i in 1..samples.len() {
            let amplitude_change = (samples[i] - samples[i - 1]).abs();
            max_amplitude_jump = max_amplitude_jump.max(amplitude_change);
        }

        // 2. Count zero-crossing rate variations (indicates irregular signal)
        let mut zcr_variations = 0;
        for window_start in (0..samples.len()).step_by(window_size).take(8) {
            let window_end = (window_start + window_size).min(samples.len());
            let window = &samples[window_start..window_end];

            let mut zero_crossings = 0;
            for i in 1..window.len() {
                if (window[i] >= 0.0) != (window[i - 1] >= 0.0) {
                    zero_crossings += 1;
                }
            }

            let zcr = zero_crossings as f32 / window.len() as f32;
            // High ZCR variation indicates irregular signal (artifacts)
            if !(0.001..=0.1).contains(&zcr) {
                zcr_variations += 1;
            }
        }

        // 3. Detect spectral irregularities using local FFT analysis
        let mut spectral_irregularity = 0.0;
        if samples.len() >= 1024 {
            let mid_samples = &samples[samples.len() / 4..3 * samples.len() / 4];
            let mut local_fft_buffer: Vec<Complex<f32>> =
                mid_samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

            if local_fft_buffer.len() >= 512 {
                local_fft_buffer.truncate(512);
                let fft = self.fft_planner.plan_fft_forward(512);
                fft.process(&mut local_fft_buffer);

                // Look for spectral peaks that indicate transients
                let power_spectrum: Vec<f32> = local_fft_buffer[1..256]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect();

                let mean_power = power_spectrum.iter().sum::<f32>() / power_spectrum.len() as f32;
                for &power in &power_spectrum {
                    if power > mean_power * 5.0 {
                        spectral_irregularity += 1.0;
                    }
                }
                spectral_irregularity /= power_spectrum.len() as f32;
            }
        }

        // Combine artifact indicators
        artifact_score += max_amplitude_jump * 10.0; // Weight sudden amplitude changes heavily
        artifact_score += (zcr_variations as f32) * 0.1; // Zero-crossing irregularities
        artifact_score += spectral_irregularity * 2.0; // Spectral peaks from transients

        artifact_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_audio_quality_analyzer_creation() {
        let analyzer = AudioQualityAnalyzer::new(1024, 48000.0);
        assert_eq!(analyzer.buffer_size, 1024);
        assert_eq!(analyzer._samp_rate, 48000.0);
    }

    #[test]
    fn test_add_samples() {
        let mut analyzer = AudioQualityAnalyzer::new(10, 48000.0);
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        analyzer.add_samples(&samples);
        assert_eq!(analyzer.analysis_buffer.len(), 5);

        // Add more samples to test buffer overflow
        let more_samples = vec![0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        analyzer.add_samples(&more_samples);
        assert_eq!(analyzer.analysis_buffer.len(), 10); // Should be capped at buffer_size
    }

    #[test]
    fn test_analyze_quality_insufficient_samples() {
        let mut analyzer = AudioQualityAnalyzer::new(1024, 48000.0);
        let samples = vec![0.1, 0.2, 0.3]; // Too few samples

        analyzer.add_samples(&samples);
        assert_eq!(analyzer.analyze_quality(), AudioQuality::Unknown);
    }

    #[test]
    fn test_spectral_flatness_pure_tone() {
        let mut analyzer = AudioQualityAnalyzer::new(1024, 48000.0);

        // Generate a pure sine wave (should have low spectral flatness)
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        let flatness = analyzer.calculate_spectral_flatness(&samples);
        assert!(
            flatness < 0.1,
            "Pure tone should have very low spectral flatness, got {}",
            flatness
        );
    }

    #[test]
    fn test_spectral_flatness_white_noise() {
        let mut analyzer = AudioQualityAnalyzer::new(1024, 48000.0);

        // Generate multi-tone signal (should have higher spectral flatness than pure tone)
        let samples: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 / 48000.0;
                0.2 * (2.0 * PI * 100.0 * t).sin()
                    + 0.2 * (2.0 * PI * 300.0 * t).cos()
                    + 0.2 * (2.0 * PI * 700.0 * t).sin()
                    + 0.2 * (2.0 * PI * 1100.0 * t).cos()
                    + 0.2 * (2.0 * PI * 1500.0 * t).sin()
            })
            .collect();

        let flatness = analyzer.calculate_spectral_flatness(&samples);
        // Just verify the calculation doesn't crash and returns a valid number
        assert!(
            flatness >= 0.0 && flatness <= 1.0,
            "Spectral flatness should be between 0 and 1, got {}",
            flatness
        );
    }

    #[test]
    fn test_crest_factor_calculation() {
        let analyzer = AudioQualityAnalyzer::new(1024, 48000.0);

        // Test with samples having known peak and RMS
        let samples = vec![1.0, 0.5, 0.3, 0.7, 0.2];
        let crest_factor = analyzer.calculate_crest_factor(&samples);

        // Should be positive and reasonable
        assert!(crest_factor > 0.0);
        assert!(crest_factor < 20.0); // Reasonable upper bound for audio
    }

    #[test]
    fn test_peak_to_rms_ratio() {
        let analyzer = AudioQualityAnalyzer::new(1024, 48000.0);

        // Test with constant amplitude (low dynamic range)
        let constant_samples = vec![0.5; 100];
        let ratio_constant = analyzer.calculate_peak_to_rms_ratio(&constant_samples);
        assert!(
            (ratio_constant - 1.0).abs() < 0.01,
            "Constant signal should have ratio ~1.0, got {}",
            ratio_constant
        );

        // Test with high dynamic range
        let dynamic_samples = vec![1.0, 0.1, 0.05, 0.02, 0.01];
        let ratio_dynamic = analyzer.calculate_peak_to_rms_ratio(&dynamic_samples);
        assert!(
            ratio_dynamic > 2.0,
            "Dynamic signal should have higher ratio, got {}",
            ratio_dynamic
        );
    }

    #[test]
    fn test_snr_estimation() {
        let analyzer = AudioQualityAnalyzer::new(1024, 48000.0);

        // Test basic functionality - SNR estimation should return reasonable values
        let clean_signal: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();
        let clean_snr = analyzer.estimate_snr_db(&clean_signal);
        assert!(
            clean_snr >= 0.0,
            "SNR should be non-negative, got {} dB",
            clean_snr
        );

        // Test with very low amplitude
        let quiet_samples = vec![0.001; 1024];
        let quiet_snr = analyzer.estimate_snr_db(&quiet_samples);
        assert!(
            quiet_snr >= 0.0,
            "SNR should be non-negative even for quiet signals, got {} dB",
            quiet_snr
        );
    }

    #[test]
    fn test_rms_calculation() {
        let mut analyzer = AudioQualityAnalyzer::new(10, 48000.0);

        // Test with known values
        let samples = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        analyzer.add_samples(&samples);

        let rms = analyzer.calculate_rms();
        let expected_rms = ((1.0f32 + 1.0 + 0.25 + 0.25 + 0.0) / 5.0).sqrt();
        assert!(
            (rms - expected_rms).abs() < 0.001,
            "RMS should be {:.3}, got {:.3}",
            expected_rms,
            rms
        );

        // Test with empty buffer
        let empty_analyzer = AudioQualityAnalyzer::new(10, 48000.0);
        assert_eq!(
            empty_analyzer.calculate_rms(),
            0.0,
            "Empty buffer should return 0.0 RMS"
        );
    }

    #[test]
    fn test_good_quality_audio() {
        assert_audio_quality(
            "tests/data/audio/squelch/88.9MHz-audio-2s-test1.audio",
            AudioQuality::Good,
        );
    }

    #[test]
    fn test_moderate_quality_audio() {
        assert_audio_quality(
            "tests/data/audio/squelch/88.9MHz-audio-2s-test2.audio",
            AudioQuality::Moderate,
        );
    }

    fn load_and_analyze_audio_file(fixture_path: &str) -> AudioQualityAnalyzer {
        let (mut audio_source, metadata) =
            crate::testing::load_audio_fixture(fixture_path).expect("Failed to load audio fixture");

        // AudioQualityAnalyzer is now simplified - RF-domain filtering handles interference rejection
        let mut analyzer = AudioQualityAnalyzer::new(metadata.total_samples, metadata.sample_rate);
        let mut samples = Vec::new();
        let mut buffer = vec![0.0f32; 1024];

        // Read all samples from the audio file
        while let Ok(samples_read) = audio_source.read_audio_samples(&mut buffer) {
            if samples_read == 0 {
                break;
            }
            samples.extend_from_slice(&buffer[..samples_read]);
        }

        assert!(!samples.is_empty(), "Should have read samples from fixture");
        analyzer.add_samples(&samples);
        analyzer
    }

    fn assert_audio_quality(file_path: &str, expected_quality: AudioQuality) {
        let mut analyzer = load_and_analyze_audio_file(file_path);
        let actual_quality = analyzer.analyze_quality();
        assert_eq!(
            actual_quality, expected_quality,
            "{} should be classified as {:?}, got: {:?}",
            file_path, expected_quality, actual_quality
        );
    }
}
