//! Original Statistical Audio Quality Analyzer
//!
//! This module provides the original statistical approach to audio quality analysis,
//! converted from stateful to stateless operation for the unified interface.

use rustfft::{FftPlanner, num_complex::Complex};
use tracing::debug;

/// Original statistical audio quality classifier
pub struct Classifier {
    // Configuration thresholds
    spectral_flatness_threshold: f32,
    crest_factor_threshold: f32,
    peak_to_rms_threshold: f32,
}

impl Classifier {
    pub fn new(_sample_rate: f32) -> Self {
        Self {
            // Adjusted thresholds for real FM audio with some distortion
            spectral_flatness_threshold: 0.7, // More lenient for FM radio quality
            crest_factor_threshold: 6.0,      // Lower threshold for slightly distorted audio
            peak_to_rms_threshold: 6.0,       // More lenient for FM broadcast quality
        }
    }

    /// Calculate RMS (Root Mean Square) power
    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate spectral flatness using FFT
    fn calculate_spectral_flatness(&self, samples: &[f32]) -> f32 {
        if samples.len() < 1024 {
            return 1.0; // Assume flat spectrum for short signals
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(1024);

        // Take first 1024 samples and zero-pad if necessary
        let mut fft_input: Vec<Complex<f32>> = samples
            .iter()
            .take(1024)
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Zero-pad if needed
        while fft_input.len() < 1024 {
            fft_input.push(Complex::new(0.0, 0.0));
        }

        fft.process(&mut fft_input);

        // Calculate power spectrum (only positive frequencies)
        let power_spectrum: Vec<f32> = fft_input[1..512] // Skip DC component
            .iter()
            .map(|c| c.norm_sqr())
            .filter(|&p| p > 1e-10) // Filter out very small values
            .collect();

        if power_spectrum.is_empty() {
            return 1.0;
        }

        // Calculate geometric and arithmetic means
        let log_sum: f32 = power_spectrum.iter().map(|&p| p.ln()).sum();
        let geometric_mean = (log_sum / power_spectrum.len() as f32).exp();
        let arithmetic_mean = power_spectrum.iter().sum::<f32>() / power_spectrum.len() as f32;

        if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            1.0
        }
    }

    /// Calculate crest factor (peak-to-RMS ratio)
    fn calculate_crest_factor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let rms = self.calculate_rms(samples);

        if rms > 1e-10 { peak / rms } else { 0.0 }
    }

    /// Calculate peak-to-RMS ratio in dB
    fn calculate_peak_to_rms_ratio(&self, samples: &[f32]) -> f32 {
        let crest_factor = self.calculate_crest_factor(samples);
        if crest_factor > 1e-10 {
            20.0 * crest_factor.log10()
        } else {
            0.0
        }
    }

    /// Estimate SNR in dB using spectral analysis
    fn estimate_snr_db(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 0.0;
        }

        // Simple SNR estimation: signal power vs noise power
        let signal_power = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;

        // Estimate noise as variance
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let noise_power =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        if noise_power > 1e-10 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            30.0 // High SNR if no detectable noise
        }
    }

    /// Detect artifacts using higher-order statistics
    fn detect_artifacts(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 0.0;
        }

        // Calculate kurtosis as an artifact indicator
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        if variance < 1e-10 {
            return 0.0;
        }

        let fourth_moment =
            samples.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / samples.len() as f32;

        let kurtosis = fourth_moment / (variance * variance) - 3.0;

        // Higher kurtosis indicates more artifacts
        kurtosis.max(0.0)
    }

    /// Extract audio features for unified interface
    fn extract_features(&self, samples: &[f32]) -> super::AudioFeatures {
        let rms_energy = self.calculate_rms(samples);
        let peak_amplitude = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let _crest_factor = self.calculate_crest_factor(samples);
        let dynamic_range = self.calculate_peak_to_rms_ratio(samples);
        let spectral_flatness = self.calculate_spectral_flatness(samples);
        let snr_estimate = self.estimate_snr_db(samples);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);

        super::AudioFeatures {
            rms_energy,
            peak_amplitude,
            dynamic_range,
            spectral_centroid: 0.0, // Not calculated in this classifier
            spectral_rolloff: 0.0,  // Not calculated in this classifier
            spectral_flux: 0.0,     // Not calculated in this classifier
            high_freq_energy: 0.0,  // Not calculated in this classifier
            zero_crossing_rate,
            silence_ratio: 0.0, // Not calculated in this classifier
            snr_estimate,
            harmonic_ratio: 1.0 - spectral_flatness, // Inverse of spectral flatness as proxy
        }
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / samples.len() as f32
    }

    /// Classify audio quality using original heuristic approach
    fn classify_quality(&self, samples: &[f32]) -> (super::AudioQuality, f32) {
        if samples.len() < 1024 {
            return (super::AudioQuality::Unknown, 0.0);
        }

        let signal_strength = self.calculate_rms(samples);
        if signal_strength < 0.17 {
            debug!(
                signal_strength = signal_strength,
                threshold = 0.17,
                "Signal strength too low - classifying as static"
            );
            return (super::AudioQuality::Static, 0.9);
        }

        let indicators = self.calculate_quality_indicators(samples);
        let (quality, confidence) = self.determine_final_quality(indicators.clone());

        debug!(
            signal_strength = signal_strength,
            spectral_flatness = indicators.spectral_flatness,
            crest_factor = indicators.crest_factor,
            peak_to_rms = indicators.peak_to_rms,
            snr_db = indicators.snr_db,
            artifact_score = indicators.artifact_score,
            good_indicators = indicators.good_count,
            moderate_indicators = indicators.moderate_count,
            poor_indicators = indicators.poor_count,
            static_indicators = indicators.static_count,
            quality = format!("{:?}", quality),
            confidence = confidence,
            "Heuristic1 statistical analysis complete"
        );

        (quality, confidence)
    }

    fn calculate_quality_indicators(&self, samples: &[f32]) -> QualityIndicators {
        let spectral_flatness = self.calculate_spectral_flatness(samples);
        let crest_factor = self.calculate_crest_factor(samples);
        let peak_to_rms = self.calculate_peak_to_rms_ratio(samples);
        let snr_db = self.estimate_snr_db(samples);
        let artifact_score = self.detect_artifacts(samples);

        let mut indicators = QualityIndicators {
            spectral_flatness,
            crest_factor,
            peak_to_rms,
            snr_db,
            artifact_score,
            good_count: 0,
            moderate_count: 0,
            poor_count: 0,
            static_count: 0,
        };

        self.evaluate_spectral_flatness(spectral_flatness, &mut indicators);
        self.evaluate_crest_factor(crest_factor, &mut indicators);
        self.evaluate_peak_to_rms(peak_to_rms, &mut indicators);
        self.evaluate_snr(snr_db, &mut indicators);
        self.evaluate_artifacts(artifact_score, &mut indicators);

        indicators
    }

    fn evaluate_spectral_flatness(
        &self,
        spectral_flatness: f32,
        indicators: &mut QualityIndicators,
    ) {
        if spectral_flatness < 2.0e-6 {
            indicators.good_count += 2;
        } else if spectral_flatness < 0.15 {
            indicators.good_count += 1;
        } else if spectral_flatness < 0.35 {
            indicators.moderate_count += 1;
        } else if spectral_flatness < 0.6 {
            indicators.poor_count += 1;
        } else if spectral_flatness > self.spectral_flatness_threshold {
            indicators.static_count += 1;
        }
    }

    fn evaluate_crest_factor(&self, crest_factor: f32, indicators: &mut QualityIndicators) {
        if crest_factor > 14.0 {
            indicators.good_count += 1;
        } else if crest_factor > 11.0 {
            indicators.moderate_count += 1;
        } else if crest_factor > self.crest_factor_threshold {
            indicators.poor_count += 1;
        } else if crest_factor < 4.0 {
            indicators.static_count += 1;
        }
    }

    fn evaluate_peak_to_rms(&self, peak_to_rms: f32, indicators: &mut QualityIndicators) {
        if peak_to_rms > 22.0 {
            indicators.good_count += 1;
        } else if peak_to_rms > 18.0 {
            indicators.moderate_count += 1;
        } else if peak_to_rms > self.peak_to_rms_threshold {
            indicators.poor_count += 1;
        } else {
            indicators.static_count += 1;
        }
    }

    fn evaluate_snr(&self, snr_db: f32, indicators: &mut QualityIndicators) {
        if snr_db > 25.0 {
            indicators.good_count += 1;
        } else if snr_db > 15.0 {
            indicators.moderate_count += 1;
        } else if snr_db > 8.0 {
            indicators.poor_count += 1;
        } else {
            indicators.static_count += 1;
        }
    }

    fn evaluate_artifacts(&self, artifact_score: f32, indicators: &mut QualityIndicators) {
        if artifact_score > 5.0 {
            indicators.poor_count += 1;
        } else if artifact_score > 2.0 {
            indicators.moderate_count += 1;
        }
    }

    fn determine_final_quality(&self, indicators: QualityIndicators) -> (super::AudioQuality, f32) {
        let total_indicators = indicators.good_count
            + indicators.moderate_count
            + indicators.poor_count
            + indicators.static_count;
        let confidence = if total_indicators > 0 {
            (indicators.good_count + indicators.moderate_count + indicators.poor_count) as f32
                / total_indicators as f32
        } else {
            0.5
        };

        let quality = if indicators.static_count >= 2
            || (indicators.static_count > 0
                && indicators.good_count == 0
                && indicators.moderate_count == 0)
        {
            super::AudioQuality::Static
        } else if indicators.good_count >= 2
            || (indicators.good_count >= 1 && indicators.moderate_count >= 1)
        {
            super::AudioQuality::Good
        } else if indicators.moderate_count >= 2
            || indicators.moderate_count >= 1
            || indicators.good_count >= 1
        {
            super::AudioQuality::Moderate
        } else if indicators.poor_count >= 1 {
            super::AudioQuality::Poor
        } else {
            super::AudioQuality::Static
        };

        (quality, confidence)
    }
}

#[derive(Clone)]
struct QualityIndicators {
    spectral_flatness: f32,
    crest_factor: f32,
    peak_to_rms: f32,
    snr_db: f32,
    artifact_score: f32,
    good_count: i32,
    moderate_count: i32,
    poor_count: i32,
    static_count: i32,
}

impl super::Classifier for Classifier {
    fn analyze(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> crate::core::types::Result<super::QualityResult> {
        debug!(
            samples_len = samples.len(),
            sample_rate = sample_rate,
            "Starting heuristic1 statistical audio quality analysis"
        );

        // Extract features
        let features = self.extract_features(samples);

        // Classify quality using original approach
        let (quality, confidence) = self.classify_quality(samples);

        Ok(super::QualityResult {
            quality,
            confidence,
            signal_strength: features.rms_energy,
            features: Some(features),
        })
    }

    fn name(&self) -> &'static str {
        "heuristic1"
    }
}

#[cfg(test)]
mod tests {
    use crate::audio::quality::AudioQuality;

    #[test]
    fn test_classifier_creation() {
        let _classifier = super::Classifier::new(48000.0);
        // Classifier created successfully
    }

    #[test]
    fn test_rms_calculation() {
        let classifier = super::Classifier::new(48000.0);
        let samples = vec![0.5; 1000];
        let rms = classifier.calculate_rms(&samples);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_samples() {
        let classifier = super::Classifier::new(48000.0);
        let result = crate::audio::quality::Classifier::analyze(&classifier, &[], 48000.0).unwrap();

        assert_eq!(result.quality, AudioQuality::Unknown);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_short_samples() {
        let classifier = super::Classifier::new(48000.0);
        let samples = vec![0.1; 100]; // Too short for reliable analysis
        let result =
            crate::audio::quality::Classifier::analyze(&classifier, &samples, 48000.0).unwrap();

        assert_eq!(result.quality, AudioQuality::Unknown);
    }

    #[test]
    fn test_sine_wave_analysis() -> crate::core::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.1).sin()).collect();

        let result = crate::audio::quality::Classifier::analyze(&classifier, &samples, 48000.0)?;

        assert!(result.confidence > 0.0);
        assert!(result.signal_strength > 0.0);
        assert!(result.features.is_some());

        Ok(())
    }

    #[test]
    fn test_static_detection() -> crate::core::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);
        // Very low amplitude noise should be classified as static
        let samples: Vec<f32> = (0..4800)
            .map(|i| 0.01 * ((i as f32 * 0.001).sin()))
            .collect();

        let result = crate::audio::quality::Classifier::analyze(&classifier, &samples, 48000.0)?;

        // Should detect as static due to low signal strength
        assert_eq!(result.quality, AudioQuality::Static);

        Ok(())
    }

    #[test]
    fn test_classifier_regression() -> crate::core::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);

        let overrides = [
            ("000.088.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.088.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.088.900.000Hz-wfm-003.wav", AudioQuality::Moderate),
            ("000.089.099.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.089.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.089.700.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.090.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.090.100.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.090.101.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.090.302.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.090.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.091.100.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.091.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.092.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.092.101.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.092.300.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.092.300.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.092.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.092.900.000Hz-wfm-001.wav", AudioQuality::Good),
            ("000.093.300.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.093.300.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.093.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.093.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.093.900.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.093.900.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.094.100.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.094.900.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.094.900.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.095.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.095.700.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.095.700.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.096.100.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.096.100.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.096.300.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.096.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.096.700.000Hz-wfm-004.wav", AudioQuality::Static),
            ("000.096.900.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.096.900.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.096.900.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.097.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.097.300.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.098.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.099.100.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.099.100.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.099.100.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.100.300.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.100.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.100.700.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.100.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.101.300.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.101.300.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.101.700.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.102.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.102.500.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.102.500.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.103.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.103.500.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.103.500.000Hz-wfm-003.wav", AudioQuality::Static),
            ("000.103.900.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.104.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.105.100.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.105.100.000Hz-wfm-002.wav", AudioQuality::Static),
            ("000.105.500.000Hz-wfm-001.wav", AudioQuality::Static),
            ("000.106.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
            ("000.107.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ];

        crate::testing::assert_classifies_audio(&classifier, &overrides)
    }
}
