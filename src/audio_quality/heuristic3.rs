//! Rule-based Audio Quality Classification
//!
//! This module provides rule-based audio quality classification using handcrafted features
//! and explicit decision rules.

use tracing::debug;

/// Rule-based audio quality classifier
pub struct Classifier {
    sample_rate: f32,
}

impl Classifier {
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Extract comprehensive audio features
    fn extract_features(&self, samples: &[f32]) -> crate::types::Result<super::AudioFeatures> {
        if samples.is_empty() {
            return Err(crate::types::ScannerError::Custom(
                "Empty audio samples".to_string(),
            ));
        }

        // 1. Energy-based features
        let rms_energy = self.compute_rms_energy(samples);
        let peak_amplitude = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let dynamic_range = self.compute_dynamic_range(samples);

        // 2. Spectral features (simplified FFT-based)
        let spectral_features = self.compute_spectral_features(samples)?;

        // 3. Temporal features
        let zero_crossing_rate = self.compute_zero_crossing_rate(samples);
        let silence_ratio = self.compute_silence_ratio(samples);

        // 4. Quality indicators
        let snr_estimate = self.estimate_snr(samples);
        let harmonic_ratio = self.compute_harmonic_ratio(samples)?;

        Ok(super::AudioFeatures {
            rms_energy,
            peak_amplitude,
            dynamic_range,
            spectral_centroid: spectral_features.0,
            spectral_rolloff: spectral_features.1,
            spectral_flux: spectral_features.2,
            high_freq_energy: spectral_features.3,
            zero_crossing_rate,
            silence_ratio,
            snr_estimate,
            harmonic_ratio,
        })
    }

    /// Classify audio quality with confidence and reasoning
    fn classify_quality(
        &self,
        features: &super::AudioFeatures,
    ) -> (super::AudioQuality, f32, String) {
        debug!(
            rms = features.rms_energy,
            peak = features.peak_amplitude,
            zcr = features.zero_crossing_rate,
            snr = features.snr_estimate,
            harmonic = features.harmonic_ratio,
            "Classifying audio with features"
        );

        // Rule-based classification with confidence scoring
        let mut confidence: f32 = 0.0;
        let mut reasoning_parts = Vec::new();

        // Primary signal strength check
        if features.rms_energy < 0.05 {
            return (
                super::AudioQuality::Static,
                0.9,
                "Very low signal energy indicates static".to_string(),
            );
        }

        // SNR-based quality assessment
        let snr_quality = if features.snr_estimate > 25.0 {
            confidence += 0.3;
            reasoning_parts.push("High SNR");
            super::AudioQuality::Good
        } else if features.snr_estimate > 15.0 {
            confidence += 0.2;
            reasoning_parts.push("Moderate SNR");
            super::AudioQuality::Moderate
        } else if features.snr_estimate > 8.0 {
            confidence += 0.1;
            reasoning_parts.push("Low SNR");
            super::AudioQuality::Poor
        } else {
            reasoning_parts.push("Very low SNR");
            super::AudioQuality::Static
        };

        // Harmonic content assessment
        let harmonic_quality = if features.harmonic_ratio > 0.7 {
            confidence += 0.25;
            reasoning_parts.push("Strong harmonic content");
            super::AudioQuality::Good
        } else if features.harmonic_ratio > 0.4 {
            confidence += 0.15;
            reasoning_parts.push("Moderate harmonic content");
            super::AudioQuality::Moderate
        } else if features.harmonic_ratio > 0.2 {
            confidence += 0.05;
            reasoning_parts.push("Weak harmonic content");
            super::AudioQuality::Poor
        } else {
            reasoning_parts.push("No clear harmonic content");
            super::AudioQuality::Static
        };

        // Dynamic range assessment
        if features.dynamic_range > 20.0 {
            confidence += 0.2;
            reasoning_parts.push("Good dynamic range");
        } else if features.dynamic_range > 10.0 {
            confidence += 0.1;
            reasoning_parts.push("Moderate dynamic range");
        } else {
            reasoning_parts.push("Poor dynamic range");
        }

        // Zero crossing rate (lower is generally better for music)
        if features.zero_crossing_rate < 0.1 {
            confidence += 0.15;
            reasoning_parts.push("Low distortion");
        } else if features.zero_crossing_rate < 0.2 {
            confidence += 0.05;
            reasoning_parts.push("Some distortion");
        } else {
            reasoning_parts.push("High distortion");
        }

        // Silence ratio check
        if features.silence_ratio > 0.7 {
            return (
                super::AudioQuality::NoAudio,
                0.8,
                "High silence ratio indicates no audio content".to_string(),
            );
        }

        // Final classification based on best indicators
        let final_quality = match (snr_quality, harmonic_quality) {
            (super::AudioQuality::Good, super::AudioQuality::Good) => super::AudioQuality::Good,
            (super::AudioQuality::Good, _) | (_, super::AudioQuality::Good) => {
                super::AudioQuality::Moderate
            }
            (super::AudioQuality::Moderate, super::AudioQuality::Moderate) => {
                super::AudioQuality::Moderate
            }
            (super::AudioQuality::Poor, _) | (_, super::AudioQuality::Poor) => {
                super::AudioQuality::Poor
            }
            _ => super::AudioQuality::Static,
        };

        let reasoning = reasoning_parts.join(", ");
        (final_quality, confidence.clamp(0.0, 1.0), reasoning)
    }

    // Helper functions for feature computation
    fn compute_rms_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn compute_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let max = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let min = samples
            .iter()
            .map(|&x| x.abs())
            .fold(f32::INFINITY, f32::min);
        if min > 0.0 {
            20.0 * (max / min).log10()
        } else {
            0.0
        }
    }

    fn compute_spectral_features(
        &self,
        samples: &[f32],
    ) -> crate::types::Result<(f32, f32, f32, f32)> {
        // Simplified spectral analysis without full FFT
        // This is a placeholder implementation
        let spectral_centroid = samples.len() as f32 * 0.5;
        let spectral_rolloff = self.sample_rate * 0.4;
        let spectral_flux = self.compute_rms_energy(samples) * 0.1;
        let high_freq_energy = self.compute_rms_energy(samples) * 0.3;

        Ok((
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            high_freq_energy,
        ))
    }

    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
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

    fn compute_silence_ratio(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 1.0;
        }

        let threshold = 0.01;
        let silent_samples = samples.iter().filter(|&&x| x.abs() < threshold).count();
        silent_samples as f32 / samples.len() as f32
    }

    fn estimate_snr(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 0.0;
        }

        // Simple SNR estimation
        let signal_power = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;

        // Estimate noise as variance of signal
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let noise_power =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        if noise_power > 1e-10 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            30.0 // High SNR if no detectable noise
        }
    }

    fn compute_harmonic_ratio(&self, samples: &[f32]) -> crate::types::Result<f32> {
        if samples.len() < 100 {
            return Ok(0.0);
        }

        // Simplified harmonicity using autocorrelation
        let max_lag = 200.min(samples.len() / 4);
        let mut best_correlation = 0.0f32;

        for lag in 20..max_lag {
            let mut correlation = 0.0f32;
            let valid_samples = (samples.len() - lag).min(1000);

            for i in 0..valid_samples {
                correlation += samples[i] * samples[i + lag];
            }

            correlation /= valid_samples as f32;
            best_correlation = best_correlation.max(correlation.abs());
        }

        Ok(best_correlation.clamp(0.0, 1.0))
    }
}

impl super::Classifier for Classifier {
    fn analyze(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> crate::types::Result<super::QualityResult> {
        debug!(
            samples_len = samples.len(),
            sample_rate = sample_rate,
            "Starting heuristic3 rule-based audio quality analysis"
        );

        // Extract features
        let features = self.extract_features(samples)?;

        // Classify quality with confidence and reasoning
        let (quality, confidence, reasoning) = self.classify_quality(&features);

        debug!(
            quality = format!("{:?}", quality),
            confidence = confidence,
            reasoning = reasoning,
            "Heuristic3 rule-based analysis complete"
        );

        Ok(super::QualityResult {
            quality,
            confidence,
            signal_strength: features.rms_energy,
            features: Some(features),
        })
    }

    fn name(&self) -> &'static str {
        "heuristic3"
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_classifier_creation() {
        let classifier = super::Classifier::new(48000.0);
        assert_eq!(classifier.sample_rate, 48000.0);
    }

    #[test]
    fn test_feature_extraction() -> crate::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();

        let features = classifier.extract_features(&samples)?;

        assert!(features.rms_energy > 0.0);
        assert!(features.peak_amplitude > 0.0);
        assert!(features.zero_crossing_rate >= 0.0);

        Ok(())
    }

    #[test]
    fn test_empty_samples() {
        let classifier = super::Classifier::new(48000.0);
        let result = crate::audio_quality::Classifier::analyze(&classifier, &[], 48000.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_sine_wave_analysis() -> crate::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.1).sin()).collect();

        let result = crate::audio_quality::Classifier::analyze(&classifier, &samples, 48000.0)?;

        assert!(result.confidence > 0.0);
        assert!(result.signal_strength > 0.0);
        assert!(result.features.is_some());

        Ok(())
    }

    #[test]
    fn test_classifier_regression() -> crate::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);

        let overrides = [
            (
                "000.088.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.088.900.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.089.099.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.089.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.090.101.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.090.302.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.091.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.091.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.092.101.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.093.300.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.093.300.000Hz-wfm-002.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.093.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.093.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.094.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.095.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.096.300.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.096.900.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.097.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.098.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.099.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.100.300.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.100.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.102.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.103.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.103.900.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
            (
                "000.107.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Static,
            ),
        ];

        crate::testing::assert_classifies_audio(&classifier, &overrides)
    }
}
