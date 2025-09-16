//! Fast Normalized Audio Quality Metrics
//!
//! This module provides gain-invariant and sample-rate-independent audio quality metrics
//! designed to preserve calibration results across different system parameters.

use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tracing::debug;

/// Fast normalized metrics classifier
pub struct Classifier {
    /// FFT processor for spectral analysis
    fft: Arc<dyn Fft<f32>>,
    /// Target sample rate for normalization
    target_sample_rate: f32,
    /// PCEN smoothing coefficient
    pcen_alpha: f32,
    /// PCEN gain normalization strength
    pcen_delta: f32,
}

impl Classifier {
    /// Create new normalized metrics classifier
    pub fn new(target_sample_rate: f32) -> Self {
        let fft_size = 1024;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        Self {
            fft,
            target_sample_rate,
            pcen_alpha: 0.98, // PCEN smoothing coefficient
            pcen_delta: 2.0,  // PCEN gain normalization strength
        }
    }

    /// Simple linear resampling to target sample rate
    fn resample_to_target(&self, samples: &[f32], original_rate: f32) -> Vec<f32> {
        let ratio = self.target_sample_rate / original_rate;
        let target_len = (samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(target_len);

        for i in 0..target_len {
            let original_index = (i as f32 / ratio) as usize;
            if original_index < samples.len() {
                resampled.push(samples[original_index]);
            } else {
                resampled.push(0.0);
            }
        }

        debug!(
            original_len = samples.len(),
            target_len = target_len,
            ratio = ratio,
            "Resampled audio to target sample rate"
        );

        resampled
    }

    /// Calculate RMS-normalized signal strength [0.0, 1.0]
    fn calculate_rms_normalized_strength(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let normalized_rms = rms.min(1.0);

        debug!(
            rms = rms,
            normalized_rms = normalized_rms,
            "RMS normalization"
        );
        normalized_rms
    }

    /// Calculate Scale-Invariant Signal-to-Distortion Ratio
    fn calculate_si_sdr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return f32::NEG_INFINITY;
        }

        let segment_size = samples.len() / 4;
        if segment_size == 0 {
            return 0.0;
        }

        let mut segment_powers = Vec::new();
        for chunk in samples.chunks(segment_size) {
            let power = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
            segment_powers.push(power);
        }

        segment_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let noise_power = segment_powers[0].max(1e-10);
        let signal_power = segment_powers[segment_powers.len() - 1].max(1e-10);
        let si_sdr_db = 10.0 * (signal_power / noise_power).log10();

        debug!(
            signal_power = signal_power,
            noise_power = noise_power,
            si_sdr_db = si_sdr_db,
            "SI-SDR calculation"
        );

        si_sdr_db
    }

    /// Calculate PCEN-normalized spectral flatness
    fn calculate_pcen_spectral_flatness(&self, samples: &[f32]) -> f32 {
        if samples.len() < self.fft.len() {
            return 0.0;
        }

        let mut fft_input: Vec<rustfft::num_complex::Complex<f32>> = samples[..self.fft.len()]
            .iter()
            .map(|&s| rustfft::num_complex::Complex::new(s, 0.0))
            .collect();

        self.fft.process(&mut fft_input);

        let power_spectrum: Vec<f32> = fft_input[..self.fft.len() / 2]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        let mut pcen_spectrum = Vec::with_capacity(power_spectrum.len());
        let mut smoothed_power = power_spectrum[0];

        for &power in &power_spectrum {
            smoothed_power = self.pcen_alpha * smoothed_power + (1.0 - self.pcen_alpha) * power;
            let normalized_power = power / (smoothed_power.powf(self.pcen_delta / 10.0) + 1e-10);
            pcen_spectrum.push(normalized_power);
        }

        let geometric_mean = pcen_spectrum.iter().map(|&p| (p + 1e-10).ln()).sum::<f32>()
            / pcen_spectrum.len() as f32;
        let geometric_mean = geometric_mean.exp();

        let arithmetic_mean = pcen_spectrum.iter().sum::<f32>() / pcen_spectrum.len() as f32;

        let spectral_flatness = if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        };

        debug!(
            geometric_mean = geometric_mean,
            arithmetic_mean = arithmetic_mean,
            spectral_flatness = spectral_flatness,
            "PCEN spectral flatness"
        );

        spectral_flatness
    }

    /// Estimate normalized SNR using spectral analysis
    fn estimate_normalized_snr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return f32::NEG_INFINITY;
        }

        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        let signal_power = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max).powi(2);
        let noise_power = variance.max(1e-10);

        let snr_db = 10.0 * (signal_power / noise_power).log10();

        debug!(
            signal_power = signal_power,
            noise_power = noise_power,
            snr_db = snr_db,
            "Normalized SNR estimation"
        );

        snr_db
    }

    /// Calculate temporal stability (consistency of signal over time)
    fn calculate_temporal_stability(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 1.0;
        }

        let window_size = samples.len() / 10;
        let mut window_rms_values = Vec::new();

        for window in samples.chunks(window_size) {
            let rms = (window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32).sqrt();
            window_rms_values.push(rms);
        }

        let mean_rms = window_rms_values.iter().sum::<f32>() / window_rms_values.len() as f32;
        let variance = window_rms_values
            .iter()
            .map(|&rms| (rms - mean_rms).powi(2))
            .sum::<f32>()
            / window_rms_values.len() as f32;

        let coefficient_of_variation = if mean_rms > 1e-10 {
            variance.sqrt() / mean_rms
        } else {
            1.0
        };

        let stability = (-coefficient_of_variation * 2.0).exp().min(1.0);

        debug!(
            mean_rms = mean_rms,
            coefficient_of_variation = coefficient_of_variation,
            stability = stability,
            "Temporal stability calculation"
        );

        stability
    }

    /// Calculate overall quality score combining all normalized metrics
    fn calculate_quality_score(
        &self,
        si_sdr_db: f32,
        normalized_signal_strength: f32,
        pcen_spectral_flatness: f32,
        normalized_snr_db: f32,
        temporal_stability: f32,
    ) -> f32 {
        let sdr_score = (si_sdr_db / 40.0 + 0.5).clamp(0.0, 1.0);
        let strength_score = normalized_signal_strength.clamp(0.0, 1.0);
        let flatness_score = (1.0 - pcen_spectral_flatness).clamp(0.0, 1.0);
        let snr_score = (normalized_snr_db / 30.0).clamp(0.0, 1.0);
        let stability_score = temporal_stability.clamp(0.0, 1.0);

        let quality_score = 0.25 * sdr_score
            + 0.20 * strength_score
            + 0.15 * flatness_score
            + 0.15 * snr_score
            + 0.25 * stability_score;

        debug!(
            sdr_score = sdr_score,
            strength_score = strength_score,
            flatness_score = flatness_score,
            snr_score = snr_score,
            stability_score = stability_score,
            final_quality_score = quality_score,
            "Quality score calculation"
        );

        quality_score
    }

    /// Classify audio quality based on quality score and signal strength
    fn classify_audio_quality(
        &self,
        quality_score: f32,
        normalized_signal_strength: f32,
    ) -> super::AudioQuality {
        if normalized_signal_strength < 0.1 {
            super::AudioQuality::Static
        } else if quality_score >= 0.8 {
            super::AudioQuality::Good
        } else if quality_score >= 0.6 {
            super::AudioQuality::Moderate
        } else if quality_score >= 0.3 {
            super::AudioQuality::Poor
        } else if normalized_signal_strength > 0.2 {
            super::AudioQuality::NoAudio
        } else {
            super::AudioQuality::Static
        }
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
            target_rate = self.target_sample_rate,
            "Starting heuristic2 audio quality analysis"
        );

        // Step 1: Resample to target sample rate if needed
        let normalized_samples = if (sample_rate - self.target_sample_rate).abs() > 1.0 {
            self.resample_to_target(samples, sample_rate)
        } else {
            samples.to_vec()
        };

        // Step 2: Calculate metrics
        let normalized_signal_strength =
            self.calculate_rms_normalized_strength(&normalized_samples);
        let si_sdr_db = self.calculate_si_sdr(&normalized_samples);
        let pcen_spectral_flatness = self.calculate_pcen_spectral_flatness(&normalized_samples);
        let normalized_snr_db = self.estimate_normalized_snr(&normalized_samples);
        let temporal_stability = self.calculate_temporal_stability(&normalized_samples);

        // Step 3: Calculate overall quality score
        let quality_score = self.calculate_quality_score(
            si_sdr_db,
            normalized_signal_strength,
            pcen_spectral_flatness,
            normalized_snr_db,
            temporal_stability,
        );

        // Step 4: Classify audio quality
        let audio_quality = self.classify_audio_quality(quality_score, normalized_signal_strength);

        debug!(
            si_sdr_db = si_sdr_db,
            normalized_signal_strength = normalized_signal_strength,
            pcen_spectral_flatness = pcen_spectral_flatness,
            normalized_snr_db = normalized_snr_db,
            temporal_stability = temporal_stability,
            quality_score = quality_score,
            audio_quality = format!("{:?}", audio_quality),
            "Heuristic2 audio quality analysis complete"
        );

        Ok(super::QualityResult {
            quality: audio_quality,
            confidence: quality_score,
            signal_strength: normalized_signal_strength,
            features: None, // This classifier doesn't extract detailed features
        })
    }

    fn name(&self) -> &'static str {
        "heuristic2"
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_classifier_creation() {
        let classifier = super::Classifier::new(48000.0);
        assert_eq!(classifier.target_sample_rate, 48000.0);
    }

    #[test]
    fn test_rms_normalization() {
        let classifier = super::Classifier::new(48000.0);

        let samples = vec![0.5; 1000];
        let normalized_strength = classifier.calculate_rms_normalized_strength(&samples);
        assert!((normalized_strength - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_gain_invariance() {
        let classifier = super::Classifier::new(48000.0);

        let base_samples: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
        let gained_samples_2x: Vec<f32> = base_samples.iter().map(|&s| s * 2.0).collect();
        let gained_samples_half: Vec<f32> = base_samples.iter().map(|&s| s * 0.5).collect();

        let result_base =
            crate::audio_quality::Classifier::analyze(&classifier, &base_samples, 48000.0).unwrap();
        let result_2x =
            crate::audio_quality::Classifier::analyze(&classifier, &gained_samples_2x, 48000.0)
                .unwrap();
        let result_half =
            crate::audio_quality::Classifier::analyze(&classifier, &gained_samples_half, 48000.0)
                .unwrap();

        // Quality should be similar across different gains
        assert!((result_base.confidence - result_2x.confidence).abs() < 0.3);
        assert!((result_base.confidence - result_half.confidence).abs() < 0.3);
    }

    #[test]
    fn test_sample_rate_handling() {
        let classifier = super::Classifier::new(48000.0);

        let samples_44k: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();

        let result_44k =
            crate::audio_quality::Classifier::analyze(&classifier, &samples_44k, 44100.0).unwrap();
        let result_48k =
            crate::audio_quality::Classifier::analyze(&classifier, &samples_44k, 48000.0).unwrap();

        // Results should be comparable despite different input sample rates
        assert!((result_44k.confidence - result_48k.confidence).abs() < 0.2);
    }

    #[test]
    fn test_classifier_regression() -> crate::types::Result<()> {
        let classifier = super::Classifier::new(48000.0);

        let overrides = [
            (
                "000.087.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.088.099.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.088.299.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.088.300.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.088.499.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.088.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.088.900.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.089.099.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.089.299.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.089.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.090.101.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.091.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.091.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.091.702.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.093.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.094.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.094.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.096.300.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.097.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.098.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.099.100.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.099.500.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.100.700.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
            (
                "000.101.100.000Hz-wfm-002.wav",
                crate::audio_quality::AudioQuality::Poor,
            ),
            (
                "000.103.900.000Hz-wfm-001.wav",
                crate::audio_quality::AudioQuality::Moderate,
            ),
        ];

        crate::testing::assert_classifies_audio(&classifier, &overrides)
    }
}
