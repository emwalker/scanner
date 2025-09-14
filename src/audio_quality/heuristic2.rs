use crate::audio_quality::AudioQuality;
use hound::{SampleFormat, WavReader};
use std::path::Path;
use tracing::debug;

/// Audio quality classifier using handcrafted features
pub struct AudioQualityClassifier {
    sample_rate: f32,
}

/// Comprehensive audio features for quality assessment
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    // Energy-based features
    pub rms_energy: f32,
    pub peak_amplitude: f32,
    pub dynamic_range: f32,

    // Spectral features
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub spectral_flux: f32,
    pub high_freq_energy: f32,

    // Temporal features
    pub zero_crossing_rate: f32,
    pub silence_ratio: f32,

    // Quality indicators
    pub snr_estimate: f32,
    pub harmonic_ratio: f32,
}

/// Rich result structure containing quality classification and supporting information
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// Primary audio quality classification
    pub quality: AudioQuality,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Extracted audio features
    pub features: AudioFeatures,
    /// Human-readable explanation of classification reasoning
    pub reasoning: String,
}

impl AudioQualityClassifier {
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Load WAV file and return audio samples
    pub fn load_wav_file<P: AsRef<Path>>(&self, path: P) -> crate::types::Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref()).map_err(|e| {
            crate::types::ScannerError::Custom(format!("Failed to open WAV file: {}", e))
        })?;

        let spec = reader.spec();
        debug!(
            path = %path.as_ref().display(),
            channels = spec.channels,
            sample_rate = spec.sample_rate,
            bits_per_sample = spec.bits_per_sample,
            sample_format = ?spec.sample_format,
            "Loading WAV file"
        );

        let mut samples = Vec::new();
        match spec.sample_format {
            SampleFormat::Float => {
                for sample in reader.samples::<f32>() {
                    samples.push(sample.map_err(|e| {
                        crate::types::ScannerError::Custom(format!("Failed to read sample: {}", e))
                    })?);
                }
            }
            SampleFormat::Int => {
                let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
                for sample in reader.samples::<i32>() {
                    let s = sample.map_err(|e| {
                        crate::types::ScannerError::Custom(format!("Failed to read sample: {}", e))
                    })?;
                    samples.push(s as f32 / max_val);
                }
            }
        }

        // Convert to mono if stereo by averaging channels
        if spec.channels == 2 {
            let mono_samples: Vec<f32> = samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect();
            samples = mono_samples;
        }

        debug!(sample_count = samples.len(), "Loaded WAV samples");
        Ok(samples)
    }

    /// Extract comprehensive audio features
    pub fn extract_features(&self, samples: &[f32]) -> crate::types::Result<AudioFeatures> {
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

        Ok(AudioFeatures {
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
    pub fn classify_quality(&self, features: &AudioFeatures) -> QualityResult {
        debug!(
            rms = features.rms_energy,
            peak = features.peak_amplitude,
            zcr = features.zero_crossing_rate,
            snr = features.snr_estimate,
            harmonic = features.harmonic_ratio,
            "Classifying audio with features"
        );

        // Rule-based classification with confidence scoring
        let scores = [
            self.score_static(features),
            self.score_no_audio(features),
            self.score_poor(features),
            self.score_moderate(features),
            self.score_good(features),
        ];

        // Find best classification
        let max_score_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let quality = match max_score_idx {
            0 => AudioQuality::Static,
            1 => AudioQuality::NoAudio,
            2 => AudioQuality::Poor,
            3 => AudioQuality::Moderate,
            4 => AudioQuality::Good,
            _ => AudioQuality::Unknown,
        };

        let max_score = scores[max_score_idx];
        let confidence = max_score.clamp(0.0, 1.0);

        // Generate reasoning
        let reasoning = self.generate_reasoning(&quality, features, confidence);

        QualityResult {
            quality,
            confidence,
            features: features.clone(),
            reasoning,
        }
    }

    /// Predict audio quality from raw samples
    pub fn predict(&self, samples: &[f32]) -> crate::types::Result<QualityResult> {
        let features = self.extract_features(samples)?;
        Ok(self.classify_quality(&features))
    }

    // Scoring functions for each quality level
    fn score_static(&self, features: &AudioFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Very low energy indicates static/noise
        if features.rms_energy < 0.02 {
            score += 0.3;
        }
        if features.rms_energy < 0.01 {
            score += 0.3;
        }

        // Low SNR indicates noisy content
        if features.snr_estimate < 8.0 {
            score += 0.3;
        }
        if features.snr_estimate < 5.0 {
            score += 0.2;
        }

        // High zero-crossing rate indicates noise
        if features.zero_crossing_rate > 0.1 {
            score += 0.2;
        }

        // Low harmonic content indicates non-tonal noise
        if features.harmonic_ratio < 0.15 {
            score += 0.3;
        }

        score.min(1.0)
    }

    fn score_no_audio(&self, features: &AudioFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Very low energy but not quite static
        if features.rms_energy < 0.03 && features.rms_energy > 0.005 {
            score += 0.4;
        }

        // Low harmonic content but some structure
        if features.harmonic_ratio < 0.2 && features.harmonic_ratio > 0.05 {
            score += 0.3;
        }

        // High silence ratio
        if features.silence_ratio > 0.8 {
            score += 0.3;
        }

        // Moderate SNR (not pure noise, but no clear audio)
        if features.snr_estimate > 5.0 && features.snr_estimate < 12.0 {
            score += 0.2;
        }

        score.min(1.0)
    }

    fn score_poor(&self, features: &AudioFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Poor audio: moderate RMS (~0.26), high SNR (~20.1), moderate harmonic (~0.42)
        if features.rms_energy > 0.15 && features.rms_energy < 0.4 {
            score += 0.3;
        }
        if features.rms_energy > 0.2 && features.rms_energy < 0.35 {
            score += 0.2;
        }

        // Surprisingly high SNR but moderate dynamic range
        if features.snr_estimate > 15.0
            && features.snr_estimate < 25.0
            && features.dynamic_range < 0.5
        {
            score += 0.4;
        }

        // Moderate harmonic content
        if features.harmonic_ratio > 0.3 && features.harmonic_ratio < 0.5 {
            score += 0.3;
        }

        score.min(1.0)
    }

    fn score_moderate(&self, features: &AudioFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Moderate audio: high RMS (~0.47), high SNR (~24.8), good harmonic (~0.46), high dynamic range (~0.89)
        if features.rms_energy > 0.35 && features.rms_energy < 0.6 {
            score += 0.3;
        }
        if features.rms_energy > 0.4 && features.rms_energy < 0.55 {
            score += 0.2;
        }

        // High SNR with good dynamic range
        if features.snr_estimate > 22.0 && features.dynamic_range > 0.6 {
            score += 0.4;
        }
        if features.snr_estimate > 24.0 && features.dynamic_range > 0.8 {
            score += 0.3;
        }

        // Good harmonic content
        if features.harmonic_ratio > 0.4 && features.harmonic_ratio < 0.5 {
            score += 0.2;
        }

        score.min(1.0)
    }

    fn score_good(&self, features: &AudioFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Good audio: moderate RMS (~0.42), high SNR (~22.4), high harmonic (~0.48), high dynamic range (~0.78)
        if features.rms_energy > 0.25 && features.rms_energy < 0.5 {
            score += 0.3;
        }
        if features.rms_energy > 0.3 && features.rms_energy < 0.45 {
            score += 0.2;
        }

        // High SNR with excellent dynamic range
        if features.snr_estimate > 20.0 && features.dynamic_range > 0.6 {
            score += 0.3;
        }
        if features.snr_estimate > 22.0 && features.dynamic_range > 0.7 {
            score += 0.3;
        }

        // Highest harmonic content
        if features.harmonic_ratio > 0.45 {
            score += 0.3;
        }
        if features.harmonic_ratio > 0.48 {
            score += 0.2;
        }

        score.min(1.0)
    }

    fn generate_reasoning(
        &self,
        quality: &AudioQuality,
        features: &AudioFeatures,
        confidence: f32,
    ) -> String {
        let mut reasons = Vec::new();

        match quality {
            AudioQuality::Static => {
                if features.rms_energy < 0.01 {
                    reasons.push("very low energy level");
                }
                if features.snr_estimate < 5.0 {
                    reasons.push("poor signal-to-noise ratio");
                }
                if features.harmonic_ratio < 0.15 {
                    reasons.push("no tonal content");
                }
            }
            AudioQuality::NoAudio => {
                if features.silence_ratio > 0.8 {
                    reasons.push("mostly silent");
                }
                if features.harmonic_ratio < 0.2 {
                    reasons.push("no clear audio content");
                }
            }
            AudioQuality::Poor => {
                if features.snr_estimate < 15.0 {
                    reasons.push("noisy background");
                }
                if features.dynamic_range < 0.25 {
                    reasons.push("compressed/limited audio");
                }
                if features.harmonic_ratio < 0.4 {
                    reasons.push("unclear audio content");
                }
            }
            AudioQuality::Moderate => {
                reasons.push("decent signal quality");
                if features.snr_estimate > 10.0 {
                    reasons.push("acceptable noise level");
                }
                if features.harmonic_ratio > 0.25 {
                    reasons.push("clear audio content");
                }
            }
            AudioQuality::Good => {
                if features.snr_estimate > 15.0 {
                    reasons.push("high signal-to-noise ratio");
                }
                if features.harmonic_ratio > 0.4 {
                    reasons.push("clear tonal content");
                }
                if features.dynamic_range > 0.25 {
                    reasons.push("good dynamic range");
                }
            }
            AudioQuality::Unknown => {
                reasons.push("insufficient data for classification");
            }
        }

        let confidence_desc = match confidence {
            c if c > 0.8 => "very confident",
            c if c > 0.6 => "confident",
            c if c > 0.4 => "moderately confident",
            _ => "low confidence",
        };

        format!(
            "{} ({}): {}",
            quality.to_human_string(),
            confidence_desc,
            reasons.join(", ")
        )
    }

    // Feature computation methods
    fn compute_rms_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    fn compute_dynamic_range(&self, samples: &[f32]) -> f32 {
        let sorted_samples: Vec<f32> = {
            let mut s: Vec<f32> = samples.iter().map(|x| x.abs()).collect();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s
        };

        let percentile_95 = sorted_samples[(0.95 * samples.len() as f32) as usize];
        let percentile_5 = sorted_samples[(0.05 * samples.len() as f32) as usize];

        percentile_95 - percentile_5
    }

    fn compute_spectral_features(
        &self,
        samples: &[f32],
    ) -> crate::types::Result<(f32, f32, f32, f32)> {
        use rustfft::{FftPlanner, num_complex::Complex};

        let n_fft = 1024.min(samples.len());
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Prepare FFT input
        let mut fft_input: Vec<Complex<f32>> = samples
            .iter()
            .take(n_fft)
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft_input.resize(n_fft, Complex::new(0.0, 0.0));

        // Compute FFT
        fft.process(&mut fft_input);

        // Compute magnitude spectrum
        let magnitudes: Vec<f32> = fft_input.iter().take(n_fft / 2).map(|c| c.norm()).collect();

        let total_energy: f32 = magnitudes.iter().sum();
        if total_energy == 0.0 {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        // Spectral centroid (weighted frequency mean)
        let spectral_centroid = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &mag)| i as f32 * mag)
            .sum::<f32>()
            / total_energy;

        // Spectral rolloff (frequency below which 85% of energy lies)
        let mut cumulative_energy = 0.0;
        let target_energy = 0.85 * total_energy;
        let mut rolloff_bin = 0;
        for (i, &mag) in magnitudes.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= target_energy {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff = rolloff_bin as f32;

        // Spectral flux (rate of change of spectrum)
        let spectral_flux = magnitudes
            .windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum::<f32>()
            / (magnitudes.len() - 1) as f32;

        // High frequency energy (> 8kHz)
        let high_freq_start = (8000.0 * n_fft as f32 / self.sample_rate) as usize;
        let high_freq_energy = magnitudes.iter().skip(high_freq_start).sum::<f32>() / total_energy;

        Ok((
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            high_freq_energy,
        ))
    }

    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        let crossings = samples
            .windows(2)
            .filter(|window| window[0] * window[1] < 0.0)
            .count();
        crossings as f32 / (samples.len() - 1) as f32
    }

    fn compute_silence_ratio(&self, samples: &[f32]) -> f32 {
        let threshold = 0.01; // Silence threshold
        let silent_samples = samples.iter().filter(|&&x| x.abs() < threshold).count();
        silent_samples as f32 / samples.len() as f32
    }

    fn estimate_snr(&self, samples: &[f32]) -> f32 {
        let rms = self.compute_rms_energy(samples);

        // Estimate noise floor from quietest 10% of samples
        let mut sorted_samples: Vec<f32> = samples.iter().map(|x| x.abs()).collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let noise_samples = &sorted_samples[..sorted_samples.len() / 10];
        let noise_rms = if !noise_samples.is_empty() {
            let noise_sum: f32 = noise_samples.iter().map(|x| x * x).sum();
            (noise_sum / noise_samples.len() as f32).sqrt()
        } else {
            0.001
        };

        // SNR in dB
        if noise_rms > 0.0 {
            20.0 * (rms / noise_rms).log10()
        } else {
            60.0 // Very high SNR if no noise detected
        }
    }

    fn compute_harmonic_ratio(&self, samples: &[f32]) -> crate::types::Result<f32> {
        use rustfft::{FftPlanner, num_complex::Complex};

        let n_fft = 512.min(samples.len());
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut fft_input: Vec<Complex<f32>> = samples
            .iter()
            .take(n_fft)
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft_input.resize(n_fft, Complex::new(0.0, 0.0));

        fft.process(&mut fft_input);

        let magnitudes: Vec<f32> = fft_input.iter().take(n_fft / 2).map(|c| c.norm()).collect();

        // Find spectral peaks (harmonics)
        let mut peak_energy = 0.0;
        let total_energy: f32 = magnitudes.iter().sum();

        if total_energy == 0.0 {
            return Ok(0.0);
        }

        // Simple peak detection
        for i in 1..magnitudes.len() - 1 {
            if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
                peak_energy += magnitudes[i];
            }
        }

        Ok(peak_energy / total_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = AudioQualityClassifier::new(48000.0);
        assert_eq!(classifier.sample_rate, 48000.0);
    }

    #[test]
    fn test_feature_extraction() {
        let classifier = AudioQualityClassifier::new(48000.0);

        // Test with sine wave (should be good quality)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();

        let features = classifier.extract_features(&samples).unwrap();

        // Sine wave should have reasonable harmonic ratio and moderate energy
        assert!(features.harmonic_ratio > 0.2); // Pure sine wave gives ~0.26
        assert!(features.rms_energy > 0.1);
        assert!(features.snr_estimate > 10.0);
    }

    #[test]
    fn test_classification() {
        let classifier = AudioQualityClassifier::new(48000.0);

        // Test noise (should be static)
        let noise_samples: Vec<f32> = (0..4800)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.01)
            .collect();

        let result = classifier.predict(&noise_samples).unwrap();

        // Should classify as static or no audio
        assert!(matches!(
            result.quality,
            AudioQuality::Static | AudioQuality::NoAudio
        ));
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning.is_empty());
    }
}
