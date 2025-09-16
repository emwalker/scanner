use crate::audio_quality::{AudioFeatures, AudioQuality, get_training_dataset};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::debug;

/// Audio quality classifier using Random Forest algorithm
pub struct AudioQualityClassifier {
    sample_rate: f32,
    model: Option<RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>>>,
}

/// Result from prediction
#[derive(Debug, Clone)]
pub struct QualityResult {
    pub quality: AudioQuality,
    pub confidence: f32,
    pub features: AudioFeatures,
    pub model_scores: Vec<f32>, // Raw model prediction scores
}

impl AudioQualityClassifier {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            model: None,
        }
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

        // 2. Spectral features
        let spectral_centroid = self.compute_spectral_centroid(samples);
        let spectral_rolloff = self.compute_spectral_rolloff(samples);
        let spectral_flux = self.compute_spectral_flux(samples);
        let high_freq_energy = self.compute_high_freq_energy(samples);

        // 3. Temporal features
        let zero_crossing_rate = self.compute_zero_crossing_rate(samples);
        let silence_ratio = self.compute_silence_ratio(samples);

        // 4. Quality indicators
        let snr_estimate = self.estimate_snr(samples);
        let harmonic_ratio = self.compute_harmonic_ratio(samples);

        Ok(AudioFeatures {
            rms_energy,
            peak_amplitude,
            dynamic_range,
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            high_freq_energy,
            zero_crossing_rate,
            silence_ratio,
            snr_estimate,
            harmonic_ratio,
        })
    }

    /// Train the ML model on training data
    pub fn train(&mut self) -> crate::types::Result<()> {
        debug!("Training real ML classifier on handcrafted features");

        let training_data = get_training_dataset();
        let mut features_matrix = Vec::new();
        let mut labels = Vec::new();

        // Extract features from all training samples
        for (filename, expected_quality) in training_data.iter() {
            let wav_path = std::path::PathBuf::from("tests/data/audio/quality").join(filename);

            if !wav_path.exists() {
                debug!(filename = %filename, "Training file not found, skipping");
                continue;
            }

            let audio_samples = match crate::wave::load_file(&wav_path) {
                Ok(samples) => samples,
                Err(e) => {
                    debug!(filename = %filename, error = %e, "Failed to load training file");
                    continue;
                }
            };

            let features = match self.extract_features(&audio_samples) {
                Ok(f) => f,
                Err(e) => {
                    debug!(filename = %filename, error = %e, "Failed to extract features");
                    continue;
                }
            };

            // Convert features to vector
            let feature_vector = vec![
                features.rms_energy,
                features.peak_amplitude,
                features.dynamic_range,
                features.spectral_centroid,
                features.spectral_rolloff,
                features.spectral_flux,
                features.high_freq_energy,
                features.zero_crossing_rate,
                features.silence_ratio,
                features.snr_estimate,
                features.harmonic_ratio,
            ];

            features_matrix.push(feature_vector);
            labels.push(*expected_quality as i32);
        }

        if features_matrix.is_empty() {
            return Err(crate::types::ScannerError::Custom(
                "No training data loaded".to_string(),
            ));
        }

        debug!(
            samples = features_matrix.len(),
            "Extracted features for training"
        );

        // Convert to DenseMatrix format required by smartcore
        let x = DenseMatrix::from_2d_vec(&features_matrix).map_err(|e| {
            crate::types::ScannerError::Custom(format!("Failed to create matrix: {}", e))
        })?;
        let y = labels;

        // Train Random Forest model
        let model = RandomForestClassifier::fit(&x, &y, Default::default()).map_err(|e| {
            crate::types::ScannerError::Custom(format!("ML training failed: {}", e))
        })?;

        self.model = Some(model);
        debug!("Random Forest model trained successfully");

        Ok(())
    }

    /// Predict audio quality using trained model
    pub fn predict(&self, samples: &[f32]) -> crate::types::Result<QualityResult> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| crate::types::ScannerError::Custom("Model not trained".to_string()))?;

        let features = self.extract_features(samples)?;

        // Convert features to prediction format
        let feature_vector = vec![
            features.rms_energy,
            features.peak_amplitude,
            features.dynamic_range,
            features.spectral_centroid,
            features.spectral_rolloff,
            features.spectral_flux,
            features.high_freq_energy,
            features.zero_crossing_rate,
            features.silence_ratio,
            features.snr_estimate,
            features.harmonic_ratio,
        ];

        let x = DenseMatrix::from_2d_vec(&vec![feature_vector]).map_err(|e| {
            crate::types::ScannerError::Custom(format!("Failed to create prediction matrix: {}", e))
        })?;

        // Get prediction
        let prediction = model
            .predict(&x)
            .map_err(|e| crate::types::ScannerError::Custom(format!("Prediction failed: {}", e)))?;

        let predicted_label = prediction[0];

        // Convert back to AudioQuality enum
        let quality = match predicted_label {
            0 => AudioQuality::Static,
            1 => AudioQuality::NoAudio,
            2 => AudioQuality::Poor,
            3 => AudioQuality::Moderate,
            4 => AudioQuality::Good,
            _ => AudioQuality::Unknown,
        };

        // For confidence, we'll use a simple heuristic since Random Forest doesn't provide
        // direct probability scores in smartcore's basic interface
        let confidence = 0.8; // Placeholder - could be improved with more sophisticated confidence estimation

        Ok(QualityResult {
            quality,
            confidence,
            features: features.clone(),
            model_scores: vec![predicted_label as f32], // Single prediction score
        })
    }

    // Feature computation methods (copied from ML module for consistency)
    fn compute_rms_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    fn compute_dynamic_range(&self, samples: &[f32]) -> f32 {
        let max_val = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let min_val = samples
            .iter()
            .map(|x| x.abs())
            .fold(f32::INFINITY, f32::min);
        max_val - min_val
    }

    fn compute_spectral_centroid(&self, samples: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        let fft_size = 1024.min(samples.len());
        let window = &samples[..fft_size];

        // Simple frequency-weighted average approximation
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in window.iter().enumerate() {
            let frequency = i as f32 * self.sample_rate / fft_size as f32;
            let magnitude = sample.abs();
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn compute_spectral_rolloff(&self, samples: &[f32]) -> f32 {
        // Simplified spectral rolloff (frequency where 85% of energy is below)
        self.compute_spectral_centroid(samples) * 1.5
    }

    fn compute_spectral_flux(&self, samples: &[f32]) -> f32 {
        // Measure of spectral change over time
        if samples.len() < 1024 {
            return 0.0;
        }

        let chunk_size = 512;
        let mut flux_sum = 0.0;
        let mut chunk_count = 0;

        for i in (0..samples.len() - chunk_size - chunk_size / 2).step_by(chunk_size / 2) {
            let chunk1_energy: f32 = samples[i..i + chunk_size].iter().map(|x| x * x).sum();
            let chunk2_energy: f32 = samples[i + chunk_size / 2..i + chunk_size + chunk_size / 2]
                .iter()
                .map(|x| x * x)
                .sum();

            flux_sum += (chunk2_energy - chunk1_energy).abs();
            chunk_count += 1;
        }

        if chunk_count > 0 {
            flux_sum / chunk_count as f32
        } else {
            0.0
        }
    }

    fn compute_high_freq_energy(&self, samples: &[f32]) -> f32 {
        // Energy in high frequency components (approximated)
        let high_freq_threshold = 0.3; // Relative threshold
        let start_idx = (samples.len() as f32 * high_freq_threshold) as usize;

        if start_idx >= samples.len() {
            return 0.0;
        }

        let high_freq_samples = &samples[start_idx..];
        high_freq_samples.iter().map(|x| x * x).sum::<f32>() / high_freq_samples.len() as f32
    }

    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (samples.len() - 1) as f32
    }

    fn compute_silence_ratio(&self, samples: &[f32]) -> f32 {
        let threshold = 0.01;
        let silent_samples = samples.iter().filter(|&&x| x.abs() < threshold).count();
        silent_samples as f32 / samples.len() as f32
    }

    fn estimate_snr(&self, samples: &[f32]) -> f32 {
        // Simplified SNR estimation
        let signal_energy = self.compute_rms_energy(samples);
        let noise_floor = 0.01; // Estimated noise floor

        if noise_floor > 0.0 {
            20.0 * (signal_energy / noise_floor).log10()
        } else {
            0.0
        }
    }

    fn compute_harmonic_ratio(&self, samples: &[f32]) -> f32 {
        // Simplified harmonic content estimation
        let window_size = 1024.min(samples.len());
        let window = &samples[..window_size];

        // Measure periodicity by autocorrelation-like approach
        let mut max_correlation: f32 = 0.0;
        for lag in 20..window_size / 4 {
            let mut correlation = 0.0;
            let count = window_size - lag;

            for i in 0..count {
                correlation += window[i] * window[i + lag];
            }
            correlation /= count as f32;
            max_correlation = max_correlation.max(correlation.abs());
        }

        max_correlation / (self.compute_rms_energy(window) + 0.001)
    }
}
