use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::debug;

/// Serializable model data for Random Forest classifier
#[derive(Serialize, Deserialize, Clone)]
pub(super) struct SerializableModel {
    pub(super) features_matrix: Vec<Vec<f32>>,
    pub(super) labels: Vec<i32>,
}

/// Random Forest audio quality classifier
pub struct Classifier {
    pub(super) sample_rate: f32,
    pub(super) model: Option<RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>>>,
    pub(super) serializable_data: Option<SerializableModel>,
}

impl Classifier {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            model: None,
            serializable_data: None,
        }
    }

    /// Load a pre-trained model from file
    pub fn load_pretrained(model_path: &str) -> crate::core::types::Result<Self> {
        use std::fs::File;
        use std::io::BufReader;

        debug!(model_path = %model_path, "Loading pre-trained Random Forest model");

        let file = File::open(model_path)?;

        let reader = BufReader::new(file);

        #[derive(Deserialize)]
        struct SavedClassifier {
            model_version: String,
            created_at: String,
            training_samples: usize,
            feature_count: usize,
            sample_rate: f32,
            serializable_data: Option<SerializableModel>,
        }

        let saved: SavedClassifier = bincode::deserialize_from(reader)?;

        if saved.serializable_data.is_none() {
            return Err(crate::core::types::ScannerError::ModelError(
                "Loaded model file contains no training data".to_string(),
            ));
        }

        // Validate model compatibility
        if saved.feature_count != 11 {
            return Err(crate::core::types::ScannerError::ModelError(format!(
                "Model feature count mismatch: expected 11, got {}",
                saved.feature_count
            )));
        }

        debug!(
            model_version = %saved.model_version,
            created_at = %saved.created_at,
            training_samples = saved.training_samples,
            feature_count = saved.feature_count,
            sample_rate = saved.sample_rate,
            "Loaded model metadata"
        );

        let mut classifier = Self::new(saved.sample_rate);
        classifier.serializable_data = saved.serializable_data;

        // Rebuild the model from saved training data
        classifier.rebuild_model()?;

        debug!(
            sample_rate = classifier.sample_rate,
            "Successfully loaded and rebuilt Random Forest model"
        );

        Ok(classifier)
    }

    /// Save the trained model to file
    pub fn save_model(
        &self,
        model_path: &str,
        model_version: &str,
    ) -> crate::core::types::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        if self.serializable_data.is_none() {
            return Err(crate::core::types::ScannerError::ModelError(
                "Cannot save model: no training data available".to_string(),
            ));
        }

        debug!(model_path = %model_path, "Saving trained Random Forest model");

        let file = File::create(model_path)?;

        #[derive(Serialize)]
        struct SavedClassifier {
            model_version: String,
            created_at: String,
            training_samples: usize,
            feature_count: usize,
            sample_rate: f32,
            serializable_data: Option<SerializableModel>,
        }

        let serializable_data = self.serializable_data.as_ref().ok_or_else(|| {
            crate::core::types::ScannerError::ModelError(
                "Internal error: serializable_data missing after validation".to_string(),
            )
        })?;
        let current_time = chrono::Utc::now().to_rfc3339();

        let to_save = SavedClassifier {
            model_version: model_version.to_string(),
            created_at: current_time,
            training_samples: serializable_data.features_matrix.len(),
            feature_count: serializable_data
                .features_matrix
                .first()
                .map(|v| v.len())
                .unwrap_or(0),
            sample_rate: self.sample_rate,
            serializable_data: self.serializable_data.clone(),
        };

        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &to_save)?;

        debug!(
            model_path = %model_path,
            sample_rate = self.sample_rate,
            "Successfully saved Random Forest model"
        );

        Ok(())
    }

    /// Rebuild model from serializable data
    pub(super) fn rebuild_model(&mut self) -> crate::core::types::Result<()> {
        let serializable_data = self.serializable_data.as_ref().ok_or_else(|| {
            crate::core::types::ScannerError::ModelError(
                "No training data available to rebuild model".to_string(),
            )
        })?;

        debug!("Rebuilding Random Forest model from serialized training data");

        // Convert to DenseMatrix format
        let x_train = DenseMatrix::from_2d_vec(&serializable_data.features_matrix)?;
        let y_train = serializable_data.labels.clone();

        // Train Random Forest model
        let model = RandomForestClassifier::fit(&x_train, &y_train, Default::default())?;

        self.model = Some(model);

        debug!(
            training_samples = serializable_data.features_matrix.len(),
            features_per_sample = serializable_data
                .features_matrix
                .first()
                .map(|v| v.len())
                .unwrap_or(0),
            "Random Forest model rebuilt successfully"
        );

        Ok(())
    }

    /// Extract comprehensive audio features
    pub(super) fn extract_features(
        &self,
        samples: &[f32],
    ) -> crate::core::types::Result<crate::audio::quality::AudioFeatures> {
        if samples.is_empty() {
            return Err(crate::core::types::ScannerError::SignalProcessingFailed(
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

        Ok(crate::audio::quality::AudioFeatures {
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

    // Feature computation methods
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
        let max_val = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let min_val = samples
            .iter()
            .map(|&x| x.abs())
            .fold(f32::INFINITY, f32::min);
        if min_val > 1e-10 {
            20.0 * (max_val / min_val).log10()
        } else {
            60.0 // Large dynamic range if min is essentially zero
        }
    }

    fn compute_spectral_centroid(&self, samples: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        let window_size = 1024.min(samples.len());
        if window_size == 0 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in samples[..window_size].iter().enumerate() {
            let magnitude = sample.abs();
            let frequency = (i as f32) * self.sample_rate / (window_size as f32 * 2.0);
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn compute_spectral_rolloff(&self, _samples: &[f32]) -> f32 {
        // Simplified spectral rolloff (frequency below which 85% of energy lies)
        self.sample_rate * 0.85 / 2.0 // Rough approximation
    }

    fn compute_spectral_flux(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut flux = 0.0;
        for i in 1..samples.len() {
            flux += (samples[i] - samples[i - 1]).abs();
        }

        flux / samples.len() as f32
    }

    fn compute_high_freq_energy(&self, samples: &[f32]) -> f32 {
        // Simplified high-frequency energy estimation
        let mut high_freq_energy = 0.0;
        let window_size = 64.min(samples.len());

        for window in samples.chunks(window_size) {
            let mut diff_energy = 0.0;
            for i in 1..window.len() {
                diff_energy += (window[i] - window[i - 1]).powi(2);
            }
            high_freq_energy += diff_energy;
        }

        high_freq_energy / samples.len() as f32
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

        let signal_power = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let noise_power =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        if noise_power > 1e-10 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            30.0
        }
    }

    fn compute_harmonic_ratio(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 0.0;
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

        best_correlation.clamp(0.0, 1.0)
    }
}
