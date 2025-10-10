use super::model::{Classifier, SerializableModel};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::debug;

impl Classifier {
    /// Train the ML model on training data
    pub fn train(&mut self) -> crate::core::types::Result<()> {
        debug!("Training Random Forest classifier on handcrafted features");

        let training_data = crate::audio::quality::training_dataset();
        let mut features_matrix = Vec::new();
        let mut labels = Vec::new();

        // Extract features from all training samples
        for (filename, expected_quality) in training_data.iter() {
            let wav_path = std::path::PathBuf::from("tests/data/audio/quality").join(filename);

            if !wav_path.exists() {
                debug!(filename = %filename, "Training file not found, skipping");
                continue;
            }

            let audio_samples = match crate::file::wave::load_file(&wav_path) {
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

            // Convert features to vector for ML model
            features_matrix.push(vec![
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
            ]);

            labels.push(*expected_quality as i32);
        }

        if features_matrix.is_empty() {
            return Err(crate::core::types::ScannerError::ModelError(
                "No training data available".to_string(),
            ));
        }

        // Convert to DenseMatrix format
        let x_train = DenseMatrix::from_2d_vec(&features_matrix)?;
        let y_train = labels;

        // Store serializable data for later saving
        self.serializable_data = Some(SerializableModel {
            features_matrix: features_matrix.clone(),
            labels: y_train.clone(),
        });

        // Train Random Forest model
        let model = RandomForestClassifier::fit(&x_train, &y_train, Default::default())?;

        self.model = Some(model);

        debug!(
            training_samples = features_matrix.len(),
            features_per_sample = features_matrix[0].len(),
            "Random Forest model training completed"
        );

        Ok(())
    }
}
