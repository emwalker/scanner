use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::debug;

use super::model::Classifier;

impl crate::audio::quality::Classifier for Classifier {
    fn analyze(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> crate::core::types::Result<crate::audio::quality::QualityResult> {
        debug!(
            samples_len = samples.len(),
            sample_rate = sample_rate,
            "Starting Random Forest audio quality analysis"
        );

        // Extract features
        let features = self.extract_features(samples)?;

        // Check if model is trained
        let model = self
            .model
            .as_ref()
            .ok_or(crate::core::types::ScannerError::ModelNotTrained)?;

        // Prepare features for prediction
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

        let x_predict = DenseMatrix::from_2d_vec(&vec![feature_vector])?;

        // Make prediction
        let prediction = model.predict(&x_predict)?;

        let predicted_class = prediction[0];

        // Convert back to AudioQuality enum
        let quality = match predicted_class {
            0 => crate::audio::quality::AudioQuality::Static,
            1 => crate::audio::quality::AudioQuality::NoAudio,
            2 => crate::audio::quality::AudioQuality::Poor,
            3 => crate::audio::quality::AudioQuality::Moderate,
            4 => crate::audio::quality::AudioQuality::Good,
            _ => crate::audio::quality::AudioQuality::Unknown,
        };

        // Calculate confidence based on feature consistency
        let confidence = self.calculate_confidence(&features, quality);

        debug!(
            predicted_class = predicted_class,
            quality = format!("{:?}", quality),
            confidence = confidence,
            "Random Forest analysis complete"
        );

        Ok(crate::audio::quality::QualityResult {
            quality,
            confidence,
            signal_strength: features.rms_energy,
            features: Some(features),
        })
    }

    fn name(&self) -> &'static str {
        "random_forest"
    }
}

impl Classifier {
    /// Calculate confidence score based on feature consistency with predicted quality
    pub(super) fn calculate_confidence(
        &self,
        features: &crate::audio::quality::AudioFeatures,
        quality: crate::audio::quality::AudioQuality,
    ) -> f32 {
        let mut confidence_factors = Vec::new();

        // Signal strength consistency
        match quality {
            crate::audio::quality::AudioQuality::Static => {
                confidence_factors.push(if features.rms_energy < 0.1 { 0.9 } else { 0.3 });
            }
            crate::audio::quality::AudioQuality::Good => {
                confidence_factors.push(if features.rms_energy > 0.3 { 0.8 } else { 0.4 });
                confidence_factors.push(if features.snr_estimate > 20.0 {
                    0.8
                } else {
                    0.5
                });
            }
            _ => {
                confidence_factors.push(0.6); // Moderate confidence for other qualities
            }
        }

        // Average confidence factors
        if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32
        }
    }
}
