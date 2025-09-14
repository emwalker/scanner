//! Machine Learning Regression Module for Audio Quality Analysis
//!
//! This module provides ML-based audio quality scoring using SmartCore algorithms.
//! It extracts features from audio samples and trains regression models
//! to predict quality scores based on human calibration data.

use crate::audio_quality::AudioQuality;
use smartcore::ensemble::random_forest_regressor::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::debug;

#[cfg(test)]
use std::collections::HashMap;

/// Training sample with features and human rating
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub frequency_hz: f64,
    pub features: Vec<f64>,
    pub human_rating: QualityScore,
}

/// Numerical quality score for ML regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityScore {
    Static = 0,
    Poor = 1,
    Moderate = 2,
    Good = 3,
}

impl From<AudioQuality> for QualityScore {
    fn from(quality: AudioQuality) -> Self {
        match quality {
            AudioQuality::Static => QualityScore::Static,
            AudioQuality::Poor => QualityScore::Poor,
            AudioQuality::Moderate => QualityScore::Moderate,
            AudioQuality::Good => QualityScore::Good,
            AudioQuality::Unknown => QualityScore::Static, // Default fallback
        }
    }
}

impl From<QualityScore> for AudioQuality {
    fn from(score: QualityScore) -> Self {
        match score {
            QualityScore::Static => AudioQuality::Static,
            QualityScore::Poor => AudioQuality::Poor,
            QualityScore::Moderate => AudioQuality::Moderate,
            QualityScore::Good => AudioQuality::Good,
        }
    }
}

/// ML-based audio quality analyzer
pub struct MLAudioQualityAnalyzer {
    model: Option<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    #[allow(dead_code)]
    sample_rate: f32,
    #[allow(dead_code)]
    fft_size: usize,
}

impl MLAudioQualityAnalyzer {
    pub fn new(sample_rate: f32, fft_size: usize) -> Self {
        Self {
            model: None,
            sample_rate,
            fft_size,
        }
    }

    /// Extract feature vector from pre-computed quality results
    fn extract_features_from_result(result: &crate::audio_quality::QualityResult) -> Vec<f64> {
        vec![
            result.normalized_signal_strength as f64,
            result.si_sdr_db as f64,
            result.integrated_loudness_lufs as f64,
            result.pcen_spectral_flatness as f64,
            result.normalized_snr_db as f64,
            result.temporal_stability as f64,
            result.quality_score as f64,
        ]
    }

    /// Simple WAV file loader for ML training (test only)
    #[cfg(test)]
    fn load_wav_samples(wav_path: &str) -> crate::types::Result<Vec<f32>> {
        use std::fs::File;
        use std::io::{BufReader, Read, Seek, SeekFrom};

        let mut file = BufReader::new(File::open(wav_path)?);

        // Skip WAV header (44 bytes for standard WAV)
        file.seek(SeekFrom::Start(44))?;

        let mut samples = Vec::new();
        let mut buffer = [0u8; 4]; // 32-bit IEEE float samples

        while file.read_exact(&mut buffer).is_ok() {
            let sample = f32::from_le_bytes(buffer);
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Extract frequency in Hz from WAV filename (test only)
    #[cfg(test)]
    fn extract_frequency_from_filename(filename: &str) -> Option<f64> {
        // Parse filename like "000.088.900.000Hz-wfm-001.wav"
        if let Some(hz_pos) = filename.find("Hz") {
            let freq_part = &filename[..hz_pos];
            // Remove leading zeros and dots, then parse as Hz
            let freq_str = freq_part.replace(".", "");
            if let Ok(freq_hz) = freq_str.parse::<u64>() {
                return Some(freq_hz as f64);
            }
        }
        None
    }

    /// Load training data from calibration files with human ratings (test only)
    #[cfg(test)]
    pub fn load_training_data() -> crate::types::Result<Vec<TrainingSample>> {
        // Human calibration data - actual filenames from our calibration session
        let human_ratings: HashMap<&str, QualityScore> = [
            // Static samples (from our calibration)
            ("000.088.099.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.088.299.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.088.300.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.088.499.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.089.099.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.089.301.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.089.500.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.090.300.000Hz-wfm-001.wav", QualityScore::Static),
            ("000.090.701.000Hz-wfm-001.wav", QualityScore::Static),
            // Poor samples
            ("000.088.700.000Hz-wfm-001.wav", QualityScore::Poor),
            ("000.091.100.000Hz-wfm-001.wav", QualityScore::Poor),
            ("000.092.101.000Hz-wfm-001.wav", QualityScore::Poor),
            // Moderate samples
            ("000.089.700.000Hz-wfm-001.wav", QualityScore::Moderate),
            ("000.091.500.000Hz-wfm-001.wav", QualityScore::Moderate),
            // Good samples
            ("000.088.900.000Hz-wfm-001.wav", QualityScore::Good),
        ]
        .into_iter()
        .collect();

        let mut training_samples = Vec::new();

        // Use our own analyzer for feature extraction
        let feature_analyzer = super::AudioQualityMetrics::new(48000.0, 1024);

        for (filename, &human_rating) in &human_ratings {
            let wav_path = format!("tests/data/audio/quality/{}", filename);

            match Self::load_wav_samples(&wav_path) {
                Ok(samples) => {
                    if !samples.is_empty() {
                        // Run normalized audio analysis on the WAV samples
                        let result = feature_analyzer.analyze(&samples, 48000.0); // All our samples are 48kHz
                        let features = vec![
                            result.normalized_signal_strength as f64,
                            result.si_sdr_db as f64,
                            result.integrated_loudness_lufs as f64,
                            result.pcen_spectral_flatness as f64,
                            result.normalized_snr_db as f64,
                            result.temporal_stability as f64,
                            result.quality_score as f64,
                        ];

                        // Extract frequency from filename (e.g., 000.088.900.000Hz -> 88.9 MHz)
                        let frequency_hz =
                            Self::extract_frequency_from_filename(filename).unwrap_or(88_900_000.0);

                        debug!(
                            filename = filename,
                            human_rating = format!("{:?}", human_rating),
                            features = format!("{:.3?}", features),
                            frequency_mhz = frequency_hz / 1e6,
                            samples_count = samples.len(),
                            "Loaded training sample with normalized analysis"
                        );

                        training_samples.push(TrainingSample {
                            frequency_hz,
                            features,
                            human_rating,
                        });
                    }
                }
                Err(e) => {
                    debug!(
                        filename = filename,
                        error = format!("{}", e),
                        "Failed to load WAV training sample"
                    );
                }
            }
        }

        debug!(
            total_samples = training_samples.len(),
            "Loaded training dataset"
        );

        Ok(training_samples)
    }

    /// Train Random Forest regression model with provided training data
    pub fn train(&mut self, training_data: Vec<TrainingSample>) -> crate::types::Result<()> {
        if training_data.is_empty() {
            return Err(crate::types::ScannerError::Custom(
                "No training data available".to_string(),
            ));
        }

        // Convert to SmartCore format
        let feature_count = training_data[0].features.len();
        let sample_count = training_data.len();

        let mut x_data = vec![0.0f64; sample_count * feature_count];
        let mut y_data = vec![0.0f64; sample_count];

        for (i, sample) in training_data.iter().enumerate() {
            // Features matrix (row-major)
            for (j, &feature) in sample.features.iter().enumerate() {
                x_data[i * feature_count + j] = feature;
            }
            // Target values
            y_data[i] = sample.human_rating as i32 as f64;
        }

        let x_matrix =
            DenseMatrix::new(sample_count, feature_count, x_data, false).map_err(|e| {
                crate::types::ScannerError::Custom(format!(
                    "Failed to create feature matrix: {}",
                    e
                ))
            })?;

        // Train Random Forest model
        let model = RandomForestRegressor::fit(
            &x_matrix,
            &y_data,
            Default::default(), // Use default hyperparameters
        )
        .map_err(|e| {
            crate::types::ScannerError::Custom(format!("Failed to train ML model: {}", e))
        })?;

        self.model = Some(model);

        debug!(
            samples = sample_count,
            features = feature_count,
            "ML model training completed"
        );

        Ok(())
    }

    /// Predict audio quality score using trained model from pre-computed quality result
    pub fn predict_from_result(
        &self,
        quality_result: &crate::audio_quality::QualityResult,
    ) -> crate::types::Result<f64> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| crate::types::ScannerError::Custom("Model not trained".to_string()))?;

        let features = Self::extract_features_from_result(quality_result);
        let feature_count = features.len();

        let x_matrix = DenseMatrix::new(1, feature_count, features, false).map_err(|e| {
            crate::types::ScannerError::Custom(format!("Failed to create prediction matrix: {}", e))
        })?;

        let predictions = model
            .predict(&x_matrix)
            .map_err(|e| crate::types::ScannerError::Custom(format!("Prediction failed: {}", e)))?;

        Ok(predictions[0])
    }

    /// Enhance quality result with ML-based quality assessment
    pub fn enhance_with_ml(
        &self,
        quality_result: &crate::audio_quality::QualityResult,
    ) -> crate::types::Result<AudioQuality> {
        let predicted_score = self.predict_from_result(quality_result)?;

        // Convert continuous score to discrete quality level
        // Refined thresholds for 18-sample balanced training dataset (8 Static, 6 Poor, 3 Moderate, 1 Good)
        let quality = if predicted_score < 0.5 {
            AudioQuality::Static
        } else if predicted_score < 1.6 {
            // Narrowed Poor range for better Moderate detection
            AudioQuality::Poor
        } else if predicted_score < 2.6 {
            // Expanded Moderate range
            AudioQuality::Moderate
        } else {
            AudioQuality::Good
        };

        debug!(
            predicted_score = predicted_score,
            audio_quality = format!("{:?}", quality),
            features = format!("{:.3?}", Self::extract_features_from_result(quality_result)),
            "ML audio quality prediction from pre-computed features"
        );

        Ok(quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_training_and_prediction() {
        let _ = tracing_subscriber::fmt::try_init();

        let mut analyzer = MLAudioQualityAnalyzer::new(48000.0, 1024);

        // Load training data and train the model
        let training_data =
            MLAudioQualityAnalyzer::load_training_data().expect("Should load training data");
        analyzer
            .train(training_data)
            .expect("Training should succeed");

        // Test prediction on known good sample using embedded feature data
        let good_sample_features = TrainingSample {
            frequency_hz: 88_900_000.0,
            features: vec![0.319, 3.002, -10.618, 0.002, 12.522, 0.703, 0.502],
            human_rating: QualityScore::Good,
        };

        // Create dummy QualityResult for testing
        let test_result = crate::audio_quality::QualityResult {
            si_sdr_db: good_sample_features.features[1] as f32,
            normalized_signal_strength: good_sample_features.features[0] as f32,
            integrated_loudness_lufs: good_sample_features.features[2] as f32,
            pcen_spectral_flatness: good_sample_features.features[3] as f32,
            normalized_snr_db: good_sample_features.features[4] as f32,
            temporal_stability: good_sample_features.features[5] as f32,
            quality_score: good_sample_features.features[6] as f32,
            audio_quality: AudioQuality::Good, // Will be overridden by ML
        };

        let ml_quality = analyzer
            .enhance_with_ml(&test_result)
            .expect("ML prediction should succeed");

        debug!(
            predicted_quality = format!("{:?}", ml_quality),
            expected_quality = format!("{:?}", good_sample_features.human_rating),
            "ML prediction for known good sample"
        );

        // Should predict some form of audio content (not Static)
        assert!(
            ml_quality.is_audio(),
            "ML should detect audio content (not static)"
        );
        debug!(
            test_result = "PASS - ML integration working correctly",
            "ML prediction system successfully integrated"
        );
    }
}
