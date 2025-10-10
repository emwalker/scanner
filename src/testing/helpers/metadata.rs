use crate::core::types::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;

/// Metadata for audio fixture files
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioFileMetadata {
    pub sample_rate: f32,
    pub squelch_learning_duration: f32,
    pub total_samples: usize,
    pub format: String,
    pub expected_squelch_decision: String,
    pub description: String,
    pub frequency_hz: f64,
    pub center_freq: f64,
    pub driver: String,
}

impl AudioFileMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f32,
        squelch_learning_duration: f32,
        total_samples: usize,
        expected_squelch_decision: String,
        description: String,
        frequency_hz: f64,
        center_freq: f64,
        driver: String,
    ) -> Self {
        Self {
            sample_rate,
            squelch_learning_duration,
            total_samples,
            format: "f32_le".to_string(),
            expected_squelch_decision,
            description,
            frequency_hz,
            center_freq,
            driver,
        }
    }

    /// Load metadata from a JSON file
    #[cfg(test)]
    pub fn from_file(metadata_path: &str) -> Result<Self> {
        let file = File::open(metadata_path)?;
        let metadata: AudioFileMetadata = serde_json::from_reader(file)?;
        Ok(metadata)
    }

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

/// Extension trait for IqFileMetadata to add from_file method
pub trait IqFileMetadataExt {
    fn from_file(metadata_path: &str) -> Result<crate::file::IqFileMetadata>;
}

impl IqFileMetadataExt for crate::file::IqFileMetadata {
    fn from_file(metadata_path: &str) -> Result<crate::file::IqFileMetadata> {
        let file = File::open(metadata_path)?;
        let metadata: crate::file::IqFileMetadata = serde_json::from_reader(file)?;
        Ok(metadata)
    }
}
