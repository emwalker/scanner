use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use chrono::Utc;

use crate::persistence::{
    h3_grid::{H3Error, H3Grid},
    location::{Location, LocationError},
    types::{CellMetadata, PersistedSignal, SignalsFile},
};

pub struct SignalStorage {
    base_path: PathBuf,
}

impl SignalStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    pub fn save_signal(
        &self,
        signal: &PersistedSignal,
        location: Location,
    ) -> Result<(), StorageError> {
        let cell_id = H3Grid::location_to_cell_id(location, 6).map_err(StorageError::H3Error)?;

        let mut signals_file = self.load_signals_file(&cell_id)?;

        let frequency_key = signal.frequency_hz as u64;
        signals_file.signals.insert(frequency_key, signal.clone());
        signals_file.metadata.last_updated = Utc::now();

        self.write_signals_file(&cell_id, &signals_file)?;

        Ok(())
    }

    pub fn load_signals_for_location(
        &self,
        location: Location,
    ) -> Result<Vec<PersistedSignal>, StorageError> {
        // For now, just load from current location's cell (will add neighbor loading later)
        tracing::debug!(
            lat = location.lat,
            lon = location.lon,
            "Loading signals for location"
        );

        let cell_id = H3Grid::location_to_cell_id(location, 6).map_err(StorageError::H3Error)?;

        tracing::debug!(cell_id = cell_id, "Calculated H3 cell ID");

        let signals_file = self.load_signals_file(&cell_id)?;

        tracing::debug!(
            signal_count = signals_file.signals.len(),
            "Converting signals to vector"
        );

        let result: Vec<_> = signals_file.signals.into_values().collect();

        tracing::debug!(result_count = result.len(), "Final result count");

        Ok(result)
    }

    fn load_signals_file(&self, cell_id: &str) -> Result<SignalsFile, StorageError> {
        let file_path = self.signals_file_path(cell_id);

        tracing::debug!(
            cell_id = cell_id,
            file_path = %file_path.display(),
            exists = file_path.exists(),
            "Loading signals file"
        );

        if !file_path.exists() {
            tracing::debug!("File does not exist, creating empty signals file");
            return self.create_empty_signals_file(cell_id);
        }

        let content =
            std::fs::read_to_string(&file_path).map_err(|_| StorageError::FileReadError)?;

        tracing::debug!(content_size = content.len(), "Read file content");

        let signals_file: SignalsFile = serde_yaml::from_str(&content).map_err(|e| {
            tracing::error!(
                error = %e,
                "YAML parsing error"
            );
            StorageError::YamlParseError
        })?;

        tracing::debug!(
            signal_count = signals_file.signals.len(),
            "Successfully parsed signals file"
        );

        Ok(signals_file)
    }

    fn write_signals_file(
        &self,
        cell_id: &str,
        signals_file: &SignalsFile,
    ) -> Result<(), StorageError> {
        let file_path = self.signals_file_path(cell_id);

        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| StorageError::DirectoryCreateError)?;
        }

        let yaml_content =
            serde_yaml::to_string(signals_file).map_err(|_| StorageError::YamlSerializeError)?;

        // Atomic write using temp file
        let temp_path = file_path.with_extension("yaml.tmp");
        std::fs::write(&temp_path, yaml_content).map_err(|_| StorageError::FileWriteError)?;

        std::fs::rename(&temp_path, &file_path).map_err(|_| StorageError::FileWriteError)?;

        Ok(())
    }

    fn create_empty_signals_file(&self, cell_id: &str) -> Result<SignalsFile, StorageError> {
        let center = H3Grid::cell_center(cell_id).map_err(StorageError::H3Error)?;

        Ok(SignalsFile {
            version: "v1.0".to_string(),
            signals: BTreeMap::new(),
            metadata: CellMetadata {
                h3_cell_id: cell_id.to_string(),
                center_lat: center.lat,
                center_lon: center.lon,
                last_updated: Utc::now(),
            },
        })
    }

    fn signals_file_path(&self, cell_id: &str) -> PathBuf {
        self.base_path.join(cell_id).join("signals.yaml")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("H3 grid operation failed: {0}")]
    H3Error(#[from] H3Error),
    #[error("Location detection failed: {0}")]
    LocationError(#[from] LocationError),
    #[error("Failed to read file")]
    FileReadError,
    #[error("Failed to write file")]
    FileWriteError,
    #[error("Failed to create directory")]
    DirectoryCreateError,
    #[error("Failed to parse YAML")]
    YamlParseError,
    #[error("Failed to serialize YAML")]
    YamlSerializeError,
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use tempfile::tempdir;

    use super::*;
    use crate::core::signals::ModulationType;

    #[test]
    fn test_save_and_load_signal() {
        let temp_dir = tempdir().unwrap();
        let storage = SignalStorage::new(temp_dir.path());
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let signal = PersistedSignal {
            frequency_hz: 88900000.0,
            signal_strength: 0.85,
            first_detected: Utc.with_ymd_and_hms(2024, 11, 9, 15, 30, 0).unwrap(),
            last_detected: Utc.with_ymd_and_hms(2024, 11, 9, 16, 45, 0).unwrap(),
            detection_count: 12,
            modulation: ModulationType::WFM,
            notes: Some("Test station".to_string()),
        };

        // Save signal
        storage.save_signal(&signal, location).unwrap();

        // Load signals
        let loaded_signals = storage.load_signals_for_location(location).unwrap();

        assert_eq!(loaded_signals.len(), 1);
        assert_eq!(loaded_signals[0].frequency_hz, 88900000.0);
        assert_eq!(loaded_signals[0].notes, Some("Test station".to_string()));
    }

    #[test]
    fn test_multiple_signals_frequency_sorted() {
        let temp_dir = tempdir().unwrap();
        let storage = SignalStorage::new(temp_dir.path());
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let signal_high = PersistedSignal {
            frequency_hz: 107900000.0,
            signal_strength: 0.9,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 5,
            modulation: ModulationType::WFM,
            notes: None,
        };

        let signal_low = PersistedSignal {
            frequency_hz: 88500000.0,
            signal_strength: 0.7,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 3,
            modulation: ModulationType::WFM,
            notes: None,
        };

        // Save in reverse order
        storage.save_signal(&signal_high, location).unwrap();
        storage.save_signal(&signal_low, location).unwrap();

        let loaded_signals = storage.load_signals_for_location(location).unwrap();

        assert_eq!(loaded_signals.len(), 2);
        // Should be frequency sorted
        assert_eq!(loaded_signals[0].frequency_hz, 88500000.0);
        assert_eq!(loaded_signals[1].frequency_hz, 107900000.0);
    }

    #[test]
    fn test_atomic_write_protection() {
        let temp_dir = tempdir().unwrap();
        let storage = SignalStorage::new(temp_dir.path());
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let signal = PersistedSignal {
            frequency_hz: 88900000.0,
            signal_strength: 0.85,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 1,
            modulation: ModulationType::WFM,
            notes: None,
        };

        storage.save_signal(&signal, location).unwrap();

        let cell_id = H3Grid::location_to_cell_id(location, 6).unwrap();
        let file_path = storage.signals_file_path(&cell_id);
        let temp_path = file_path.with_extension("yaml.tmp");

        // Temp file should not exist after successful write
        assert!(!temp_path.exists());
        assert!(file_path.exists());
    }
}
