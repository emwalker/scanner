use std::path::PathBuf;

use chrono::{DateTime, Utc};
use dirs::home_dir;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Location {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserSettings {
    pub version: String,
    pub last_known_location: Option<CachedLocation>,
    pub preferences: UserPreferences,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CachedLocation {
    pub lat: f64,
    pub lon: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserPreferences {
    pub auto_save_interval_seconds: u64,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            version: "v1.0".to_string(),
            last_known_location: None,
            preferences: UserPreferences {
                auto_save_interval_seconds: 30,
            },
        }
    }
}

pub struct LocationDetector;

impl LocationDetector {
    pub fn get_current_location(
        fallback_location: Option<Location>,
    ) -> Result<Location, LocationError> {
        // For now, always use fallback location (will implement OS detection later)
        fallback_location.ok_or(LocationError::NoLocationAvailable)
    }

    pub fn settings_path() -> Result<PathBuf, LocationError> {
        let home = home_dir().ok_or(LocationError::HomeDirectoryNotFound)?;
        Ok(home.join(".scanner").join("settings.json"))
    }

    pub fn load_user_settings() -> Result<UserSettings, LocationError> {
        let settings_path = Self::settings_path()?;

        if !settings_path.exists() {
            return Ok(UserSettings::default());
        }

        let content = std::fs::read_to_string(&settings_path)
            .map_err(|_| LocationError::SettingsFileReadError)?;

        let settings: UserSettings =
            serde_json::from_str(&content).map_err(|_| LocationError::SettingsFileParseError)?;

        Ok(settings)
    }

    pub fn save_user_settings(settings: &UserSettings) -> Result<(), LocationError> {
        let settings_path = Self::settings_path()?;

        if let Some(parent) = settings_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|_| LocationError::SettingsDirectoryCreateError)?;
        }

        let content = serde_json::to_string_pretty(settings)
            .map_err(|_| LocationError::SettingsFileSerializeError)?;

        std::fs::write(&settings_path, content)
            .map_err(|_| LocationError::SettingsFileWriteError)?;

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocationError {
    #[error("No location available from any source")]
    NoLocationAvailable,
    #[error("Home directory not found")]
    HomeDirectoryNotFound,
    #[error("Failed to read settings file")]
    SettingsFileReadError,
    #[error("Failed to parse settings file")]
    SettingsFileParseError,
    #[error("Failed to create settings directory")]
    SettingsDirectoryCreateError,
    #[error("Failed to serialize settings")]
    SettingsFileSerializeError,
    #[error("Failed to write settings file")]
    SettingsFileWriteError,
}

#[cfg(test)]
mod tests {

    use chrono::Utc;

    use super::*;

    #[test]
    fn test_user_settings_serialization() {
        let settings = UserSettings {
            version: "v1.0".to_string(),
            last_known_location: Some(CachedLocation {
                lat: 37.7749,
                lon: -122.4194,
                timestamp: Utc::now(),
            }),
            preferences: UserPreferences {
                auto_save_interval_seconds: 30,
            },
        };

        let json = serde_json::to_string_pretty(&settings).unwrap();
        let deserialized: UserSettings = serde_json::from_str(&json).unwrap();

        assert_eq!(settings.version, deserialized.version);
        assert!(json.contains("v1.0"));
    }

    #[test]
    fn test_location_detection_fallback() {
        let fallback = Location {
            lat: 37.7749,
            lon: -122.4194,
        };
        let result = LocationDetector::get_current_location(Some(fallback));
        assert!(result.is_ok());

        let location = result.unwrap();
        assert_eq!(location.lat, 37.7749);
        assert_eq!(location.lon, -122.4194);
    }

    #[test]
    fn test_location_detection_no_fallback() {
        let result = LocationDetector::get_current_location(None);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            LocationError::NoLocationAvailable
        ));
    }
}
