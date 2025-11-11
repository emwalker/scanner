use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ecs::resources::{Clock, FileSystem, LocationResource, StdFileSystem};

/// Default location coordinates for San Francisco (used as fallback)
pub const DEFAULT_LOCATION: Location = Location {
    lat: 37.7749,
    lon: -122.4194,
};

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
    /// Enhanced location detection using IP-based geolocation with locality name
    pub fn current_location_with_resource(
        location_resource: &LocationResource,
        fallback_location: Option<Location>,
    ) -> Result<(Location, String), LocationError> {
        // Try IP-based detection first
        if let Ok(mut resource) = location_resource.try_lock() {
            if let Ok(detected_location) = resource.detect_current_location() {
                let location = Location {
                    lat: detected_location.lat,
                    lon: detected_location.lon,
                };
                let locality = detected_location.locality_name();
                tracing::debug!(
                    lat = location.lat,
                    lon = location.lon,
                    locality = locality,
                    "Successfully detected current location via IP geolocation"
                );
                return Ok((location, locality));
            } else {
                tracing::debug!("IP-based location detection failed, falling back to settings");
            }
        } else {
            tracing::debug!("LocationResource is locked, falling back to settings/default");
        }

        // Fallback to settings-based location
        if let Ok(settings) = Self::load_user_settings()
            && let Some(cached_loc) = settings.last_known_location
        {
            // Check if location is recent (within 30 days)
            let now = Utc::now();
            let age = now.signed_duration_since(cached_loc.timestamp);

            if age.num_days() < 30 {
                let location = Location {
                    lat: cached_loc.lat,
                    lon: cached_loc.lon,
                };
                return Ok((location, "Unknown".to_string()));
            }
        }

        // Final fallback to provided location or error
        match fallback_location {
            Some(location) => Ok((location, "Unknown".to_string())),
            None => Err(LocationError::NoLocationAvailable),
        }
    }

    /// Legacy synchronous method for backward compatibility
    pub fn current_location(
        fallback_location: Option<Location>,
    ) -> Result<Location, LocationError> {
        // For backward compatibility, just use fallback location
        fallback_location.ok_or(LocationError::NoLocationAvailable)
    }

    /// Clock-aware location detection for testing
    pub fn current_location_with_resource_and_clock<F: FileSystem, C: Clock>(
        location_resource: &LocationResource,
        fallback_location: Option<Location>,
        filesystem: &F,
        clock: &C,
    ) -> Result<(Location, String), LocationError> {
        // Try IP-based detection first
        if let Ok(mut resource) = location_resource.try_lock() {
            if let Ok(detected_location) = resource.detect_current_location() {
                let location = Location {
                    lat: detected_location.lat,
                    lon: detected_location.lon,
                };
                let locality = detected_location.locality_name();
                tracing::debug!(
                    lat = location.lat,
                    lon = location.lon,
                    locality = locality,
                    "Successfully detected current location via IP geolocation"
                );
                return Ok((location, locality));
            } else {
                tracing::debug!("IP-based location detection failed, falling back to settings");
            }
        } else {
            tracing::debug!("LocationResource is locked, falling back to settings/default");
        }

        // Fallback to settings-based location with controlled clock
        if let Ok(settings) = Self::load_user_settings_with_filesystem(filesystem)
            && let Some(cached_loc) = settings.last_known_location
        {
            // Check if location is recent (within 30 days)
            let now = clock.now_utc();
            let age = now.signed_duration_since(cached_loc.timestamp);

            if age.num_days() < 30 {
                let location = Location {
                    lat: cached_loc.lat,
                    lon: cached_loc.lon,
                };
                return Ok((location, "Unknown".to_string()));
            }
        }

        // Final fallback to provided location or error
        match fallback_location {
            Some(location) => Ok((location, "Unknown".to_string())),
            None => Err(LocationError::NoLocationAvailable),
        }
    }

    pub fn settings_path() -> Result<PathBuf, LocationError> {
        Self::settings_path_with_filesystem(&StdFileSystem)
    }

    pub fn settings_path_with_filesystem<F: FileSystem>(
        filesystem: &F,
    ) -> Result<PathBuf, LocationError> {
        let home = filesystem
            .home_dir()
            .ok_or(LocationError::HomeDirectoryNotFound)?;
        Ok(home.join(".scanner").join("settings.json"))
    }

    pub fn load_user_settings() -> Result<UserSettings, LocationError> {
        Self::load_user_settings_with_filesystem(&StdFileSystem)
    }

    pub fn load_user_settings_with_filesystem<F: FileSystem>(
        filesystem: &F,
    ) -> Result<UserSettings, LocationError> {
        let settings_path = Self::settings_path_with_filesystem(filesystem)?;

        if !filesystem.exists(&settings_path) {
            return Ok(UserSettings::default());
        }

        let content = filesystem
            .read_to_string(&settings_path)
            .map_err(|_| LocationError::SettingsFileReadError)?;

        let settings: UserSettings =
            serde_json::from_str(&content).map_err(|_| LocationError::SettingsFileParseError)?;

        Ok(settings)
    }

    pub fn save_user_settings(settings: &UserSettings) -> Result<(), LocationError> {
        Self::save_user_settings_with_filesystem(settings, &StdFileSystem)
    }

    pub fn save_user_settings_with_filesystem<F: FileSystem>(
        settings: &UserSettings,
        filesystem: &F,
    ) -> Result<(), LocationError> {
        let settings_path = Self::settings_path_with_filesystem(filesystem)?;

        if let Some(parent) = settings_path.parent() {
            filesystem
                .create_dir_all(parent)
                .map_err(|_| LocationError::SettingsDirectoryCreateError)?;
        }

        let content = serde_json::to_string_pretty(settings)
            .map_err(|_| LocationError::SettingsFileSerializeError)?;

        filesystem
            .write(&settings_path, &content)
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
        let result = LocationDetector::current_location(Some(fallback));
        assert!(result.is_ok());

        let location = result.unwrap();
        assert_eq!(location.lat, 37.7749);
        assert_eq!(location.lon, -122.4194);
    }

    #[test]
    fn test_location_detection_no_fallback() {
        let result = LocationDetector::current_location(None);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            LocationError::NoLocationAvailable
        ));
    }

    #[test]
    fn test_location_detection_with_resource() {
        use crate::ecs::resources::new_location_resource;

        // Create a location resource (offline database may not be available in tests)
        let location_resource = new_location_resource();

        let fallback = Location {
            lat: 40.4173, // Loveland, CO
            lon: -105.0178,
        };

        let result =
            LocationDetector::current_location_with_resource(&location_resource, Some(fallback));

        assert!(result.is_ok());
        let (location, locality) = result.unwrap();

        // Should return some location and locality name
        assert!(location.lat.is_finite());
        assert!(location.lon.is_finite());
        assert!(!locality.is_empty());

        // In test environment, will likely fall back to fallback location
        // but should not panic or fail
    }

    #[test]
    fn test_location_detection_with_resource_no_fallback() {
        use crate::ecs::resources::new_location_resource;

        let location_resource = new_location_resource();

        let result = LocationDetector::current_location_with_resource(&location_resource, None);

        // May succeed with IP detection or fail if no connection/database
        // We just check that it doesn't panic
        match result {
            Ok((location, locality)) => {
                assert!(location.lat.is_finite());
                assert!(location.lon.is_finite());
                assert!(!locality.is_empty());
            }
            Err(LocationError::NoLocationAvailable) => {
                // Expected when no IP detection available and no fallback
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}
