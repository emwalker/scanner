//! Tests for signal path construction and loading
//! This test suite verifies that signals are loaded correctly on TUI startup

#[cfg(test)]
mod tests {
    use std::fs;

    use chrono::Utc;
    use tempfile::TempDir;

    use crate::{
        core::signals::ModulationType,
        persistence::{location::Location, storage::SignalStorage, types::PersistedSignal},
    };

    #[test]
    fn test_signal_storage_uses_correct_file_path() {
        // TDD RED: Test that SignalStorage looks for files in the correct location
        // This should pass now that we've fixed the double path issue

        let temp_dir = TempDir::new().unwrap();
        let signals_base = temp_dir.path().join("signals");

        // Create the expected directory structure: base/cell_id/signals.yaml
        // NOT: base/signals/cell_id/signals.yaml
        let cell_dir = signals_base.join("86283082fffffff");
        fs::create_dir_all(&cell_dir).unwrap();

        // Create a test signal file
        let test_signal = PersistedSignal {
            frequency_hz: 88900000.0,
            signal_strength: 42.0,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 1,
            modulation: ModulationType::WFM,
            notes: Some("Test FM Station".to_string()),
        };

        // Write a signal file at the expected location
        let signals_file_content = format!(
            r#"
version: v1.0
signals:
  88900000:
    frequency_hz: 88900000.0
    signal_strength: 42.0
    first_detected: {}
    last_detected: {}
    detection_count: 1
    modulation: WFM
    notes: Test FM Station
metadata:
  h3_cell_id: "86283082fffffff"
  center_lat: 37.7749
  center_lon: -122.4194
  last_updated: {}
"#,
            test_signal.first_detected.to_rfc3339(),
            test_signal.last_detected.to_rfc3339(),
            Utc::now().to_rfc3339()
        );

        fs::write(cell_dir.join("signals.yaml"), signals_file_content).unwrap();

        // Test that SignalStorage can find the file
        let storage = SignalStorage::new(&signals_base);
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let result = storage.load_signals_for_location(location);

        assert!(
            result.is_ok(),
            "Should load signals successfully: {:?}",
            result.err()
        );
        let signals = result.unwrap();
        assert_eq!(signals.len(), 1, "Should load exactly one signal");
        assert_eq!(signals[0].frequency_hz, 88900000.0);
        assert_eq!(signals[0].modulation, ModulationType::WFM);
        assert_eq!(signals[0].notes, Some("Test FM Station".to_string()));
    }

    #[test]
    fn test_tui_signal_loading_integration() {
        // TDD RED: Test that TUI can load signals through the proper path
        // This tests the integration between TUI and SignalStorage

        let temp_dir = TempDir::new().unwrap();
        let project_root = temp_dir.path();
        let signals_dir = project_root.join("data").join("signals");

        // Create signal file at correct location
        let cell_dir = signals_dir.join("86283082fffffff");
        fs::create_dir_all(&cell_dir).unwrap();

        let signals_file_content = r#"
version: v1.0
signals:
  88900000:
    frequency_hz: 88900000.0
    signal_strength: 42.0
    first_detected: 2025-11-10T04:39:08.932571933Z
    last_detected: 2025-11-10T04:39:08.932574429Z
    detection_count: 1
    modulation: WFM
    notes: KRFC
metadata:
  h3_cell_id: "86283082fffffff"
  center_lat: 37.7749
  center_lon: -122.4194
  last_updated: 2025-11-10T04:41:34.858481386Z
"#;

        fs::write(cell_dir.join("signals.yaml"), signals_file_content).unwrap();

        // Test that TUI path construction works correctly
        // This simulates what get_signals_storage_path() should return
        let signals_storage_path = project_root.join("data").join("signals");
        let storage = SignalStorage::new(&signals_storage_path);

        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let result = storage.load_signals_for_location(location);
        assert!(
            result.is_ok(),
            "TUI should load signals: {:?}",
            result.err()
        );

        let signals = result.unwrap();
        assert_eq!(signals.len(), 1, "Should load the signal from storage");
        assert_eq!(signals[0].notes, Some("KRFC".to_string()));
    }
}
