use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::signals::ModulationType;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PersistedSignal {
    pub frequency_hz: f64,
    pub signal_strength: f64,
    pub first_detected: DateTime<Utc>,
    pub last_detected: DateTime<Utc>,
    pub detection_count: u32,
    pub modulation: ModulationType,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct CellMetadata {
    pub h3_cell_id: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct SignalsFile {
    pub version: String,
    pub signals: BTreeMap<u64, PersistedSignal>, // frequency_hz as key
    pub metadata: CellMetadata,
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_persisted_signal_yaml_serialization() {
        let signal = PersistedSignal {
            frequency_hz: 88900000.0,
            signal_strength: 0.85,
            first_detected: Utc.with_ymd_and_hms(2024, 11, 9, 15, 30, 0).unwrap(),
            last_detected: Utc.with_ymd_and_hms(2024, 11, 9, 16, 45, 0).unwrap(),
            detection_count: 12,
            modulation: ModulationType::WFM,
            notes: Some("Classical music station".to_string()),
        };

        let yaml = serde_yaml::to_string(&signal).unwrap();
        let deserialized: PersistedSignal = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(signal, deserialized);
        assert!(yaml.contains("frequency_hz: 88900000"));
        assert!(yaml.contains("Classical music station"));
    }

    #[test]
    fn test_yaml_deserialization_handles_legacy_fm_modulation() {
        // TDD RED: Test that YAML with "FM" modulation can be parsed
        // This should fail initially because "FM" is not a valid ModulationType variant
        let yaml_content = r#"
frequency_hz: 88900000.0
signal_strength: 0.0
first_detected: 2025-11-10T04:39:08.932571933Z
last_detected: 2025-11-10T04:39:08.932574429Z
detection_count: 1
modulation: FM
notes: KRFC
"#;

        let result: Result<PersistedSignal, serde_yaml::Error> = serde_yaml::from_str(yaml_content);

        // This test should pass - we want to handle legacy "FM" in YAML
        assert!(
            result.is_ok(),
            "Should parse YAML with legacy FM modulation: {:?}",
            result.err()
        );

        let signal = result.unwrap();
        assert_eq!(signal.frequency_hz, 88900000.0);
        assert_eq!(signal.notes, Some("KRFC".to_string()));
        // FM should map to WFM (wideband FM) as the most common FM type
        assert_eq!(signal.modulation, ModulationType::WFM);
    }

    #[test]
    fn test_signals_file_btreemap_ordering() {
        let mut signals_file = SignalsFile {
            version: "v1.0".to_string(),
            signals: BTreeMap::new(),
            metadata: CellMetadata::default(),
        };

        // Add signals in reverse frequency order to test BTreeMap sorting
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

        signals_file.signals.insert(107900000, signal_high);
        signals_file.signals.insert(88500000, signal_low);

        let yaml = serde_yaml::to_string(&signals_file).unwrap();

        // BTreeMap should serialize lower frequency first
        let low_pos = yaml.find("88500000").unwrap();
        let high_pos = yaml.find("107900000").unwrap();
        assert!(
            low_pos < high_pos,
            "Lower frequency should appear first in YAML"
        );
    }
}
