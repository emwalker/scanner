//! UI update system - syncs entity state to TUI Model

use crate::core::types::Result;
use crate::ecs::system::{System, SystemContext};
use crate::ui::tui::model::types::SpectrumStation;
use tracing::debug;

/// System that updates TUI model with entity state
///
/// This system:
/// - Queries StationEntity for discovered stations
/// - Queries AudioEntity for active playback
/// - Populates Model spectrum fields
/// - Maintains TEA pattern (Model is single source of UI truth)
pub struct UIUpdateSystem {
    stations: Vec<SpectrumStation>,
    active_frequency: Option<f64>,
}

impl Default for UIUpdateSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl UIUpdateSystem {
    pub fn new() -> Self {
        Self {
            stations: Vec::new(),
            active_frequency: None,
        }
    }

    /// Get discovered stations for spectrum display
    pub fn stations(&self) -> &[SpectrumStation] {
        &self.stations
    }

    /// Get active audio frequency if playing
    pub fn active_frequency(&self) -> Option<f64> {
        self.active_frequency
    }
}

impl System for UIUpdateSystem {
    fn name(&self) -> &'static str {
        "UIUpdate"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        self.stations.clear();
        self.active_frequency = None;

        if let Some(ref station_entities) = context.station_entities {
            let entities = station_entities.lock().unwrap();

            for entity in entities.iter() {
                self.stations.push(SpectrumStation {
                    frequency_hz: entity.frequency(),
                    signal_strength: entity.info.signal_strength,
                    is_active: false,
                });
            }

            debug!(
                station_count = self.stations.len(),
                "UIUpdateSystem collected stations"
            );
        }

        if let Some(ref audio_entities) = context.audio_entities {
            let entities = audio_entities.lock().unwrap();

            if let Some(audio_entity) = entities.iter().find(|e| e.is_playing()) {
                let freq = audio_entity.frequency();
                self.active_frequency = Some(freq);

                for station in &mut self.stations {
                    if (station.frequency_hz - freq).abs() < 1000.0 {
                        station.is_active = true;
                    }
                }

                debug!(frequency_hz = freq, "UIUpdateSystem found active audio");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioQuality;
    use crate::core::types::{ModulationType, Signal};
    use crate::ecs::{AudioEntity, EntityWorld, StationEntity};
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

    fn create_test_signal(frequency: f64) -> Signal {
        Signal {
            frequency_hz: frequency,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: frequency,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_ui_update_system_with_no_entities() {
        let mut system = UIUpdateSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 0);
        assert_eq!(system.active_frequency(), None);
    }

    #[test]
    fn test_ui_update_system_with_stations() {
        let mut system = UIUpdateSystem::new();

        let mut station_world = EntityWorld::new();
        let signal1 = create_test_signal(88.9e6);
        let signal2 = create_test_signal(95.5e6);

        let scan_id = crate::ecs::ScanId::new();
        let metadata = crate::scanning::window::WindowMetadata {
            window_id: 0,
            center_frequency_hz: 88.9e6,
        };

        station_world.insert(StationEntity::from_signal(&signal1, scan_id, metadata));
        station_world.insert(StationEntity::from_signal(&signal2, scan_id, metadata));

        let context_entities = Arc::new(Mutex::new(station_world));
        let mut context = SystemContext::new().with_station_entities(context_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 2);
        assert!(system.stations().iter().any(|s| s.frequency_hz == 88.9e6));
        assert!(system.stations().iter().any(|s| s.frequency_hz == 95.5e6));
    }

    #[test]
    fn test_ui_update_system_with_active_audio() {
        let mut system = UIUpdateSystem::new();

        let mut station_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);

        let scan_id = crate::ecs::ScanId::new();
        let metadata = crate::scanning::window::WindowMetadata {
            window_id: 0,
            center_frequency_hz: 88.9e6,
        };

        station_world.insert(StationEntity::from_signal(&signal, scan_id, metadata));

        let mut audio_world = EntityWorld::new();
        audio_world.insert(AudioEntity::new(signal, 88.9e6, None));

        let station_entities = Arc::new(Mutex::new(station_world));
        let audio_entities = Arc::new(Mutex::new(audio_world));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities)
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 1);
        assert_eq!(system.active_frequency(), Some(88.9e6));
        assert!(system.stations()[0].is_active);
    }

    #[test]
    fn test_ui_update_system_marks_correct_station_active() {
        let mut system = UIUpdateSystem::new();

        let mut station_world = EntityWorld::new();
        let scan_id = crate::ecs::ScanId::new();
        let metadata = crate::scanning::window::WindowMetadata {
            window_id: 0,
            center_frequency_hz: 88.9e6,
        };

        station_world.insert(StationEntity::from_signal(
            &create_test_signal(88.9e6),
            scan_id,
            metadata,
        ));
        station_world.insert(StationEntity::from_signal(
            &create_test_signal(95.5e6),
            scan_id,
            metadata,
        ));

        let mut audio_world = EntityWorld::new();
        audio_world.insert(AudioEntity::new(create_test_signal(95.5e6), 95.5e6, None));

        let station_entities = Arc::new(Mutex::new(station_world));
        let audio_entities = Arc::new(Mutex::new(audio_world));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities)
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let active_station = system.stations().iter().find(|s| s.is_active).unwrap();
        assert_eq!(active_station.frequency_hz, 95.5e6);

        let inactive_station = system.stations().iter().find(|s| !s.is_active).unwrap();
        assert_eq!(inactive_station.frequency_hz, 88.9e6);
    }
}
