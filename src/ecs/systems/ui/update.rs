//! UI update system - syncs entity state to TUI Model

use crate::audio::quality::AudioQuality;
use crate::core::types::Result;
use crate::ecs::system::{System, SystemContext};
use crate::ecs::{CandidateState, Entity};
use crate::hardware::pool::PoolStatus;
use crate::ui::TuiEvent;
use crate::ui::tui::model::types::SpectrumStation;
use indexmap::IndexMap;
use std::sync::mpsc::Sender;
use tracing::debug;

/// Candidate data for TUI display
#[derive(Debug, Clone)]
pub struct CandidateData {
    pub candidate_id: String,
    pub frequency_hz: f64,
    pub state: CandidateState,
    pub completion: f64,
    pub audio_quality: Option<AudioQuality>,
    pub signal_strength: Option<f64>,
    pub transition_status: Option<String>,
}

/// System that updates TUI model with entity state
///
/// This system:
/// - Queries StationEntity for discovered stations
/// - Queries AudioEntity for active playback
/// - Queries CandidateEntity for scanning progress
/// - Queries TunerEntity for pool status
/// - Sends ActiveTunersUpdated events when pool status changes
/// - Populates Model spectrum fields and candidate lists
/// - Maintains TEA pattern (Model is single source of UI truth)
pub struct UIUpdateSystem {
    stations: Vec<SpectrumStation>,
    active_frequency: Option<f64>,
    candidates_by_window: IndexMap<usize, Vec<CandidateData>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    last_pool_status: Option<PoolStatus>,
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
            candidates_by_window: IndexMap::new(),
            tui_event_sender: None,
            last_pool_status: None,
        }
    }

    pub fn with_tui_event_sender(mut self, sender: Sender<TuiEvent>) -> Self {
        self.tui_event_sender = Some(sender);
        self
    }

    /// Get discovered stations for spectrum display
    pub fn stations(&self) -> &[SpectrumStation] {
        &self.stations
    }

    /// Get active audio frequency if playing
    pub fn active_frequency(&self) -> Option<f64> {
        self.active_frequency
    }

    /// Get candidates grouped by window ID (preserves insertion order)
    pub fn candidates_by_window(&self) -> &IndexMap<usize, Vec<CandidateData>> {
        &self.candidates_by_window
    }

    fn send_pool_status_update(&mut self, status: PoolStatus) {
        debug!(
            total_tuners = status.tuners.len(),
            available_count = status.available_tuner_count,
            allocated_count = status.allocated_tuner_count,
            has_sender = self.tui_event_sender.is_some(),
            "UIUpdateSystem: Pool status changed, attempting to send event"
        );

        for tuner in &status.tuners {
            debug!(
                tuner_id = ?tuner.id,
                state = ?tuner.state,
                activity = ?tuner.activity,
                "UIUpdateSystem: Tuner in status being sent"
            );
        }

        if let Some(ref sender) = self.tui_event_sender {
            match sender.send(TuiEvent::ActiveTunersUpdated {
                status: status.clone(),
            }) {
                Ok(_) => debug!("UIUpdateSystem: Event sent successfully"),
                Err(e) => debug!(error = ?e, "UIUpdateSystem: Failed to send event"),
            }
        } else {
            debug!("UIUpdateSystem: No TUI event sender configured");
        }

        self.last_pool_status = Some(status);
    }
}

impl System for UIUpdateSystem {
    fn name(&self) -> &'static str {
        "UIUpdate"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        self.stations.clear();
        self.active_frequency = None;
        self.candidates_by_window.clear();

        if let Some(ref tuner_entities) = context.tuner_entities
            && let Ok(entities) = tuner_entities.try_lock()
        {
            let status = crate::hardware::pool::Pool::build_status_from_entities(&entities, 0);

            let status_changed = match &self.last_pool_status {
                None => true,
                Some(last) => {
                    last.available_tuner_count != status.available_tuner_count
                        || last.allocated_tuner_count != status.allocated_tuner_count
                        || last.tuners.len() != status.tuners.len()
                        || last
                            .tuners
                            .iter()
                            .zip(&status.tuners)
                            .any(|(a, b)| a.state != b.state || a.activity != b.activity)
                }
            };

            if status_changed {
                self.send_pool_status_update(status);
            }
        }

        if let Some(ref station_entities) = context.station_entities {
            let entities = station_entities.read().unwrap();

            for entity in entities.iter() {
                self.stations.push(SpectrumStation {
                    frequency_hz: entity.frequency(),
                    signal_strength: entity.info.signal_strength,
                    audio_quality: entity.info.audio_quality,
                    is_active: entity.playback.is_playing(),
                });
            }
        }

        if let Some(ref candidate_entities) = context.candidate_entities {
            let entities = candidate_entities.read().unwrap();

            // Build frequency-to-transition-status lookup from stations
            let transition_statuses: std::collections::HashMap<u64, String> =
                if let Some(ref station_entities) = context.station_entities {
                    let stations = station_entities.read().unwrap();
                    stations
                        .iter()
                        .filter_map(|s| {
                            s.transition.as_ref().map(|t| {
                                (
                                    (s.frequency() * 1000.0) as u64,
                                    t.status_message().to_string(),
                                )
                            })
                        })
                        .collect()
                } else {
                    std::collections::HashMap::new()
                };

            for entity in entities.iter() {
                let window_id = entity.progress.metadata.window_id;
                let freq_key = (entity.info.frequency_hz * 1000.0) as u64;
                let transition_status = transition_statuses.get(&freq_key).cloned();

                let candidate_data = CandidateData {
                    candidate_id: entity.id().as_str().to_string(),
                    frequency_hz: entity.info.frequency_hz,
                    state: entity.lifecycle.state(),
                    completion: entity.completion(),
                    audio_quality: entity.info.audio_quality,
                    signal_strength: entity.info.signal_strength,
                    transition_status,
                };

                self.candidates_by_window
                    .entry(window_id)
                    .or_default()
                    .push(candidate_data);
            }
        }

        if let Some(ref audio_entities) = context.audio_entities {
            let entities = audio_entities.read().unwrap();

            if let Some(audio_entity) = entities.iter().find(|e| e.is_playing()) {
                let freq = audio_entity.frequency();
                self.active_frequency = Some(freq);

                let mut marked_count = 0;
                for station in &mut self.stations {
                    if (station.frequency_hz - freq).abs() < 1000.0 {
                        station.is_active = true;
                        marked_count += 1;
                    }
                }

                for candidates in self.candidates_by_window.values_mut() {
                    for candidate in candidates {
                        if (candidate.frequency_hz - freq).abs() < 1000.0 {
                            candidate.state = CandidateState::Playing;
                        }
                    }
                }

                debug!(
                    frequency_mhz = freq / 1e6,
                    marked_count = marked_count,
                    "UIUpdateSystem found active audio and marked stations"
                );
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
    use std::sync::{Arc, RwLock};
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

        let context_entities = Arc::new(RwLock::new(station_world));
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

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

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

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

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
