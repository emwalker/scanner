//! Audio coordination system - processes tune requests from stations

use crate::core::types::Result;
use crate::ecs::Entity;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

/// System that processes TuneRequest components and creates AudioEntity
///
/// Flow:
/// 1. Query StationEntity with tune_request.is_some()
/// 2. Create AudioEntity for requested station
/// 3. Clear tune_request from StationEntity
/// 4. UIUpdateSystem will mark station as active based on AudioEntity.is_playing()
pub struct CoordinationSystem;

impl Default for CoordinationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinationSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for CoordinationSystem {
    fn name(&self) -> &'static str {
        "AudioCoordination"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let (station_entities, audio_entities) =
            match (&context.station_entities, &context.audio_entities) {
                (Some(se), Some(ae)) => (se.clone(), ae.clone()),
                _ => return Ok(()),
            };

        let mut stations = station_entities.write().unwrap();
        let mut audios = audio_entities.write().unwrap();

        // First, check if any audio entities need to be stopped because their station's tune request was cleared
        let station_frequencies_with_tune_requests: std::collections::HashSet<u64> = stations
            .iter()
            .filter(|s| s.has_tune_request())
            .map(|s| (s.frequency() * 1000.0) as u64)
            .collect();

        for audio in audios.iter_mut() {
            let audio_freq_key = (audio.frequency() * 1000.0) as u64;
            if !station_frequencies_with_tune_requests.contains(&audio_freq_key)
                && audio.is_playing()
            {
                debug!(
                    audio_id = ?audio.id(),
                    frequency_mhz = audio.frequency() / 1e6,
                    "AudioCoordinationSystem: Stopping audio (tune request cleared)"
                );
                audio.stop();
            }
        }

        // Process stop_listening requests
        for audio in audios.iter_mut() {
            if audio.stop_listening_request.is_some() {
                debug!(
                    audio_id = ?audio.id(),
                    frequency_mhz = audio.frequency() / 1e6,
                    "AudioCoordinationSystem: Processing stop_listening request"
                );
                audio.stop();
                audio.clear_stop_listening_request();
            }
        }

        for station in stations.iter_mut() {
            if let Some(ref tune_request) = station.tune_request {
                let station_id = *station.id();
                debug!(
                    station_id = ?station_id,
                    frequency_mhz = station.frequency() / 1e6,
                    window_id = tune_request.window_id,
                    "AudioCoordinationSystem: Processing tune request"
                );

                // Create AudioEntity for listening
                // TODO: Allocate actual tuner and create real audio session
                // For now, create a placeholder Signal from station info
                use crate::core::types::{ModulationType, Signal};
                let signal = Signal {
                    frequency_hz: station.frequency(),
                    signal_strength: station.signal_strength(),
                    bandwidth_hz: 200_000.0,
                    modulation: ModulationType::WFM,
                    audio_sample_rate: 48000,
                    detected_at: std::time::SystemTime::now(),
                    analysis_duration_ms: 100,
                    detection_center_freq: tune_request.center_frequency,
                    audio_quality: station
                        .info
                        .audio_quality
                        .unwrap_or(crate::audio::quality::AudioQuality::Unknown),
                };

                let audio_entity = crate::ecs::AudioEntity::new(
                    signal,
                    tune_request.center_frequency,
                    None, // TODO: allocate tuner from pool
                );

                debug!(
                    station_id = ?station_id,
                    audio_id = ?audio_entity.id(),
                    frequency_mhz = station.frequency() / 1e6,
                    "AudioCoordinationSystem: Created AudioEntity (placeholder - no actual audio)"
                );

                audios.insert(audio_entity);
                station.clear_tune_request();
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
    use crate::ecs::{EntityWorld, ScanId, StationEntity};
    use crate::scanning::window::WindowMetadata;
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
    fn test_no_tune_requests() {
        let mut system = CoordinationSystem::new();

        let mut station_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);
        let metadata = WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        };

        station_world.insert(StationEntity::from_signal(&signal, ScanId::new(), metadata));

        let audio_world = EntityWorld::new();

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities.clone())
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(!station.has_tune_request());
        }
    }

    #[test]
    fn test_processes_tune_request() {
        let mut system = CoordinationSystem::new();

        let mut station_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);
        let metadata = WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        };

        let mut station = StationEntity::from_signal(&signal, ScanId::new(), metadata);
        station.request_tune(1, 88.9e6);
        assert!(station.has_tune_request());

        station_world.insert(station);

        let audio_world = EntityWorld::new();

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities.clone())
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(
                !station.has_tune_request(),
                "Tune request should be cleared"
            );
        }
    }
}
