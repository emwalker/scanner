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

        // ECS Pattern: Phase 1 - Read-only check to see if there's work to do
        // This avoids marking audio_entities as changed when nothing needs to change
        let (audios_to_stop, audios_to_remove) = {
            let stations = match station_entities.try_read() {
                Ok(s) => s,
                Err(_) => return Ok(()),
            };

            let audios = match audio_entities.try_read() {
                Ok(a) => a,
                Err(_) => return Ok(()),
            };

            // Early exit if no audio entities exist
            if audios.is_empty() {
                return Ok(());
            }

            // Collect station frequencies that are playing
            let station_frequencies_playing: std::collections::HashSet<u64> = stations
                .iter()
                .filter(|s| s.playback.is_playing())
                .map(|s| (s.frequency() * 1000.0) as u64)
                .collect();

            // Find audios that need to be stopped (playing but station is idle)
            let to_stop: Vec<_> = audios
                .iter()
                .filter(|audio| {
                    let audio_freq_key = (audio.frequency() * 1000.0) as u64;
                    !station_frequencies_playing.contains(&audio_freq_key) && audio.is_playing()
                })
                .map(|a| (*a.id(), a.frequency()))
                .collect();

            // Find audios that need to be removed (stop_listening_request)
            let to_remove: Vec<_> = audios
                .iter()
                .filter(|a| a.stop_listening_request.is_some())
                .map(|a| (*a.id(), a.frequency()))
                .collect();

            (to_stop, to_remove)
        };

        // Early exit if no work to do
        if audios_to_stop.is_empty() && audios_to_remove.is_empty() {
            return Ok(());
        }

        // ECS Pattern: Phase 2 - Acquire write lock and mutate
        // Only reached if there's actual work to do
        let mut audios = match audio_entities.try_write() {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };

        // Stop audio entities
        for (audio_id, frequency) in audios_to_stop {
            if let Some(audio) = audios.get_mut(&audio_id) {
                debug!(
                    audio_id = ?audio_id,
                    frequency_mhz = frequency / 1e6,
                    "AudioCoordinationSystem: Stopping audio (station playback state changed to Idle)"
                );
                audio.stop();
            }
        }

        // Remove audio entities
        for (audio_id, frequency) in audios_to_remove {
            if let Some(mut audio) = audios.remove(&audio_id) {
                debug!(
                    audio_id = ?audio_id,
                    frequency_mhz = frequency / 1e6,
                    "AudioCoordinationSystem: Stopping audio playback"
                );

                audio.allocation.cancel_graph();

                if let Some(handle) = audio.allocation.take_thread() {
                    debug!(audio_id = ?audio_id, "Waiting for audio graph thread to finish");
                    let _ = handle.join();
                    debug!(audio_id = ?audio_id, "Audio graph thread finished");
                }

                debug!(audio_id = ?audio_id, "Audio playback stopped");
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
    use crate::ecs::{AudioEntity, EntityWorld, ScanId, StationEntity};
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
    fn test_stops_audio_when_playback_state_idle() {
        let mut system = CoordinationSystem::new();

        let mut station_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);
        let metadata = WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        };

        let station = StationEntity::from_signal(&signal, ScanId::new(), metadata);
        station_world.insert(station);

        let mut audio_world = EntityWorld::new();
        let audio = AudioEntity::new(signal, 88.9e6, None);
        assert!(audio.is_playing());
        audio_world.insert(audio);

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities.clone())
            .with_audio_entities(audio_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let audios = audio_entities.read().unwrap();
        for audio in audios.iter() {
            assert!(
                !audio.is_playing(),
                "Audio should be stopped when station playback state is Idle"
            );
        }
    }
}
