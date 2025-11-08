//! Audio coordination system - processes tune requests from stations

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        system::{System, SystemContext},
    },
};

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
        let (signal_entities, audio_entities) =
            match (&context.signal_entities, &context.audio_entities) {
                (Some(se), Some(ae)) => (se.clone(), ae.clone()),
                _ => return Ok(()),
            };

        // ECS Pattern: Phase 1 - Read-only check to see if there's work to do
        // This avoids marking audio_entities as changed when nothing needs to change
        let (audios_to_stop, audios_to_remove) = {
            let signals = match signal_entities.try_read() {
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

            // Collect signal frequencies that are playing
            let signal_frequencies_playing: std::collections::HashSet<u64> = signals
                .iter()
                .filter(|s| s.playback.is_playing())
                .map(|s| (s.frequency() * 1000.0) as u64)
                .collect();

            // Find audios that need to be stopped (playing but signal is not playing)
            let to_stop: Vec<_> = audios
                .iter()
                .filter(|audio| {
                    let audio_freq_key = (audio.frequency() * 1000.0) as u64;
                    !signal_frequencies_playing.contains(&audio_freq_key) && audio.is_playing()
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

                // Clean up coordinator resources
                if let Some(streams_resource) = &context.audio_streams
                    && let Ok(mut streams) = streams_resource.try_lock()
                    && streams.remove(&audio_id).is_some()
                {
                    debug!(audio_id = ?audio_id, "Removed audio stream from coordinator resource");
                }

                if let Some(segments_resource) = &context.audio_segments
                    && let Ok(mut segments) = segments_resource.try_lock()
                    && segments.remove(&audio_id).is_some()
                {
                    debug!(audio_id = ?audio_id, "Removed audio segment from coordinator resource");
                }

                debug!(audio_id = ?audio_id, "Audio playback stopped");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, RwLock},
        time::SystemTime,
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, Signal},
        ecs::{AudioEntity, EntityWorld, SignalEntity, TaskId, components::window::WindowId},
    };

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

        let mut signal_world = EntityWorld::new();
        let window_id = WindowId::new(TaskId::new("test-scan".to_string()), 1);

        let signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        signal_world.insert(signal);

        let audio_world = EntityWorld::new();

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let signals = signal_entities.read().unwrap();
        for signal in signals.iter() {
            assert!(signal.tune_state.is_idle());
        }
    }

    #[test]
    fn test_stops_audio_when_playback_state_idle() {
        let mut system = CoordinationSystem::new();

        let mut signal_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);
        let window_id = WindowId::new(TaskId::new("test-scan".to_string()), 1);

        // StationEntity is no longer used after migration

        // Create SignalEntity with NotPlaying state (default)
        // This represents the signal that the audio is for, but it's not currently playing
        let signal_entity = crate::ecs::SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
        // Note: signal_entity.playback defaults to PlaybackState::NotPlaying
        signal_world.insert(signal_entity);

        let mut audio_world = EntityWorld::new();
        let audio = AudioEntity::new(signal, 88.9e6, None);
        assert!(audio.is_playing());
        audio_world.insert(audio);

        let audio_entities = Arc::new(RwLock::new(audio_world));
        let signal_entities = Arc::new(RwLock::new(signal_world));

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities.clone())
            .with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let audios = audio_entities.read().unwrap();
        for audio in audios.iter() {
            assert!(
                !audio.is_playing(),
                "Audio should be stopped when signal playback state is NotPlaying"
            );
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn test_removes_audio_entity_on_stop_listening_request() {
        let mut system = CoordinationSystem::new();

        let mut audio_world = EntityWorld::new();
        let mut signal_world = EntityWorld::new();
        let signal = create_test_signal(88.9e6);

        // Create AudioEntity with stop_listening_request
        let mut audio = AudioEntity::new(signal, 88.9e6, None);
        let audio_id = *audio.id();
        audio.request_stop_listening();
        audio_world.insert(audio);

        // Create SignalEntity - this provides the context the system needs
        // Note: SignalEntity playback state doesn't matter for this test since
        // the removal is based on stop_listening_request, not playing state
        let task_id = TaskId::new("test_task");
        let window_id = WindowId::new(task_id, 0);
        let signal_entity = crate::ecs::SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
        signal_world.insert(signal_entity);

        let audio_entities = Arc::new(RwLock::new(audio_world));
        let signal_entities = Arc::new(RwLock::new(signal_world));

        let audio_streams = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let audio_segments = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities.clone())
            .with_signal_entities(signal_entities.clone())
            .with_audio_streams(audio_streams.clone())
            .with_audio_segments(audio_segments.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let audios = audio_entities.read().unwrap();
        assert!(
            audios.get(&audio_id).is_none(),
            "AudioEntity should be removed when stop_listening_request is set"
        );

        let streams = audio_streams.lock().unwrap();
        assert!(
            streams.get(&audio_id).is_none(),
            "Audio stream should be removed from coordinator resources"
        );

        let segments = audio_segments.lock().unwrap();
        assert!(
            segments.get(&audio_id).is_none(),
            "Audio segment should be removed from coordinator resources"
        );
    }
}
