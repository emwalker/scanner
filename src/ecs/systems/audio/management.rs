//! Audio management system

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        system::{System, SystemContext},
    },
};

/// System that manages audio playback sessions
///
/// This system:
/// - Monitors active audio entities
/// - Cleans up stopped or expired sessions
/// - Coordinates audio with tuner allocation
pub struct ManagementSystem {
    max_session_duration_secs: Option<u64>,
}

impl Default for ManagementSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ManagementSystem {
    pub fn new() -> Self {
        Self {
            max_session_duration_secs: None,
        }
    }

    pub fn with_max_duration(mut self, duration_secs: u64) -> Self {
        self.max_session_duration_secs = Some(duration_secs);
        self
    }
}

impl System for ManagementSystem {
    fn name(&self) -> &'static str {
        "AudioManagement"
    }

    #[allow(clippy::cognitive_complexity)]
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let audio_entities = match &context.audio_entities {
            Some(entities) => entities.clone(),
            None => {
                debug!("No audio entities in context");
                return Ok(());
            }
        };

        let mut entities_to_remove = Vec::new();

        {
            let entities = audio_entities.read().unwrap();

            for entity in entities.iter() {
                if !entity.is_playing() {
                    debug!(audio_id = ?entity.id(), "Audio session stopped, marking for cleanup");
                    entities_to_remove.push(*entity.id());
                    continue;
                }

                if let Some(max_duration) = self.max_session_duration_secs {
                    let duration = entity.play_duration();
                    if duration.as_secs() > max_duration {
                        debug!(
                            audio_id = ?entity.id(),
                            duration_secs = duration.as_secs(),
                            max_duration_secs = max_duration,
                            "Audio session exceeded max duration, marking for cleanup"
                        );
                        entities_to_remove.push(*entity.id());
                    }
                }
            }
        }

        if !entities_to_remove.is_empty() {
            // Collect frequencies of audio being removed for signal updates
            let mut audio_frequencies = Vec::new();
            {
                let mut entities = audio_entities.write().unwrap();
                for audio_id in entities_to_remove.iter() {
                    if let Some(mut entity) = entities.remove(audio_id) {
                        audio_frequencies.push(entity.frequency());
                        entity.stop();
                        debug!(audio_id = ?audio_id, "Cleaned up audio session");
                    }
                }
            } // Release lock

            // Clean up audio segments (releases tuners back to pool via Drop)
            if let Some(audio_segments) = &context.audio_segments
                && let Ok(mut segments) = audio_segments.try_lock()
            {
                for audio_id in &entities_to_remove {
                    if let Some(segment) = segments.remove(audio_id) {
                        drop(segment);
                        debug!(audio_id = ?audio_id, "Cleaned up audio segment, tuner returned to pool");
                    }
                }
            }

            // Clean up audio streams
            if let Some(audio_streams) = &context.audio_streams
                && let Ok(mut streams) = audio_streams.try_lock()
            {
                for audio_id in &entities_to_remove {
                    if streams.remove(audio_id).is_some() {
                        debug!(audio_id = ?audio_id, "Cleaned up audio stream");
                    }
                }
            }

            // Update signal states for removed audio
            let mut windows_to_clear = Vec::new();
            if let Some(signal_entities) = &context.signal_entities
                && let Ok(mut signals) = signal_entities.try_write()
            {
                const FREQ_TOLERANCE_HZ: f64 = 1000.0;
                for freq in &audio_frequencies {
                    // Find signals with matching frequency and mark as completed
                    for signal in signals.iter_mut() {
                        if (signal.frequency() - freq).abs() < FREQ_TOLERANCE_HZ
                            && signal.playback.is_playing()
                        {
                            signal.playback.transition_to(
                                crate::ecs::components::signal::PlaybackState::Completed,
                            );
                            signal.history.end_play_session();
                            windows_to_clear.push(signal.window_id().clone());
                            debug!(
                                signal_id = ?signal.id(),
                                frequency_mhz = freq / 1e6,
                                "ManagementSystem: Transitioned signal to Completed after removing audio"
                            );
                        }
                    }
                }
            }

            // Clear window playback state so the next queued signal can start
            if let Some(window_entities) = &context.window_entities {
                use std::collections::HashSet;
                let unique_windows: HashSet<_> = windows_to_clear.into_iter().collect();

                if let Ok(mut windows) = window_entities.try_write() {
                    for window in windows.iter_mut() {
                        if unique_windows.contains(window.id()) {
                            window.allocation.stop_playing();
                            debug!(
                                window_index = window.window_index(),
                                "ManagementSystem: Cleared window current_playing after audio \
                                 completion"
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, RwLock},
        time::{Duration, SystemTime},
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, Signal},
        ecs::{AudioEntity, EntityWorld},
    };

    fn create_test_signal() -> Signal {
        Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_management_system_with_empty_context() {
        let mut system = ManagementSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_management_system_with_playing_sessions() {
        let mut system = ManagementSystem::new();

        let mut world = EntityWorld::new();
        let signal = create_test_signal();
        world.insert(AudioEntity::new(signal.clone(), 88.9e6, None));
        world.insert(AudioEntity::new(signal, 89.3e6, None));

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_audio_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_management_system_removes_stopped_sessions() {
        let mut system = ManagementSystem::new();

        let mut world = EntityWorld::new();
        let signal = create_test_signal();
        let mut stopped_audio = AudioEntity::new(signal.clone(), 88.9e6, None);
        stopped_audio.stop();
        world.insert(stopped_audio);
        world.insert(AudioEntity::new(signal, 89.3e6, None));

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_audio_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(entities.len(), 1);
        assert!(entities.iter().all(|e| e.is_playing()));
    }

    #[test]
    fn test_cleanup_clears_station_history_play_start() {
        use crate::ecs::{SignalEntity, StationEntity, TaskId, components::window::WindowId};

        let mut system = ManagementSystem::new();

        let signal = create_test_signal();
        let window_id = WindowId::new(TaskId::new("test-scan"), 1);

        let mut station = StationEntity::from_signal(&signal, window_id.clone());

        // Create SignalEntity and set it to playing
        let mut signal_entity = SignalEntity::new(88.9e6, window_id);
        signal_entity
            .playback
            .transition_to(crate::ecs::components::signal::PlaybackState::Playing);
        signal_entity.history.start_play_session();

        // Record initial play count before system runs
        let initial_play_count = signal_entity.history.play_count();

        let mut audio_entity = AudioEntity::new(signal, 88.9e6, None);
        let audio_id = *audio_entity.id();

        station.playback.start_playing(audio_id);

        audio_entity.stop();

        let mut station_world = EntityWorld::new();
        station_world.insert(station);

        let mut signal_world = EntityWorld::new();
        signal_world.insert(signal_entity);

        let mut audio_world = EntityWorld::new();
        audio_world.insert(audio_entity);

        let _station_entities = Arc::new(RwLock::new(station_world));
        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities)
            .with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let signals = signal_entities.read().unwrap();
        let signal = signals.iter().next().unwrap();

        assert_eq!(
            signal.history.play_count(),
            initial_play_count + 1,
            "play_count should be incremented when play session ends (indicating end_play_session \
             was called)"
        );

        assert_eq!(
            signal.playback.state(),
            crate::ecs::components::signal::PlaybackState::Completed,
            "signal playback state should be set to Completed when audio stops"
        );
    }

    #[test]
    fn test_management_system_respects_max_duration() {
        let mut system = ManagementSystem::new().with_max_duration(1);

        let mut world = EntityWorld::new();
        let signal = create_test_signal();
        world.insert(AudioEntity::new(signal, 88.9e6, None));

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_audio_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        {
            let entities = context_entities.read().unwrap();
            assert_eq!(
                entities.len(),
                1,
                "Session should still exist before duration exceeded"
            );
        }

        {
            let mut entities = context_entities.write().unwrap();
            let entity = entities.iter_mut().next().unwrap();
            let old_time = entity.playback.started_at() - Duration::from_secs(2);
            entity.playback.set_started_at_for_test(old_time);
        }

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(
            entities.len(),
            0,
            "Session should be removed after exceeding max duration"
        );
    }

    #[test]
    fn test_management_system_with_no_max_duration() {
        let mut system = ManagementSystem::new();

        let mut world = EntityWorld::new();
        let signal = create_test_signal();
        world.insert(AudioEntity::new(signal, 88.9e6, None));

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_audio_entities(context_entities.clone());

        {
            let mut entities = context_entities.write().unwrap();
            let entity = entities.iter_mut().next().unwrap();
            let old_time = entity.playback.started_at() - Duration::from_secs(3600);
            entity.playback.set_started_at_for_test(old_time);
        }

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(
            entities.len(),
            1,
            "Session should remain without max duration set"
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn test_cleanup_removes_segments_and_streams() {
        use std::{collections::HashMap, sync::Mutex};

        use cpal::traits::{DeviceTrait, HostTrait};

        use crate::ecs::components::audio::AudioId;

        let mut system = ManagementSystem::new();

        let signal = create_test_signal();
        let mut audio_entity = AudioEntity::new(signal, 88.9e6, None);
        let audio_id = *audio_entity.id();

        audio_entity.stop();

        let mut audio_world = EntityWorld::new();
        audio_world.insert(audio_entity);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let audio_streams = Arc::new(Mutex::new(HashMap::<AudioId, cpal::Stream>::new()));
        let audio_segments = Arc::new(Mutex::new(HashMap::new()));

        {
            let mut streams = audio_streams.lock().unwrap();
            streams.insert(
                audio_id,
                cpal::default_host()
                    .default_output_device()
                    .unwrap()
                    .build_output_stream(
                        &cpal::StreamConfig {
                            channels: 2,
                            sample_rate: cpal::SampleRate(48000),
                            buffer_size: cpal::BufferSize::Default,
                        },
                        |_data: &mut [f32], _: &cpal::OutputCallbackInfo| {},
                        |_err| {},
                        None,
                    )
                    .unwrap(),
            );
        }

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities.clone())
            .with_audio_streams(audio_streams.clone())
            .with_audio_segments(audio_segments.clone());

        assert_eq!(
            audio_streams.lock().unwrap().len(),
            1,
            "Stream should exist before cleanup"
        );

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let audio_world = audio_entities.read().unwrap();
        assert_eq!(audio_world.len(), 0, "AudioEntity should be removed");

        assert_eq!(
            audio_streams.lock().unwrap().len(),
            0,
            "Stream should be removed when audio entity is cleaned up"
        );

        assert_eq!(
            audio_segments.lock().unwrap().len(),
            0,
            "Segment should be removed when audio entity is cleaned up"
        );
    }
}
