//! Audio management system

use crate::core::types::Result;
use crate::ecs::Entity;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

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
            let mut entities = audio_entities.write().unwrap();
            for audio_id in entities_to_remove {
                if let Some(mut entity) = entities.remove(&audio_id) {
                    entity.stop();
                    debug!(audio_id = ?audio_id, "Cleaned up audio session");
                }
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
    use crate::ecs::{AudioEntity, EntityWorld};
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, SystemTime};

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
}
