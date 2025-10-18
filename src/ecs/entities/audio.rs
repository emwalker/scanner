//! Audio entity combining audio components

use crate::core::types::Signal;
use crate::ecs::Entity;
use crate::ecs::components::audio::{
    AudioAllocationComponent, AudioId, AudioPlaybackComponent, AudioTuningComponent,
};
use crate::hardware::DeviceId;
use std::time::{Duration, Instant};

/// Entity representing an active audio playback session
pub struct AudioEntity {
    pub id: AudioId,
    pub tuning: AudioTuningComponent,
    pub playback: AudioPlaybackComponent,
    pub allocation: AudioAllocationComponent,
}

impl AudioEntity {
    pub fn new(signal: Signal, center_frequency_hz: f64, tuner_id: Option<DeviceId>) -> Self {
        Self {
            id: AudioId::new(),
            tuning: AudioTuningComponent::new(signal, center_frequency_hz),
            playback: AudioPlaybackComponent::new(),
            allocation: AudioAllocationComponent::new(tuner_id),
        }
    }

    pub fn frequency(&self) -> f64 {
        self.tuning.frequency()
    }

    pub fn signal_strength(&self) -> f32 {
        self.tuning.signal_strength()
    }

    pub fn is_playing(&self) -> bool {
        self.playback.is_playing()
    }

    pub fn play_duration(&self) -> Duration {
        self.playback.play_duration()
    }

    pub fn started_at(&self) -> Instant {
        self.playback.started_at()
    }

    pub fn tuner_id(&self) -> Option<&DeviceId> {
        self.allocation.tuner_id.as_ref()
    }

    pub fn stop(&mut self) {
        self.playback.stop();
        self.allocation.cancel_graph();
    }
}

impl Entity for AudioEntity {
    type Id = AudioId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioQuality;
    use crate::core::types::ModulationType;
    use std::thread;
    use std::time::SystemTime;

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
    fn test_create_audio_entity() {
        let signal = create_test_signal();
        let device_id = DeviceId::from_serial("test", "device");
        let audio = AudioEntity::new(signal.clone(), 88.9e6, Some(device_id.clone()));

        assert_eq!(audio.frequency(), 88.9e6);
        assert_eq!(audio.signal_strength(), 0.8);
        assert!(audio.is_playing());
        assert_eq!(audio.tuner_id(), Some(&device_id));
    }

    #[test]
    fn test_create_without_tuner() {
        let signal = create_test_signal();
        let audio = AudioEntity::new(signal, 88.9e6, None);

        assert_eq!(audio.tuner_id(), None);
        assert!(audio.is_playing());
    }

    #[test]
    fn test_stop() {
        let signal = create_test_signal();
        let mut audio = AudioEntity::new(signal, 88.9e6, None);

        audio.stop();
        assert!(!audio.is_playing());
    }

    #[test]
    fn test_play_duration() {
        let signal = create_test_signal();
        let audio = AudioEntity::new(signal, 88.9e6, None);

        thread::sleep(Duration::from_millis(10));
        let duration = audio.play_duration();
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_entity_trait() {
        let signal = create_test_signal();
        let audio = AudioEntity::new(signal, 88.9e6, None);
        let id = audio.id();

        assert_eq!(id, &audio.id);
    }

    #[test]
    fn test_convenience_methods() {
        let signal = create_test_signal();
        let audio = AudioEntity::new(signal.clone(), 88.9e6, None);

        assert_eq!(audio.frequency(), signal.frequency_hz);
        assert_eq!(audio.signal_strength(), signal.signal_strength);
        assert!(audio.started_at().elapsed().as_millis() < 10);
    }
}
