//! Station playback component

use crate::ecs::AudioId;

/// Playback state for a station
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationPlaybackState {
    /// No audio playing
    Idle,
    /// Audio entity active
    Playing,
}

/// Component tracking whether station is being listened to
#[derive(Debug, Clone)]
pub struct StationPlaybackComponent {
    state: StationPlaybackState,
    audio_id: Option<AudioId>,
}

impl StationPlaybackComponent {
    pub fn new() -> Self {
        Self {
            state: StationPlaybackState::Idle,
            audio_id: None,
        }
    }

    pub fn state(&self) -> StationPlaybackState {
        self.state
    }

    pub fn audio_id(&self) -> Option<AudioId> {
        self.audio_id
    }

    pub fn is_playing(&self) -> bool {
        self.state == StationPlaybackState::Playing
    }

    pub fn is_idle(&self) -> bool {
        self.state == StationPlaybackState::Idle
    }

    pub fn start_playing(&mut self, audio_id: AudioId) {
        self.state = StationPlaybackState::Playing;
        self.audio_id = Some(audio_id);
    }

    pub fn stop_playing(&mut self) {
        self.state = StationPlaybackState::Idle;
        self.audio_id = None;
    }
}

impl Default for StationPlaybackComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_playback_is_idle() {
        let playback = StationPlaybackComponent::new();
        assert!(playback.is_idle());
        assert!(!playback.is_playing());
        assert_eq!(playback.audio_id(), None);
    }

    #[test]
    fn test_start_playing() {
        let mut playback = StationPlaybackComponent::new();
        let audio_id = AudioId::new();

        playback.start_playing(audio_id);

        assert!(playback.is_playing());
        assert!(!playback.is_idle());
        assert_eq!(playback.audio_id(), Some(audio_id));
    }

    #[test]
    fn test_stop_playing() {
        let mut playback = StationPlaybackComponent::new();
        let audio_id = AudioId::new();

        playback.start_playing(audio_id);
        assert!(playback.is_playing());

        playback.stop_playing();
        assert!(playback.is_idle());
        assert_eq!(playback.audio_id(), None);
    }

    #[test]
    fn test_state_transitions() {
        let mut playback = StationPlaybackComponent::new();
        let audio_id1 = AudioId::new();
        let audio_id2 = AudioId::new();

        assert_eq!(playback.state(), StationPlaybackState::Idle);

        playback.start_playing(audio_id1);
        assert_eq!(playback.state(), StationPlaybackState::Playing);
        assert_eq!(playback.audio_id(), Some(audio_id1));

        playback.start_playing(audio_id2);
        assert_eq!(playback.audio_id(), Some(audio_id2));

        playback.stop_playing();
        assert_eq!(playback.state(), StationPlaybackState::Idle);
    }
}
