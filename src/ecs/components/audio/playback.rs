//! Audio playback component

use std::time::{Duration, Instant};

/// Component tracking playback state
#[derive(Debug)]
pub struct AudioPlaybackComponent {
    /// When playback started
    started_at: Instant,

    /// Whether currently playing
    is_playing: bool,
}

impl AudioPlaybackComponent {
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            is_playing: true,
        }
    }

    pub fn started_at(&self) -> Instant {
        self.started_at
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn play_duration(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn stop(&mut self) {
        self.is_playing = false;
    }

    #[cfg(test)]
    pub fn set_started_at_for_test(&mut self, started_at: Instant) {
        self.started_at = started_at;
    }
}

impl Default for AudioPlaybackComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn test_create_playback() {
        let playback = AudioPlaybackComponent::new();

        assert!(playback.is_playing());
        assert!(playback.play_duration().as_millis() < 10);
    }

    #[test]
    fn test_stop_playback() {
        let mut playback = AudioPlaybackComponent::new();
        playback.stop();

        assert!(!playback.is_playing());
    }

    #[test]
    fn test_play_duration() {
        let playback = AudioPlaybackComponent::new();
        thread::sleep(Duration::from_millis(10));

        let duration = playback.play_duration();
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_started_at() {
        let playback = AudioPlaybackComponent::new();
        let started = playback.started_at();

        assert!(started.elapsed().as_millis() < 10);
    }
}
