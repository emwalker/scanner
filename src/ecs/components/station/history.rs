//! Station history component

use std::time::{Duration, Instant};

/// Component tracking station play history
#[derive(Debug, Clone)]
pub struct StationHistoryComponent {
    /// Last time this station was heard (played)
    pub last_heard: Option<Instant>,

    /// Total number of times this station was played
    pub play_count: usize,

    /// Total duration of all plays
    pub total_play_duration: Duration,

    /// When current play started (if currently playing)
    current_play_start: Option<Instant>,
}

impl StationHistoryComponent {
    /// Create a new history component
    pub fn new() -> Self {
        Self {
            last_heard: None,
            play_count: 0,
            total_play_duration: Duration::ZERO,
            current_play_start: None,
        }
    }

    /// Record that playback started
    pub fn record_play_start(&mut self) {
        self.current_play_start = Some(Instant::now());
        self.play_count += 1;
        self.last_heard = Some(Instant::now());
    }

    /// Record that playback ended
    pub fn record_play_end(&mut self) {
        if let Some(start) = self.current_play_start.take() {
            let duration = start.elapsed();
            self.total_play_duration += duration;
        }
    }

    /// Update last heard time without starting playback
    pub fn update_last_heard(&mut self) {
        self.last_heard = Some(Instant::now());
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.current_play_start.is_some()
    }

    /// Get current play duration (if playing)
    pub fn current_play_duration(&self) -> Option<Duration> {
        self.current_play_start.map(|start| start.elapsed())
    }
}

impl Default for StationHistoryComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn test_create_history() {
        let history = StationHistoryComponent::new();

        assert_eq!(history.last_heard, None);
        assert_eq!(history.play_count, 0);
        assert_eq!(history.total_play_duration, Duration::ZERO);
        assert!(!history.is_playing());
    }

    #[test]
    fn test_record_play_start() {
        let mut history = StationHistoryComponent::new();
        history.record_play_start();

        assert_eq!(history.play_count, 1);
        assert!(history.last_heard.is_some());
        assert!(history.is_playing());
    }

    #[test]
    fn test_record_play_end() {
        let mut history = StationHistoryComponent::new();
        history.record_play_start();

        thread::sleep(Duration::from_millis(10));
        history.record_play_end();

        assert!(!history.is_playing());
        assert!(history.total_play_duration.as_millis() >= 10);
    }

    #[test]
    fn test_multiple_plays() {
        let mut history = StationHistoryComponent::new();

        history.record_play_start();
        thread::sleep(Duration::from_millis(5));
        history.record_play_end();

        history.record_play_start();
        thread::sleep(Duration::from_millis(5));
        history.record_play_end();

        assert_eq!(history.play_count, 2);
        assert!(history.total_play_duration.as_millis() >= 10);
    }

    #[test]
    fn test_current_play_duration() {
        let mut history = StationHistoryComponent::new();
        history.record_play_start();

        thread::sleep(Duration::from_millis(10));

        let duration = history.current_play_duration();
        assert!(duration.is_some());
        assert!(duration.unwrap().as_millis() >= 10);
    }

    #[test]
    fn test_update_last_heard() {
        let mut history = StationHistoryComponent::new();
        history.update_last_heard();

        assert!(history.last_heard.is_some());
        assert_eq!(history.play_count, 0);
        assert!(!history.is_playing());
    }
}
