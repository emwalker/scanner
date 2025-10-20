//! Tune transition component for tracking multi-step playback transitions

use std::time::{Duration, Instant};

/// Component that tracks the state of a tune transition
///
/// When a user selects a station during scanning, the transition involves:
/// 1. Pausing the scan and releasing the tuner
/// 2. Acquiring resources (tuner segment) for playback
/// 3. Spawning the audio graph
/// 4. Waiting for playback to start
///
/// This component tracks progress through these stages and handles retries/timeouts.
#[derive(Debug, Clone)]
pub struct TuneTransitionComponent {
    pub stage: TuneStage,
    pub window_id: usize,
    pub center_frequency: f64,
    pub requested_at: Instant,
    pub retry_count: u8,
    pub max_retries: u8,
    pub last_retry: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneStage {
    /// Scan is pausing, tuner still allocated to scanning
    AwaitingTunerRelease,

    /// Tuner released, enqueuing request for AudioPlaybackSystem
    AcquiringResources,
}

impl TuneTransitionComponent {
    pub fn new(window_id: usize, center_frequency: f64) -> Self {
        Self {
            stage: TuneStage::AwaitingTunerRelease,
            window_id,
            center_frequency,
            requested_at: Instant::now(),
            retry_count: 0,
            max_retries: 10,
            last_retry: None,
        }
    }

    /// Check if transition has timed out
    pub fn should_timeout(&self) -> bool {
        self.requested_at.elapsed() > Duration::from_secs(10)
    }

    /// Check if should retry resource acquisition
    pub fn should_retry_resources(&self) -> bool {
        self.retry_count < self.max_retries
            && self
                .last_retry
                .map(|t| t.elapsed() > Duration::from_millis(100))
                .unwrap_or(true)
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
        self.last_retry = Some(Instant::now());
    }

    /// Get human-readable status message
    pub fn status_message(&self) -> &'static str {
        match self.stage {
            TuneStage::AwaitingTunerRelease => "Pausing scan...",
            TuneStage::AcquiringResources => "Queueing request...",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_transition() {
        let transition = TuneTransitionComponent::new(5, 88.9e6);

        assert_eq!(transition.stage, TuneStage::AwaitingTunerRelease);
        assert_eq!(transition.window_id, 5);
        assert_eq!(transition.center_frequency, 88.9e6);
        assert_eq!(transition.retry_count, 0);
        assert_eq!(transition.max_retries, 10);
        assert!(transition.last_retry.is_none());
    }

    #[test]
    fn test_timeout() {
        let mut transition = TuneTransitionComponent::new(5, 88.9e6);
        assert!(!transition.should_timeout());

        transition.requested_at = Instant::now() - Duration::from_secs(11);
        assert!(transition.should_timeout());
    }

    #[test]
    fn test_retry_logic() {
        let mut transition = TuneTransitionComponent::new(5, 88.9e6);

        assert!(transition.should_retry_resources());

        transition.increment_retry();
        assert_eq!(transition.retry_count, 1);
        assert!(transition.last_retry.is_some());

        for _ in 0..transition.max_retries {
            transition.increment_retry();
        }
        assert!(!transition.should_retry_resources());
    }

    #[test]
    fn test_status_messages() {
        let mut transition = TuneTransitionComponent::new(5, 88.9e6);

        assert_eq!(transition.status_message(), "Pausing scan...");

        transition.stage = TuneStage::AcquiringResources;
        assert_eq!(transition.status_message(), "Queueing request...");
    }
}
