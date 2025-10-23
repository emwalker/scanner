use std::time::Duration;

/// Playback lifetime constraint for audio requests
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaybackDuration {
    /// Audio stops after this duration (for discovery playback)
    Limited(Duration),
    /// Audio continues until explicitly stopped (for listening)
    Indefinite,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limited_duration_creation() {
        let duration = PlaybackDuration::Limited(std::time::Duration::from_secs(5));
        assert_eq!(
            duration,
            PlaybackDuration::Limited(std::time::Duration::from_secs(5))
        );
    }

    #[test]
    fn test_indefinite_duration() {
        let duration = PlaybackDuration::Indefinite;
        assert_eq!(duration, PlaybackDuration::Indefinite);
    }
}
