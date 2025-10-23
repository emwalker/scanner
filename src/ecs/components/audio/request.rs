use super::duration::PlaybackDuration;
use crate::core::types::Signal;

/// Request to play audio via the ECS queue
#[derive(Debug, Clone)]
pub enum AudioRequest {
    /// Play signal discovered during scanning (auto-stops after duration)
    StationDiscovery {
        signal: Signal,
        duration: PlaybackDuration,
    },
    /// Play signal via ENTER listening mode (indefinite duration)
    Listening { signal: Signal },
}

impl AudioRequest {
    pub fn frequency(&self) -> f64 {
        match self {
            AudioRequest::StationDiscovery { signal, .. } => signal.frequency_hz,
            AudioRequest::Listening { signal } => signal.frequency_hz,
        }
    }

    pub fn signal(&self) -> &Signal {
        match self {
            AudioRequest::StationDiscovery { signal, .. } => signal,
            AudioRequest::Listening { signal } => signal,
        }
    }

    pub fn duration(&self) -> PlaybackDuration {
        match self {
            AudioRequest::StationDiscovery { duration, .. } => duration.clone(),
            AudioRequest::Listening { .. } => PlaybackDuration::Indefinite,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;
    use crate::{audio::quality::AudioQuality, core::signals::ModulationType};

    fn test_signal() -> Signal {
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
    fn test_station_discovery_request() {
        let signal = test_signal();
        let req = AudioRequest::StationDiscovery {
            signal: signal.clone(),
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        };
        assert_eq!(req.frequency(), 88.9e6);
    }

    #[test]
    fn test_listening_request() {
        let signal = test_signal();
        let req = AudioRequest::Listening {
            signal: signal.clone(),
        };
        assert_eq!(req.frequency(), 88.9e6);
        assert_eq!(req.duration(), PlaybackDuration::Indefinite);
    }
}
