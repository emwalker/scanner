use super::request::AudioRequest;

/// Component holding a pending audio request for an entity
///
/// Used by discovery and listening paths to stage requests
/// that need to be processed by AudioPlaybackSystem.
#[derive(Debug, Clone)]
pub struct AudioRequestComponent {
    request: Option<AudioRequest>,
}

impl AudioRequestComponent {
    /// Create component with a pending request
    pub fn pending(request: AudioRequest) -> Self {
        AudioRequestComponent {
            request: Some(request),
        }
    }

    /// Create component with no request
    pub fn empty() -> Self {
        AudioRequestComponent { request: None }
    }

    /// Check if component has a pending request
    pub fn is_pending(&self) -> bool {
        self.request.is_some()
    }

    /// Get the request (if any)
    pub fn request(&self) -> Option<&AudioRequest> {
        self.request.as_ref()
    }

    /// Take ownership of the request
    pub fn take(&mut self) -> Option<AudioRequest> {
        self.request.take()
    }

    /// Set a new request, replacing any existing one
    pub fn set(&mut self, request: AudioRequest) {
        self.request = Some(request);
    }

    /// Clear the request
    pub fn clear(&mut self) {
        self.request = None;
    }
}

impl Default for AudioRequestComponent {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::{signals::ModulationType, types::Signal},
        ecs::components::audio::PlaybackDuration,
    };

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
    fn test_pending_request_lifecycle() {
        let req = AudioRequest::StationDiscovery {
            signal: test_signal(),
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        };
        let component = AudioRequestComponent::pending(req.clone());
        assert!(component.is_pending());
        assert_eq!(component.request().unwrap().frequency(), 88.9e6);
    }

    #[test]
    fn test_clear_request() {
        let req = AudioRequest::StationDiscovery {
            signal: test_signal(),
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        };
        let mut component = AudioRequestComponent::pending(req);
        component.clear();
        assert!(!component.is_pending());
    }
}
