use super::request::AudioRequest;

/// Component managing the audio playback queue
///
/// Maintains a FIFO queue of audio requests. Discovery adds
/// StationDiscovery requests, ENTER adds Listening request.
/// System processes queue in order.
#[derive(Debug, Clone)]
pub struct AudioQueueComponent {
    requests: Vec<AudioRequest>,
}

impl AudioQueueComponent {
    pub fn new() -> Self {
        AudioQueueComponent {
            requests: Vec::new(),
        }
    }

    /// Add request to end of queue
    pub fn enqueue(&mut self, request: AudioRequest) {
        self.requests.push(request);
    }

    /// Remove and return first request
    pub fn dequeue(&mut self) -> Option<AudioRequest> {
        if self.requests.is_empty() {
            None
        } else {
            Some(self.requests.remove(0))
        }
    }

    /// Peek at first request without removing
    pub fn peek(&self) -> Option<&AudioRequest> {
        self.requests.first()
    }

    /// Remove all requests
    pub fn clear(&mut self) {
        self.requests.clear();
    }

    /// Get all requests (immutable)
    pub fn requests(&self) -> &[AudioRequest] {
        &self.requests
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.requests.len()
    }
}

impl Default for AudioQueueComponent {
    fn default() -> Self {
        Self::new()
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
    fn test_new_queue_is_empty() {
        let queue = AudioQueueComponent::new();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_enqueue_and_peek() {
        let mut queue = AudioQueueComponent::new();
        let req = AudioRequest::StationDiscovery {
            signal: test_signal(),
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        };
        queue.enqueue(req.clone());
        assert!(!queue.is_empty());
        assert_eq!(queue.peek().map(|r| r.frequency()), Some(88.9e6));
    }

    #[test]
    fn test_dequeue_fifo_order() {
        let mut queue = AudioQueueComponent::new();
        let req1 = AudioRequest::StationDiscovery {
            signal: Signal {
                frequency_hz: 88.9e6,
                ..test_signal()
            },
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        };
        let req2 = AudioRequest::Listening {
            signal: Signal {
                frequency_hz: 89.1e6,
                ..test_signal()
            },
        };
        queue.enqueue(req1);
        queue.enqueue(req2);

        let first = queue.dequeue();
        assert_eq!(first.map(|r| r.frequency()), Some(88.9e6));
    }

    #[test]
    fn test_clear_empties_queue() {
        let mut queue = AudioQueueComponent::new();
        queue.enqueue(AudioRequest::StationDiscovery {
            signal: test_signal(),
            duration: PlaybackDuration::Limited(std::time::Duration::from_secs(5)),
        });
        queue.clear();
        assert!(queue.is_empty());
    }
}
