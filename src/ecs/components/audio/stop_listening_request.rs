//! Stop listening request component - marks audio for stopping

use std::time::Instant;

/// Component marking audio playback as requested for stopping
///
/// When user requests to stop listening (e.g., presses Backspace in TUI),
/// this component is added to the AudioEntity. AudioCoordinationSystem queries
/// for entities with this component, stops playback, and clears the request.
#[derive(Debug, Clone)]
pub struct StopListeningRequestComponent {
    pub requested_at: Instant,
}

impl StopListeningRequestComponent {
    pub fn new() -> Self {
        Self {
            requested_at: Instant::now(),
        }
    }
}

impl Default for StopListeningRequestComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_listening_request_creation() {
        let request = StopListeningRequestComponent::new();
        assert!(request.requested_at.elapsed().as_millis() < 10);
    }

    #[test]
    fn test_stop_listening_request_default() {
        let request = StopListeningRequestComponent::default();
        assert!(request.requested_at.elapsed().as_millis() < 10);
    }
}
