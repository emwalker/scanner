//! Tune request component - marks a station for audio playback

/// Component marking a station as requested for tuning/playback
///
/// When user selects a station (e.g., presses Enter in TUI), this component
/// is added to the StationEntity. The AudioPlaybackSystem processes tune requests
/// deterministically by checking if the scan's window tasks have released their tuners.
use crate::ecs::components::window::WindowId;

#[derive(Debug, Clone)]
pub struct TuneRequestComponent {
    /// Window associated with the tune request
    pub window_id: WindowId,
}

impl TuneRequestComponent {
    pub fn new(window_id: WindowId) -> Self {
        Self { window_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{TaskId, components::window::WindowId};

    #[test]
    fn test_tune_request_creation() {
        let window_id = WindowId::new(TaskId::new("test"), 5);
        let request = TuneRequestComponent::new(window_id.clone());
        assert_eq!(request.window_id, window_id);
    }
}
