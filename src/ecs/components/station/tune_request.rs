//! Tune request component - marks a station for audio playback

/// Component marking a station as requested for tuning/playback
///
/// When user selects a station (e.g., presses Enter in TUI), this component
/// is added to the StationEntity. AudioCoordinationSystem queries for entities
/// with this component, creates AudioEntity, and clears the request.
#[derive(Debug, Clone)]
pub struct TuneRequestComponent {
    /// Additional context for tuning (e.g., window metadata)
    pub window_id: usize,
    pub center_frequency: f64,
}

impl TuneRequestComponent {
    pub fn new(window_id: usize, center_frequency: f64) -> Self {
        Self {
            window_id,
            center_frequency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_request_creation() {
        let request = TuneRequestComponent::new(5, 88.9e6);
        assert_eq!(request.window_id, 5);
        assert_eq!(request.center_frequency, 88.9e6);
    }
}
