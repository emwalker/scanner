//! Pause request component - marks a scan for pausing

use std::time::Instant;

/// Component marking a scan as requested for pausing
///
/// When user requests to pause (e.g., presses Enter in TUI to tune to a station),
/// this component is added to the ScanEntity. RequestProcessorSystem queries for
/// entities with this component, pauses the scan, optionally tunes to a station,
/// and clears the request.
#[derive(Debug, Clone)]
pub struct PauseRequestComponent {
    pub requested_at: Instant,
    pub window_num: usize,
    pub station_frequency_hz: Option<f64>,
    pub window_center_frequency_hz: Option<f64>,
}

impl PauseRequestComponent {
    pub fn new(window_num: usize) -> Self {
        Self {
            requested_at: Instant::now(),
            window_num,
            station_frequency_hz: None,
            window_center_frequency_hz: None,
        }
    }

    pub fn with_station(
        window_num: usize,
        station_frequency_hz: f64,
        window_center_frequency_hz: f64,
    ) -> Self {
        Self {
            requested_at: Instant::now(),
            window_num,
            station_frequency_hz: Some(station_frequency_hz),
            window_center_frequency_hz: Some(window_center_frequency_hz),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_request_creation() {
        let request = PauseRequestComponent::new(5);
        assert_eq!(request.window_num, 5);
        assert!(request.requested_at.elapsed().as_millis() < 10);
    }
}
