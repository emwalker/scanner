//! Resume request component - marks a scan for resuming

use std::time::Instant;

/// Component marking a scan as requested for resuming
///
/// When user requests to resume scanning (e.g., presses Space in TUI while paused),
/// this component is added to the ScanEntity. ScanCoordinationSystem queries for
/// entities with this component, resumes the scan, and clears the request.
#[derive(Debug, Clone)]
pub struct ResumeRequestComponent {
    pub requested_at: Instant,
    pub window_num: usize,
}

impl ResumeRequestComponent {
    pub fn new(window_num: usize) -> Self {
        Self {
            requested_at: Instant::now(),
            window_num,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resume_request_creation() {
        let request = ResumeRequestComponent::new(5);
        assert_eq!(request.window_num, 5);
        assert!(request.requested_at.elapsed().as_millis() < 10);
    }
}
