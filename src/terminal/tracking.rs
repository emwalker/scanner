//! Progress tracking and percentage calculation for window processing

use std::time::{Duration, Instant};

/// Tracks progress through a scanning window with completion percentage and time estimates
#[derive(Debug, Clone)]
pub struct WindowProgress {
    total_peaks: usize,
    processed_peaks: usize,
    start_time: Instant,
    window_id: usize,
}

impl WindowProgress {
    /// Create new progress tracker for a window
    pub fn new(window_id: usize, total_peaks: usize) -> Self {
        Self {
            total_peaks,
            processed_peaks: 0,
            start_time: Instant::now(),
            window_id,
        }
    }

    /// Mark one more peak as processed
    pub fn increment(&mut self) {
        self.processed_peaks += 1;
    }

    /// Get completion percentage (0.0 to 1.0)
    pub fn completion_percentage(&self) -> f64 {
        if self.total_peaks == 0 {
            return 1.0; // Consider empty window as complete
        }
        self.processed_peaks as f64 / self.total_peaks as f64
    }

    /// Get estimated time remaining
    pub fn estimated_time_remaining(&self) -> Option<Duration> {
        if self.processed_peaks == 0 {
            return None; // Can't estimate without any completed work
        }

        let elapsed = self.start_time.elapsed();
        let avg_time_per_peak = elapsed / self.processed_peaks as u32;
        let remaining_peaks = self.total_peaks.saturating_sub(self.processed_peaks);

        Some(avg_time_per_peak * remaining_peaks as u32)
    }

    /// Check if processing is complete
    pub fn is_complete(&self) -> bool {
        self.processed_peaks >= self.total_peaks
    }

    /// Get window ID
    pub fn window_id(&self) -> usize {
        self.window_id
    }

    /// Get current progress info as (processed, total)
    pub fn current_progress(&self) -> (usize, usize) {
        (self.processed_peaks, self.total_peaks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_progress_percentage_calculation() {
        let mut progress = WindowProgress::new(1, 10);

        // Initially 0%
        assert_eq!(progress.completion_percentage(), 0.0);
        assert!(!progress.is_complete());
        assert_eq!(progress.current_progress(), (0, 10));

        // Process some peaks
        progress.increment();
        progress.increment();
        progress.increment();

        // Should be 30%
        assert_eq!(progress.completion_percentage(), 0.3);
        assert!(!progress.is_complete());
        assert_eq!(progress.current_progress(), (3, 10));

        // Complete all peaks
        for _ in 0..7 {
            progress.increment();
        }

        // Should be 100%
        assert_eq!(progress.completion_percentage(), 1.0);
        assert!(progress.is_complete());
        assert_eq!(progress.current_progress(), (10, 10));
    }

    #[test]
    fn test_empty_window_progress() {
        let progress = WindowProgress::new(2, 0);

        // Empty window should be considered complete
        assert_eq!(progress.completion_percentage(), 1.0);
        assert!(progress.is_complete());
        assert_eq!(progress.current_progress(), (0, 0));
        assert_eq!(progress.window_id(), 2);
    }

    #[test]
    fn test_time_estimation() {
        let mut progress = WindowProgress::new(3, 5);

        // No time estimate before processing any peaks
        assert!(progress.estimated_time_remaining().is_none());

        // Process first peak with small delay
        thread::sleep(Duration::from_millis(10));
        progress.increment();

        // Should have time estimate now
        let estimate = progress.estimated_time_remaining();
        assert!(estimate.is_some());

        // Estimate should be reasonable (4 remaining peaks * ~10ms each)
        let estimate_ms = estimate.unwrap().as_millis();
        assert!(
            (20..=100).contains(&estimate_ms),
            "Estimate should be reasonable, got {}ms",
            estimate_ms
        );
    }

    #[test]
    fn test_progress_tracking_edge_cases() {
        let mut progress = WindowProgress::new(4, 3);

        // Process more peaks than expected (shouldn't panic)
        for _ in 0..5 {
            progress.increment();
        }

        // Completion percentage should max out at 100%
        assert!(progress.completion_percentage() >= 1.0);
        assert!(progress.is_complete());

        // Time estimate should be zero when done
        if let Some(remaining) = progress.estimated_time_remaining() {
            assert_eq!(remaining, Duration::from_secs(0));
        }
    }
}
