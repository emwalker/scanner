//! Scan-related components

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SCAN_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a scan
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScanId(u64);

impl ScanId {
    /// Generate a new unique scan ID
    pub fn new() -> Self {
        Self(NEXT_SCAN_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Create ScanId from TaskId
    pub fn from_task_id(task_id: &crate::ecs::TaskId) -> Self {
        let id_str = &task_id.0;
        if let Some(scan_num) = id_str.strip_prefix("scan_")
            && let Ok(num) = scan_num.parse::<u64>()
        {
            return Self(num);
        }
        Self(0)
    }
}

impl Default for ScanId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of scan being performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanType {
    /// Scan an entire frequency band
    Band,
    /// Scan specific station frequencies
    Stations,
}

mod config;
mod lifecycle;
mod pause_request;
mod pending_request;
mod progress;
mod results;
mod resume_request;
mod tuner;
mod window_allocation;
mod window_worker;

pub use config::ScanConfigComponent;
pub use lifecycle::ScanLifecycleComponent;
pub use pause_request::PauseRequestComponent;
pub use pending_request::PendingScanRequest;
pub use progress::{PreviousPauseState, ScanPauseState, ScanProgressComponent};
pub use results::ScanResultsComponent;
pub use resume_request::ResumeRequestComponent;
pub use tuner::ScanTunerComponent;
pub use window_allocation::WindowAllocationRequest;
pub use window_worker::{
    SignalData, WindowWorkerComponent, WindowWorkerOutcome, WindowWorkerResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::WindowId;

    #[test]
    fn test_scan_id_generation() {
        let id1 = ScanId::new();
        let id2 = ScanId::new();
        assert_ne!(id1, id2);
        assert!(id2.value() > id1.value());
    }

    #[test]
    fn test_scan_id_clone() {
        let id = ScanId::new();
        let cloned = id;
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_scan_config_bandwidth() {
        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 200e3, 2.4e6, 40.0, 1.0, 3);
        assert_eq!(config.bandwidth(), 20.0e6);
    }

    #[test]
    fn test_scan_config_total_windows() {
        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 98.0e6, 1.0e6, 2.4e6, 40.0, 1.0, 3);
        // Range 88-98 MHz with 1 MHz steps: 11 windows (88, 89, ..., 98)
        assert_eq!(config.total_windows(), 11);
    }

    #[test]
    fn test_scan_progress_initialization() {
        let progress = ScanProgressComponent::new(10);
        assert_eq!(progress.total_windows, 10);
        assert_eq!(progress.current_window, None);
        assert_eq!(progress.windows_completed, 0);
        assert!(progress.is_pending());
        assert!(!progress.is_scanning());
        assert!(!progress.is_paused());
        assert!(!progress.is_completed());
    }

    #[test]
    fn test_scan_progress_window_operations() {
        let mut progress = ScanProgressComponent::new(10);
        let task_id = crate::ecs::TaskId::new("test".to_string());

        let window_id = WindowId::new(task_id, 0);
        progress.start_window(window_id);
        assert_eq!(
            progress.current_window.as_ref().map(|w| w.window_index),
            Some(0)
        );
        assert!(progress.is_scanning());

        progress.complete_window();
        assert_eq!(progress.windows_completed, 1);
    }

    #[test]
    fn test_scan_progress_pause_resume() {
        let mut progress = ScanProgressComponent::new(10);
        let task_id = crate::ecs::TaskId::new("test".to_string());
        let window_id = WindowId::new(task_id, 5);

        progress.pause(window_id.clone());
        assert!(progress.is_paused());
        assert!(!progress.is_scanning());
        assert!(matches!(
            &progress.state,
            ScanPauseState::PausedAtWindow { window_id: w } if w == &window_id
        ));

        progress.resume();
        assert!(progress.is_scanning());
        assert!(!progress.is_paused());
    }

    #[test]
    fn test_scan_progress_listening() {
        let mut progress = ScanProgressComponent::new(10);
        let task_id = crate::ecs::TaskId::new("test".to_string());
        let window_id = WindowId::new(task_id, 3);

        progress.start_listening(window_id.clone());
        assert!(progress.is_listening());
        assert!(progress.is_paused());
        assert!(!progress.is_scanning());

        progress.stop_listening(window_id);
        assert!(!progress.is_listening());
        assert!(progress.is_paused());
    }

    #[test]
    fn test_scan_progress_completion() {
        let mut progress = ScanProgressComponent::new(10);

        progress.mark_complete();
        assert!(progress.is_completed());
        assert!(!progress.is_scanning());
    }

    #[test]
    fn test_scan_progress_percentage() {
        let mut progress = ScanProgressComponent::new(10);
        assert_eq!(progress.progress_percentage(), 0.0);

        progress.windows_completed = 5;
        assert_eq!(progress.progress_percentage(), 0.5);

        progress.windows_completed = 10;
        assert_eq!(progress.progress_percentage(), 1.0);
    }

    #[test]
    fn test_scan_progress_percentage_zero_windows() {
        let progress = ScanProgressComponent::new(0);
        assert_eq!(progress.progress_percentage(), 0.0);
    }

    #[test]
    fn test_scan_results_initialization() {
        let results = ScanResultsComponent::new();
        assert_eq!(results.signals_found, 0);
        assert_eq!(results.signals_rejected, 0);
        assert_eq!(results.stations_discovered, 0);
        assert_eq!(results.total_signals(), 0);
    }

    #[test]
    fn test_scan_results_add_signal() {
        let mut results = ScanResultsComponent::new();
        results.add_signal();
        assert_eq!(results.signals_found, 1);
        assert_eq!(results.total_signals(), 1);
    }

    #[test]
    fn test_scan_results_reject_signal() {
        let mut results = ScanResultsComponent::new();
        results.reject_signal();
        assert_eq!(results.signals_rejected, 1);
        assert_eq!(results.total_signals(), 1);
    }

    #[test]
    fn test_scan_results_add_station() {
        let mut results = ScanResultsComponent::new();
        results.add_station();
        assert_eq!(results.stations_discovered, 1);
    }

    #[test]
    fn test_scan_results_total_signals() {
        let mut results = ScanResultsComponent::new();
        results.add_signal();
        results.add_signal();
        results.reject_signal();
        assert_eq!(results.total_signals(), 3);
    }

    #[test]
    fn test_scan_lifecycle_initialization() {
        let lifecycle = ScanLifecycleComponent::new();
        assert!(!lifecycle.is_started());
        assert!(!lifecycle.is_completed());
        assert_eq!(lifecycle.pause_count(), 0);
        assert!(lifecycle.duration().is_none());
    }

    #[test]
    fn test_scan_lifecycle_start() {
        let mut lifecycle = ScanLifecycleComponent::new();
        lifecycle.start();
        assert!(lifecycle.is_started());
        assert!(lifecycle.duration().is_some());
    }

    #[test]
    fn test_scan_lifecycle_start_idempotent() {
        let mut lifecycle = ScanLifecycleComponent::new();
        lifecycle.start();
        let first_start = lifecycle.started_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        lifecycle.start();
        assert_eq!(lifecycle.started_at, first_start);
    }

    #[test]
    fn test_scan_lifecycle_complete() {
        let mut lifecycle = ScanLifecycleComponent::new();
        lifecycle.start();
        lifecycle.complete();
        assert!(lifecycle.is_completed());
    }

    #[test]
    fn test_scan_lifecycle_complete_idempotent() {
        let mut lifecycle = ScanLifecycleComponent::new();
        lifecycle.start();
        lifecycle.complete();
        let first_complete = lifecycle.completed_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        lifecycle.complete();
        assert_eq!(lifecycle.completed_at, first_complete);
    }

    #[test]
    fn test_scan_lifecycle_pause_history() {
        let mut lifecycle = ScanLifecycleComponent::new();
        assert_eq!(lifecycle.pause_count(), 0);

        lifecycle.pause();
        assert_eq!(lifecycle.pause_count(), 1);

        lifecycle.pause();
        assert_eq!(lifecycle.pause_count(), 2);
    }

    #[test]
    fn test_scan_lifecycle_duration() {
        let mut lifecycle = ScanLifecycleComponent::new();
        assert!(lifecycle.duration().is_none());

        lifecycle.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = lifecycle.duration().unwrap();
        assert!(duration.as_millis() >= 10);
    }
}
