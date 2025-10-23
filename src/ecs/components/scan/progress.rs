//! Scan progress component

use std::collections::HashSet;

use crate::ecs::components::WindowId;

/// Pause state for a scan
#[derive(Debug, Clone, PartialEq)]
pub enum ScanPauseState {
    /// Scan has been requested but not yet started
    Pending,
    /// Scan is actively running
    Scanning,
    /// Scan is paused at a specific window
    PausedAtWindow { window_id: WindowId },
    /// Globally paused by user (spacebar)
    PausedGlobally {
        window_id: WindowId,
        previous_state: PreviousPauseState,
    },
    /// Scan has completed all windows
    Completed,
    /// User is listening to a station (paused for audio)
    Listening { window_id: WindowId },
    /// No tuners available at creation time
    WaitingForTuner,
    /// Assigned tuner disappeared from pool
    TunerOffline,
}

/// Captures what was happening before global pause for resume
#[derive(Debug, Clone, PartialEq)]
pub enum PreviousPauseState {
    /// Was actively scanning
    WasScanning,
    /// Was listening to a station
    WasListening {
        window_id: WindowId,
        station_frequency_hz: f64,
    },
}

/// Component tracking scan progress
#[derive(Debug, Clone)]
pub struct ScanProgressComponent {
    /// Current pause state
    pub state: ScanPauseState,

    /// Current window being processed (when Scanning)
    pub current_window: Option<WindowId>,

    /// Total number of windows to process
    pub total_windows: usize,

    /// Number of windows completed
    pub windows_completed: usize,

    /// Set of completed window IDs (for non-sequential processing)
    pub completed_windows: HashSet<WindowId>,
}

impl ScanProgressComponent {
    /// Create a new progress component in Pending state
    pub fn new(total_windows: usize) -> Self {
        Self {
            state: ScanPauseState::Pending,
            current_window: None,
            total_windows,
            windows_completed: 0,
            completed_windows: HashSet::new(),
        }
    }

    /// Start processing a window
    pub fn start_window(&mut self, window_id: WindowId) {
        self.current_window = Some(window_id);
        self.state = ScanPauseState::Scanning;
    }

    /// Complete a window
    pub fn complete_window(&mut self) {
        self.windows_completed += 1;
    }

    /// Mark a specific window as completed
    pub fn complete_window_at(&mut self, window_id: WindowId) {
        if self.completed_windows.insert(window_id) {
            self.windows_completed += 1;
        }
    }

    /// Check if a window has been completed
    pub fn is_window_completed(&self, window_id: &WindowId) -> bool {
        self.completed_windows.contains(window_id)
    }

    /// Pause at a specific window
    pub fn pause(&mut self, window_id: WindowId) {
        self.state = ScanPauseState::PausedAtWindow { window_id };
    }

    /// Resume scanning
    pub fn resume(&mut self) {
        self.state = ScanPauseState::Scanning;
    }

    /// Mark scan as completed
    pub fn mark_complete(&mut self) {
        self.state = ScanPauseState::Completed;
    }

    /// Enter listening mode
    pub fn start_listening(&mut self, window_id: WindowId) {
        self.state = ScanPauseState::Listening { window_id };
    }

    /// Exit listening mode
    pub fn stop_listening(&mut self, window_id: WindowId) {
        self.state = ScanPauseState::PausedAtWindow { window_id };
    }

    /// Check if scan is pending
    pub fn is_pending(&self) -> bool {
        matches!(self.state, ScanPauseState::Pending)
    }

    /// Check if scan is paused
    pub fn is_paused(&self) -> bool {
        matches!(
            self.state,
            ScanPauseState::PausedAtWindow { .. }
                | ScanPauseState::PausedGlobally { .. }
                | ScanPauseState::Listening { .. }
        )
    }

    /// Check if scan is actively scanning
    pub fn is_scanning(&self) -> bool {
        matches!(self.state, ScanPauseState::Scanning)
    }

    /// Check if scan is completed
    pub fn is_completed(&self) -> bool {
        matches!(self.state, ScanPauseState::Completed)
    }

    /// Check if currently listening
    pub fn is_listening(&self) -> bool {
        matches!(self.state, ScanPauseState::Listening { .. })
    }

    /// Calculate progress percentage (0.0 to 1.0)
    pub fn progress_percentage(&self) -> f64 {
        if self.total_windows == 0 {
            return 0.0;
        }
        self.windows_completed as f64 / self.total_windows as f64
    }

    /// Pause globally (user-initiated via spacebar)
    pub fn pause_globally(&mut self, window_id: WindowId, previous_state: PreviousPauseState) {
        self.state = ScanPauseState::PausedGlobally {
            window_id,
            previous_state,
        };
    }

    /// Resume from global pause, restoring previous state
    pub fn resume_from_global_pause(&mut self) {
        if let ScanPauseState::PausedGlobally { previous_state, .. } = &self.state.clone() {
            match previous_state {
                PreviousPauseState::WasScanning => {
                    self.state = ScanPauseState::Scanning;
                }
                PreviousPauseState::WasListening { window_id, .. } => {
                    self.state = ScanPauseState::Listening {
                        window_id: window_id.clone(),
                    };
                }
            }
        }
    }

    /// Check if globally paused
    pub fn is_globally_paused(&self) -> bool {
        matches!(self.state, ScanPauseState::PausedGlobally { .. })
    }

    /// Create a new progress component in WaitingForTuner state
    pub fn new_waiting_for_tuner(total_windows: usize) -> Self {
        Self {
            state: ScanPauseState::WaitingForTuner,
            current_window: None,
            total_windows,
            windows_completed: 0,
            completed_windows: HashSet::new(),
        }
    }

    /// Set state to TunerOffline
    pub fn set_tuner_offline(&mut self) {
        self.state = ScanPauseState::TunerOffline;
    }

    /// Check if scan is blocked by tuner issues
    pub fn is_blocked_by_tuner(&self) -> bool {
        matches!(
            self.state,
            ScanPauseState::WaitingForTuner | ScanPauseState::TunerOffline
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::TaskId;

    #[test]
    fn test_pause_globally_from_scanning() {
        let mut progress = ScanProgressComponent::new(5);
        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id, 2);
        progress.start_window(window_id.clone());
        assert!(progress.is_scanning());

        progress.pause_globally(window_id, PreviousPauseState::WasScanning);
        assert!(matches!(
            progress.state,
            ScanPauseState::PausedGlobally { .. }
        ));
        assert!(!progress.is_scanning());
    }

    #[test]
    fn test_pause_globally_from_listening() {
        let mut progress = ScanProgressComponent::new(5);
        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id, 3);
        progress.start_listening(window_id.clone());
        assert!(progress.is_listening());

        progress.pause_globally(window_id, PreviousPauseState::WasListening {
            window_id: WindowId::new(TaskId::new("test_scan".to_string()), 3),
            station_frequency_hz: 88.9e6,
        });
        assert!(matches!(
            progress.state,
            ScanPauseState::PausedGlobally { .. }
        ));
        assert!(!progress.is_listening());
    }

    #[test]
    fn test_resume_from_globally_paused_to_scanning() {
        let mut progress = ScanProgressComponent::new(5);
        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id, 2);
        progress.pause_globally(window_id, PreviousPauseState::WasScanning);

        progress.resume_from_global_pause();
        assert!(progress.is_scanning());
    }

    #[test]
    fn test_resume_from_globally_paused_to_listening() {
        let mut progress = ScanProgressComponent::new(5);
        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id.clone(), 3);
        progress.pause_globally(window_id.clone(), PreviousPauseState::WasListening {
            window_id,
            station_frequency_hz: 88.9e6,
        });

        progress.resume_from_global_pause();
        assert!(progress.is_listening());
        if let ScanPauseState::Listening {
            window_id: resumed_window_id,
        } = &progress.state
        {
            assert_eq!(resumed_window_id.window_index, 3);
        } else {
            panic!("Expected Listening state");
        }
    }

    #[test]
    fn test_is_globally_paused() {
        let mut progress = ScanProgressComponent::new(5);
        assert!(!progress.is_globally_paused());

        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id, 2);
        progress.pause_globally(window_id, PreviousPauseState::WasScanning);
        assert!(progress.is_globally_paused());

        progress.resume_from_global_pause();
        assert!(!progress.is_globally_paused());
    }

    #[test]
    fn test_waiting_for_tuner_state() {
        let component = ScanProgressComponent::new_waiting_for_tuner(10);

        assert_eq!(component.state, ScanPauseState::WaitingForTuner);
        assert_eq!(component.total_windows, 10);
    }

    #[test]
    fn test_transition_to_tuner_offline() {
        let mut component = ScanProgressComponent::new(10);
        let task_id = TaskId::new("test_scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        component.start_window(window_id);

        component.set_tuner_offline();

        assert_eq!(component.state, ScanPauseState::TunerOffline);
    }

    #[test]
    fn test_is_blocked_by_tuner() {
        let component_waiting = ScanProgressComponent::new_waiting_for_tuner(10);
        let mut component_offline = ScanProgressComponent::new(10);
        component_offline.set_tuner_offline();
        let component_scanning = ScanProgressComponent::new(10);

        assert!(component_waiting.is_blocked_by_tuner());
        assert!(component_offline.is_blocked_by_tuner());
        assert!(!component_scanning.is_blocked_by_tuner());
    }
}
