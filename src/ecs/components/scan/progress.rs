//! Scan progress component

use std::collections::HashSet;

/// Pause state for a scan
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanPauseState {
    /// Scan has been requested but not yet started
    Pending,
    /// Scan is actively running
    Scanning,
    /// Scan is paused at a specific window
    PausedAtWindow { window_index: usize },
    /// Scan has completed all windows
    Completed,
    /// User is listening to a station (paused for audio)
    Listening { paused_at_window: usize },
}

/// Component tracking scan progress
#[derive(Debug, Clone)]
pub struct ScanProgressComponent {
    /// Current pause state
    pub state: ScanPauseState,

    /// Current window being processed (when Scanning)
    pub current_window: usize,

    /// Total number of windows to process
    pub total_windows: usize,

    /// Number of windows completed
    pub windows_completed: usize,

    /// Set of completed window indices (for non-sequential processing)
    pub completed_windows: HashSet<usize>,
}

impl ScanProgressComponent {
    /// Create a new progress component in Pending state
    pub fn new(total_windows: usize) -> Self {
        Self {
            state: ScanPauseState::Pending,
            current_window: 0,
            total_windows,
            windows_completed: 0,
            completed_windows: HashSet::new(),
        }
    }

    /// Start processing a window
    pub fn start_window(&mut self, window_index: usize) {
        self.current_window = window_index;
        self.state = ScanPauseState::Scanning;
    }

    /// Complete a window
    pub fn complete_window(&mut self) {
        self.windows_completed += 1;
    }

    /// Mark a specific window as completed
    pub fn complete_window_at(&mut self, window_index: usize) {
        if self.completed_windows.insert(window_index) {
            self.windows_completed += 1;
        }
    }

    /// Check if a window has been completed
    pub fn is_window_completed(&self, window_index: usize) -> bool {
        self.completed_windows.contains(&window_index)
    }

    /// Pause at a specific window
    pub fn pause(&mut self, window_index: usize) {
        self.state = ScanPauseState::PausedAtWindow { window_index };
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
    pub fn start_listening(&mut self, paused_at_window: usize) {
        self.state = ScanPauseState::Listening { paused_at_window };
    }

    /// Exit listening mode
    pub fn stop_listening(&mut self, paused_at_window: usize) {
        self.state = ScanPauseState::PausedAtWindow {
            window_index: paused_at_window,
        };
    }

    /// Check if scan is pending
    pub fn is_pending(&self) -> bool {
        matches!(self.state, ScanPauseState::Pending)
    }

    /// Check if scan is paused
    pub fn is_paused(&self) -> bool {
        matches!(
            self.state,
            ScanPauseState::PausedAtWindow { .. } | ScanPauseState::Listening { .. }
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
}
