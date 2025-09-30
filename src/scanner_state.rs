//! Scanner state machine for pause/resume functionality
//!
//! This module contains the core state machine logic for scanner operation,
//! designed to be testable in isolation from SDR hardware dependencies.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Shared pause signal for immediate cancellation of background operations
#[derive(Debug, Clone)]
pub struct PauseSignal {
    paused: Arc<AtomicBool>,
}

impl PauseSignal {
    /// Create a new pause signal in unpaused state
    pub fn new() -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the pause signal (request immediate pause)
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    /// Clear the pause signal (allow resuming)
    pub fn unpause(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }
}

impl Default for PauseSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a scanning window
#[derive(Debug, Clone, PartialEq)]
pub enum WindowState {
    /// Window has not been started yet
    NotStarted,
    /// Window processing is currently in progress
    InProgress {
        started_at: Instant,
        candidates_found: usize,
    },
    /// Window processing has completed
    Completed {
        completed_at: Instant,
        signals_found: usize,
    },
}

/// Operating mode of the scanner
#[derive(Debug, Clone, PartialEq)]
pub enum ScanMode {
    /// Actively scanning through windows
    Scanning,
    /// Paused at a specific window, browsing previous results
    Paused { paused_at_window: usize },
    /// Listening to a specific station
    Listening {
        paused_at_window: usize,
        listening_start: Instant,
    },
}

/// Scanner state machine for managing scan/pause/resume operations
#[derive(Debug)]
pub struct ScannerState {
    /// Current operating mode
    pub mode: ScanMode,
    /// Current window being processed (when Scanning)
    pub current_window: usize,
    /// State of each window
    pub window_states: HashMap<usize, WindowState>,
}

impl Default for ScannerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ScannerState {
    /// Create a new scanner state in Scanning mode
    pub fn new() -> Self {
        Self {
            mode: ScanMode::Scanning,
            current_window: 0,
            window_states: HashMap::new(),
        }
    }

    /// Start processing a new window
    pub fn start_window(&mut self, window_id: usize) {
        self.current_window = window_id;
        self.window_states.insert(
            window_id,
            WindowState::InProgress {
                started_at: Instant::now(),
                candidates_found: 0,
            },
        );
    }

    /// Mark a window as completed
    pub fn complete_window(&mut self, window_id: usize, signals_found: usize) {
        self.window_states.insert(
            window_id,
            WindowState::Completed {
                completed_at: Instant::now(),
                signals_found,
            },
        );
    }

    /// Handle pause command - transitions to Paused mode
    pub fn handle_pause(&mut self, at_window: usize) {
        // If window is in progress, mark it as not started for idempotent resume
        if let Some(WindowState::InProgress { .. }) = self.window_states.get(&at_window) {
            self.window_states
                .insert(at_window, WindowState::NotStarted);
        }

        self.mode = ScanMode::Paused {
            paused_at_window: at_window,
        };
    }

    /// Handle resume command - returns the window to process next
    pub fn handle_resume(&mut self) -> usize {
        let next_window = if let ScanMode::Paused { paused_at_window } = self.mode {
            // Check if paused window was completed
            match self.window_states.get(&paused_at_window) {
                Some(WindowState::Completed { .. }) => paused_at_window + 1,
                _ => paused_at_window, // Resume from paused window
            }
        } else {
            self.current_window
        };

        self.mode = ScanMode::Scanning;
        self.current_window = next_window;
        next_window
    }

    /// Handle tune to station command - transitions to Listening mode
    pub fn handle_tune(&mut self, paused_at_window: usize) {
        self.mode = ScanMode::Listening {
            paused_at_window,
            listening_start: Instant::now(),
        };
    }

    /// Handle stop listening command - returns to Paused mode
    pub fn handle_stop_listening(&mut self) {
        if let ScanMode::Listening {
            paused_at_window, ..
        } = self.mode
        {
            self.mode = ScanMode::Paused { paused_at_window };
        }
    }

    /// Check if currently paused
    pub fn is_paused(&self) -> bool {
        matches!(self.mode, ScanMode::Paused { .. })
    }

    /// Check if currently listening
    pub fn is_listening(&self) -> bool {
        matches!(self.mode, ScanMode::Listening { .. })
    }

    /// Check if currently scanning
    pub fn is_scanning(&self) -> bool {
        matches!(self.mode, ScanMode::Scanning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = ScannerState::new();
        assert!(state.is_scanning());
        assert_eq!(state.current_window, 0);
        assert!(state.window_states.is_empty());
    }

    #[test]
    fn test_start_and_complete_window() {
        let mut state = ScannerState::new();

        state.start_window(1);
        assert_eq!(state.current_window, 1);
        assert!(matches!(
            state.window_states.get(&1),
            Some(WindowState::InProgress { .. })
        ));

        state.complete_window(1, 3);
        assert!(matches!(
            state.window_states.get(&1),
            Some(WindowState::Completed {
                signals_found: 3,
                ..
            })
        ));
    }

    #[test]
    fn test_pause_marks_incomplete_window_as_not_started() {
        let mut state = ScannerState::new();

        state.start_window(5);
        state.handle_pause(5);

        assert!(state.is_paused());
        assert_eq!(state.window_states.get(&5), Some(&WindowState::NotStarted));
        assert!(matches!(
            state.mode,
            ScanMode::Paused {
                paused_at_window: 5
            }
        ));
    }

    #[test]
    fn test_resume_from_incomplete_window() {
        let mut state = ScannerState::new();

        state.start_window(5);
        state.handle_pause(5);

        let next_window = state.handle_resume();

        assert_eq!(next_window, 5); // Should retry window 5
        assert!(state.is_scanning());
        assert_eq!(state.current_window, 5);
    }

    #[test]
    fn test_resume_from_completed_window() {
        let mut state = ScannerState::new();

        state.start_window(5);
        state.complete_window(5, 2);
        state.handle_pause(5);

        let next_window = state.handle_resume();

        assert_eq!(next_window, 6); // Should move to next window
        assert!(state.is_scanning());
        assert_eq!(state.current_window, 6);
    }

    #[test]
    fn test_tune_and_stop_listening() {
        let mut state = ScannerState::new();

        state.start_window(3);
        state.handle_pause(3);
        state.handle_tune(3);

        assert!(state.is_listening());
        assert!(matches!(
            state.mode,
            ScanMode::Listening {
                paused_at_window: 3,
                ..
            }
        ));

        state.handle_stop_listening();

        assert!(state.is_paused());
        assert!(matches!(
            state.mode,
            ScanMode::Paused {
                paused_at_window: 3
            }
        ));
    }

    #[test]
    fn test_idempotent_window_completion() {
        let mut state = ScannerState::new();

        // Start and pause window 8
        state.start_window(8);
        state.handle_pause(8);
        let next = state.handle_resume();
        assert_eq!(next, 8); // Reprocess window 8

        // Now complete it
        state.start_window(8);
        state.complete_window(8, 4);

        // Pause and resume again
        state.handle_pause(8);
        let next = state.handle_resume();
        assert_eq!(next, 9); // Skip completed window, move to 9
    }

    #[test]
    fn test_state_transitions() {
        let mut state = ScannerState::new();

        // Scanning -> Paused
        assert!(state.is_scanning());
        state.handle_pause(1);
        assert!(state.is_paused());

        // Paused -> Listening
        state.handle_tune(1);
        assert!(state.is_listening());

        // Listening -> Paused
        state.handle_stop_listening();
        assert!(state.is_paused());

        // Paused -> Scanning
        state.handle_resume();
        assert!(state.is_scanning());
    }

    #[test]
    fn test_pause_signal() {
        let signal = PauseSignal::new();

        assert!(!signal.is_paused());

        signal.pause();
        assert!(signal.is_paused());

        signal.unpause();
        assert!(!signal.is_paused());
    }

    #[test]
    fn test_pause_signal_clone_shares_state() {
        let signal1 = PauseSignal::new();
        let signal2 = signal1.clone();

        assert!(!signal1.is_paused());
        assert!(!signal2.is_paused());

        signal1.pause();
        assert!(signal1.is_paused());
        assert!(signal2.is_paused()); // Clone should see the same state

        signal2.unpause();
        assert!(!signal1.is_paused());
        assert!(!signal2.is_paused());
    }
}
