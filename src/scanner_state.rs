//! Scanner state machine for pause/resume functionality
//!
//! This module contains the core state machine logic for scanner operation,
//! designed to be testable in isolation from SDR hardware dependencies.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Events that trigger state transitions
#[derive(Debug, Clone, PartialEq)]
pub enum ScannerEvent {
    /// Pause scanning at the given window
    Pause { at_window: usize },
    /// Resume scanning
    Resume,
    /// Tune to a specific station
    TuneToStation { at_window: usize },
    /// Stop listening and return to browsing
    StopListening,
    /// Mark scan as complete
    ScanComplete { windows_processed: usize },
    /// Initiate shutdown
    Shutdown,
}

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

/// State: Actively scanning through windows
#[derive(Debug, Clone, PartialEq)]
pub struct Scanning;

impl Scanning {
    /// Transition to Paused state
    pub fn pause(self, at_window: usize) -> Paused {
        Paused {
            paused_at_window: at_window,
        }
    }

    /// Transition to ScanComplete state
    pub fn complete(self, windows_processed: usize) -> ScanComplete {
        ScanComplete { windows_processed }
    }
}

/// State: Paused at a specific window, browsing previous results
#[derive(Debug, Clone, PartialEq)]
pub struct Paused {
    pub paused_at_window: usize,
}

impl Paused {
    /// Transition to Scanning state, returning the window to resume from
    pub fn resume(self) -> (Scanning, usize) {
        (Scanning, self.paused_at_window)
    }

    /// Transition to Listening state
    pub fn tune(self) -> Listening {
        Listening {
            paused_at_window: self.paused_at_window,
            listening_start: Instant::now(),
        }
    }

    /// Transition to ScanCompletePaused state
    pub fn complete(self, windows_processed: usize) -> ScanCompletePaused {
        ScanCompletePaused { windows_processed }
    }
}

/// State: Listening to a specific station
#[derive(Debug, Clone, PartialEq)]
pub struct Listening {
    pub paused_at_window: usize,
    pub listening_start: Instant,
}

impl Listening {
    /// Transition back to Paused state
    pub fn stop_listening(self) -> Paused {
        Paused {
            paused_at_window: self.paused_at_window,
        }
    }

    /// Get the duration we've been listening
    pub fn duration(&self) -> std::time::Duration {
        self.listening_start.elapsed()
    }
}

/// State: Scan complete, waiting for further commands
#[derive(Debug, Clone, PartialEq)]
pub struct ScanComplete {
    pub windows_processed: usize,
}

impl ScanComplete {
    /// Transition to ScanCompletePaused state
    pub fn pause(self) -> ScanCompletePaused {
        ScanCompletePaused {
            windows_processed: self.windows_processed,
        }
    }
}

/// State: Scan complete and paused (browsing results)
#[derive(Debug, Clone, PartialEq)]
pub struct ScanCompletePaused {
    pub windows_processed: usize,
}

impl ScanCompletePaused {
    /// Transition back to ScanComplete state
    pub fn resume(self) -> ScanComplete {
        ScanComplete {
            windows_processed: self.windows_processed,
        }
    }
}

/// State: Shutting down - cleanup in progress
#[derive(Debug, Clone, PartialEq)]
pub struct ShuttingDown;

/// Operating mode of the scanner
///
/// This enum wraps typestate structs, allowing compile-time type safety
/// for state transitions while maintaining runtime flexibility for
/// dynamic event handling.
#[derive(Debug, Clone, PartialEq)]
pub enum ScanMode {
    /// Actively scanning through windows
    Scanning(Scanning),
    /// Paused at a specific window, browsing previous results
    Paused(Paused),
    /// Listening to a specific station
    Listening(Listening),
    /// Scan complete, waiting for further commands
    ScanComplete(ScanComplete),
    /// Scan complete and paused (browsing results)
    ScanCompletePaused(ScanCompletePaused),
    /// Shutting down - cleanup in progress
    ShuttingDown(ShuttingDown),
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
            mode: ScanMode::Scanning(Scanning),
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

        self.mode = ScanMode::Paused(Paused {
            paused_at_window: at_window,
        });
    }

    /// Handle resume command - returns the window to process next
    pub fn handle_resume(&mut self) -> usize {
        let next_window = if let ScanMode::Paused(ref paused) = self.mode {
            // Check if paused window was completed
            match self.window_states.get(&paused.paused_at_window) {
                Some(WindowState::Completed { .. }) => paused.paused_at_window + 1,
                _ => paused.paused_at_window, // Resume from paused window
            }
        } else {
            self.current_window
        };

        self.mode = ScanMode::Scanning(Scanning);
        self.current_window = next_window;
        next_window
    }

    /// Handle tune to station command - transitions to Listening mode
    pub fn handle_tune(&mut self, paused_at_window: usize) {
        self.mode = ScanMode::Listening(Listening {
            paused_at_window,
            listening_start: Instant::now(),
        });
    }

    /// Handle stop listening command - returns to Paused mode
    pub fn handle_stop_listening(&mut self) {
        if let ScanMode::Listening(ref listening) = self.mode {
            self.mode = ScanMode::Paused(Paused {
                paused_at_window: listening.paused_at_window,
            });
        }
    }

    /// Check if currently paused
    pub fn is_paused(&self) -> bool {
        matches!(
            self.mode,
            ScanMode::Paused(_) | ScanMode::ScanCompletePaused(_)
        )
    }

    /// Check if currently listening
    pub fn is_listening(&self) -> bool {
        matches!(self.mode, ScanMode::Listening(_))
    }

    /// Check if currently scanning
    pub fn is_scanning(&self) -> bool {
        matches!(self.mode, ScanMode::Scanning(_))
    }

    /// Check if scan is complete
    pub fn is_scan_complete(&self) -> bool {
        matches!(
            self.mode,
            ScanMode::ScanComplete(_) | ScanMode::ScanCompletePaused(_)
        )
    }

    /// Check if shutting down
    pub fn is_shutting_down(&self) -> bool {
        matches!(self.mode, ScanMode::ShuttingDown(_))
    }

    /// Mark scan as complete
    pub fn mark_scan_complete(&mut self, windows_processed: usize) {
        match self.mode {
            ScanMode::Scanning(_) => {
                self.mode = ScanMode::ScanComplete(ScanComplete { windows_processed });
            }
            ScanMode::Paused(_) => {
                self.mode = ScanMode::ScanCompletePaused(ScanCompletePaused { windows_processed });
            }
            _ => {}
        }
    }

    /// Transition to shutting down state
    pub fn shutdown(&mut self) {
        self.mode = ScanMode::ShuttingDown(ShuttingDown);
    }

    /// Centralized state transition function
    ///
    /// This is the single point where state transitions occur, making it easier to:
    /// - Test all possible transitions
    /// - Validate state transition logic
    /// - Add logging/debugging for state changes
    ///
    /// Returns the next window to process (if applicable)
    pub fn transition(&mut self, event: ScannerEvent) -> Option<usize> {
        match (&self.mode, &event) {
            // Shutdown can happen from any state
            (_, ScannerEvent::Shutdown) => {
                self.mode = ScanMode::ShuttingDown(ShuttingDown);
                None
            }

            // Scanning -> Paused
            (ScanMode::Scanning(_), ScannerEvent::Pause { at_window }) => {
                self.handle_pause(*at_window);
                None
            }

            // Paused -> Scanning
            (ScanMode::Paused(_), ScannerEvent::Resume) => {
                let next_window = self.handle_resume();
                Some(next_window)
            }

            // Paused -> Listening
            (ScanMode::Paused(_), ScannerEvent::TuneToStation { at_window }) => {
                self.handle_tune(*at_window);
                None
            }

            // Listening -> Paused
            (ScanMode::Listening(_), ScannerEvent::StopListening) => {
                self.handle_stop_listening();
                None
            }

            // Scanning -> ScanComplete
            (ScanMode::Scanning(_), ScannerEvent::ScanComplete { windows_processed }) => {
                self.mode = ScanMode::ScanComplete(ScanComplete {
                    windows_processed: *windows_processed,
                });
                None
            }

            // Paused -> ScanCompletePaused
            (ScanMode::Paused(_), ScannerEvent::ScanComplete { windows_processed }) => {
                self.mode = ScanMode::ScanCompletePaused(ScanCompletePaused {
                    windows_processed: *windows_processed,
                });
                None
            }

            // ScanComplete -> Paused (when user pauses after scan)
            (ScanMode::ScanComplete(complete), ScannerEvent::Pause { at_window: _ }) => {
                self.mode = ScanMode::ScanCompletePaused(ScanCompletePaused {
                    windows_processed: complete.windows_processed,
                });
                None
            }

            // ScanCompletePaused -> ScanComplete (when user resumes)
            (ScanMode::ScanCompletePaused(paused), ScannerEvent::Resume) => {
                self.mode = ScanMode::ScanComplete(ScanComplete {
                    windows_processed: paused.windows_processed,
                });
                None
            }

            // Invalid transitions - no state change
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_shutdown_from_any_state() {
        let states = vec![
            ScanMode::Scanning(Scanning),
            ScanMode::Paused(Paused {
                paused_at_window: 5,
            }),
            ScanMode::Listening(Listening {
                paused_at_window: 5,
                listening_start: Instant::now(),
            }),
            ScanMode::ScanComplete(ScanComplete {
                windows_processed: 10,
            }),
            ScanMode::ScanCompletePaused(ScanCompletePaused {
                windows_processed: 10,
            }),
        ];

        for initial_state in states {
            let mut state = ScannerState::new();
            state.mode = initial_state.clone();

            state.transition(ScannerEvent::Shutdown);

            assert!(
                matches!(state.mode, ScanMode::ShuttingDown(_)),
                "Shutdown should work from {:?}",
                initial_state
            );
        }
    }

    #[test]
    fn test_transition_scanning_to_paused() {
        let mut state = ScannerState::new();
        assert!(matches!(state.mode, ScanMode::Scanning(_)));

        state.transition(ScannerEvent::Pause { at_window: 5 });

        assert!(matches!(state.mode, ScanMode::Paused(p) if p.paused_at_window == 5));
    }

    #[test]
    fn test_transition_paused_to_scanning() {
        let mut state = ScannerState::new();
        state.handle_pause(5);
        assert!(state.is_paused());

        let next_window = state.transition(ScannerEvent::Resume);

        assert!(state.is_scanning());
        assert_eq!(next_window, Some(5));
    }

    #[test]
    fn test_transition_paused_to_listening() {
        let mut state = ScannerState::new();
        state.handle_pause(5);

        state.transition(ScannerEvent::TuneToStation { at_window: 5 });

        assert!(state.is_listening());
        assert!(matches!(
            state.mode,
            ScanMode::Listening(Listening {
                paused_at_window: 5,
                ..
            })
        ));
    }

    #[test]
    fn test_transition_listening_to_paused() {
        let mut state = ScannerState::new();
        state.handle_pause(5);
        state.handle_tune(5);
        assert!(state.is_listening());

        state.transition(ScannerEvent::StopListening);

        assert!(state.is_paused());
        assert!(matches!(state.mode, ScanMode::Paused(p) if p.paused_at_window == 5));
    }

    #[test]
    fn test_transition_scanning_to_scan_complete() {
        let mut state = ScannerState::new();
        assert!(state.is_scanning());

        state.transition(ScannerEvent::ScanComplete {
            windows_processed: 100,
        });

        assert!(state.is_scan_complete());
        assert!(matches!(state.mode, ScanMode::ScanComplete(sc) if sc.windows_processed == 100));
    }

    #[test]
    fn test_transition_paused_to_scan_complete_paused() {
        let mut state = ScannerState::new();
        state.handle_pause(50);

        state.transition(ScannerEvent::ScanComplete {
            windows_processed: 100,
        });

        assert!(state.is_scan_complete());
        assert!(state.is_paused());
        assert!(
            matches!(state.mode, ScanMode::ScanCompletePaused(scp) if scp.windows_processed == 100)
        );
    }

    #[test]
    fn test_transition_scan_complete_to_paused() {
        let mut state = ScannerState::new();
        state.mode = ScanMode::ScanComplete(ScanComplete {
            windows_processed: 100,
        });

        state.transition(ScannerEvent::Pause { at_window: 50 });

        assert!(state.is_paused());
        assert!(
            matches!(state.mode, ScanMode::ScanCompletePaused(scp) if scp.windows_processed == 100)
        );
    }

    #[test]
    fn test_transition_scan_complete_paused_to_scan_complete() {
        let mut state = ScannerState::new();
        state.mode = ScanMode::ScanCompletePaused(ScanCompletePaused {
            windows_processed: 100,
        });

        state.transition(ScannerEvent::Resume);

        assert!(state.is_scan_complete());
        assert!(!state.is_paused());
        assert!(matches!(state.mode, ScanMode::ScanComplete(sc) if sc.windows_processed == 100));
    }

    #[test]
    fn test_transition_invalid_transitions_ignored() {
        let mut state = ScannerState::new();
        state.mode = ScanMode::Scanning(Scanning);

        state.transition(ScannerEvent::StopListening);

        assert!(state.is_scanning(), "Invalid transition should be ignored");
    }

    #[test]
    fn test_transition_state_machine_coverage() {
        let mut state = ScannerState::new();

        state.transition(ScannerEvent::Pause { at_window: 1 });
        assert!(state.is_paused());

        state.transition(ScannerEvent::TuneToStation { at_window: 1 });
        assert!(state.is_listening());

        state.transition(ScannerEvent::StopListening);
        assert!(state.is_paused());

        state.transition(ScannerEvent::Resume);
        assert!(state.is_scanning());

        state.transition(ScannerEvent::ScanComplete {
            windows_processed: 10,
        });
        assert!(state.is_scan_complete());

        state.transition(ScannerEvent::Shutdown);
        assert!(state.is_shutting_down());
    }

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
        assert!(matches!(state.mode, ScanMode::Paused(p) if p.paused_at_window == 5));
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
            ScanMode::Listening(Listening {
                paused_at_window: 3,
                ..
            })
        ));

        state.handle_stop_listening();

        assert!(state.is_paused());
        assert!(matches!(state.mode, ScanMode::Paused(p) if p.paused_at_window == 3));
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
