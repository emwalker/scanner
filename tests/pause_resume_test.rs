//! Integration tests for pause/resume functionality
//!
//! These tests verify that the pause mechanism works correctly:
//! - No straggler events after pause
//! - Idempotent window resumption
//! - State machine transitions

use scanner::scanner_state::{PauseSignal, ScanMode, ScannerState, WindowState};
use std::time::Duration;

#[test]
fn test_pause_signal_immediate_response() {
    let signal = PauseSignal::new();

    // Start unpaused
    assert!(!signal.is_paused());

    // Pause immediately
    signal.pause();
    assert!(signal.is_paused());

    // Verify cloned signals see the same state
    let signal_clone = signal.clone();
    assert!(signal_clone.is_paused());

    // Resume
    signal.unpause();
    assert!(!signal.is_paused());
    assert!(!signal_clone.is_paused());
}

#[test]
fn test_scanner_state_pause_marks_incomplete_window() {
    let mut state = ScannerState::new();

    // Start processing window 5
    state.start_window(5);
    assert!(matches!(
        state.window_states.get(&5),
        Some(WindowState::InProgress { .. })
    ));

    // Pause at window 5 (incomplete)
    state.handle_pause(5);

    // Window should be marked as NotStarted for idempotent resume
    assert_eq!(state.window_states.get(&5), Some(&WindowState::NotStarted));
    assert!(matches!(state.mode, ScanMode::Paused(ref p) if p.paused_at_window == 5));
}

#[test]
fn test_idempotent_resume_incomplete_window() {
    let mut state = ScannerState::new();

    // Start window 10
    state.start_window(10);

    // Pause before completion
    state.handle_pause(10);

    // Resume should return window 10 (retry)
    let next_window = state.handle_resume();
    assert_eq!(next_window, 10);
    assert!(state.is_scanning());
}

#[test]
fn test_idempotent_resume_completed_window() {
    let mut state = ScannerState::new();

    // Complete window 10
    state.start_window(10);
    state.complete_window(10, 3);

    // Pause after completion
    state.handle_pause(10);

    // Resume should skip to window 11
    let next_window = state.handle_resume();
    assert_eq!(next_window, 11);
    assert!(state.is_scanning());
}

#[test]
fn test_state_transitions_pause_listen_resume() {
    let mut state = ScannerState::new();

    // Scanning
    assert!(state.is_scanning());
    state.start_window(5);

    // Pause
    state.handle_pause(5);
    assert!(state.is_paused());
    assert!(!state.is_scanning());

    // Tune to station (enter Listening mode)
    state.handle_tune(5);
    assert!(state.is_listening());
    assert!(!state.is_paused());

    // Stop listening (return to Paused)
    state.handle_stop_listening();
    assert!(state.is_paused());
    assert!(!state.is_listening());

    // Resume scanning
    state.handle_resume();
    assert!(state.is_scanning());
    assert!(!state.is_paused());
}

#[test]
fn test_multiple_pause_resume_cycles() {
    let mut state = ScannerState::new();

    // Window 1: Complete
    state.start_window(1);
    state.complete_window(1, 2);

    // Window 2: Pause incomplete
    state.start_window(2);
    state.handle_pause(2);
    assert_eq!(state.window_states.get(&2), Some(&WindowState::NotStarted));

    // Resume - should retry window 2
    let next = state.handle_resume();
    assert_eq!(next, 2);

    // Complete window 2
    state.start_window(2);
    state.complete_window(2, 1);

    // Pause again at window 2 (now completed)
    state.handle_pause(2);

    // Resume - should skip to window 3
    let next = state.handle_resume();
    assert_eq!(next, 3);
}

#[test]
fn test_pause_signal_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let signal = Arc::new(PauseSignal::new());
    let mut handles = vec![];

    // Spawn multiple threads checking the signal
    for _ in 0..5 {
        let signal_clone = signal.clone();
        let handle = thread::spawn(move || {
            // Check signal state repeatedly
            for _ in 0..100 {
                let _ = signal_clone.is_paused();
                thread::sleep(Duration::from_micros(10));
            }
        });
        handles.push(handle);
    }

    // Main thread toggles pause state
    for _ in 0..10 {
        signal.pause();
        thread::sleep(Duration::from_micros(50));
        signal.unpause();
        thread::sleep(Duration::from_micros(50));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // No panics = success
}

#[test]
fn test_window_state_transitions() {
    let mut state = ScannerState::new();

    // NotStarted -> InProgress
    state.start_window(1);
    assert!(matches!(
        state.window_states.get(&1),
        Some(WindowState::InProgress { .. })
    ));

    // InProgress -> Completed
    state.complete_window(1, 5);
    assert!(matches!(
        state.window_states.get(&1),
        Some(WindowState::Completed {
            signals_found: 5,
            ..
        })
    ));

    // InProgress -> NotStarted (via pause)
    state.start_window(2);
    state.handle_pause(2);
    assert_eq!(state.window_states.get(&2), Some(&WindowState::NotStarted));
}

#[test]
fn test_scanner_state_mode_invariants() {
    let mut state = ScannerState::new();

    // Start in Scanning mode
    assert!(matches!(state.mode, ScanMode::Scanning(_)));

    // Pause transitions to Paused
    state.start_window(1);
    state.handle_pause(1);
    assert!(matches!(state.mode, ScanMode::Paused(ref p) if p.paused_at_window == 1));

    // Tune transitions to Listening
    state.handle_tune(1);
    assert!(matches!(
        state.mode,
        ScanMode::Listening(ref l) if l.paused_at_window == 1
    ));

    // StopListening returns to Paused
    state.handle_stop_listening();
    assert!(matches!(state.mode, ScanMode::Paused(ref p) if p.paused_at_window == 1));

    // Resume returns to Scanning
    state.handle_resume();
    assert!(matches!(state.mode, ScanMode::Scanning(_)));
}
