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

    assert_eq!(next_window, 5);
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

    assert_eq!(next_window, 6);
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

    state.start_window(8);
    state.handle_pause(8);
    let next = state.handle_resume();
    assert_eq!(next, 8);

    state.start_window(8);
    state.complete_window(8, 4);

    state.handle_pause(8);
    let next = state.handle_resume();
    assert_eq!(next, 9);
}

#[test]
fn test_state_transitions() {
    let mut state = ScannerState::new();

    assert!(state.is_scanning());
    state.handle_pause(1);
    assert!(state.is_paused());

    state.handle_tune(1);
    assert!(state.is_listening());

    state.handle_stop_listening();
    assert!(state.is_paused());

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
    assert!(signal2.is_paused());

    signal2.unpause();
    assert!(!signal1.is_paused());
    assert!(!signal2.is_paused());
}
