use crate::ui::tui::model::{CandidateStatus, Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

#[test]
fn test_ui_mode_transition_idle_to_navigating() {
    let mut model = Model::new();
    assert!(matches!(model.ui_mode, UiMode::Idle));

    // Simulate pressing Up arrow (first navigation)
    model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    assert!(model.is_navigating());
    assert!(!model.is_idle());
}

#[test]
fn test_ui_mode_transition_navigating_to_awaiting_tune() {
    let mut model = Model::new();
    model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    // Simulate pressing Enter
    model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    assert!(model.is_awaiting_tune());
    assert!(!model.is_navigating());
}

#[test]
fn test_ui_mode_transition_awaiting_tune_to_listening() {
    let mut model = Model::new();
    let window_id = 1;
    let candidate_id = "88.9-1".to_string();

    // Setup: Create a candidate
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    // Simulate AudioPlaybackStarted event
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Should transition to Listening
    assert!(model.is_listening());
    match &model.ui_mode {
        UiMode::Listening {
            playing_candidate_id,
            ..
        } => {
            assert_eq!(playing_candidate_id, &candidate_id);
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_listening_switch_station() {
    let mut model = Model::new();
    let window_id = 1;

    // Create two candidates
    let candidate1_id = "88.5-1".to_string();
    let candidate2_id = "88.9-1".to_string();

    for (id, freq) in [
        (candidate1_id.clone(), 88_500_000.0),
        (candidate2_id.clone(), 88_900_000.0),
    ] {
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(id),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });
    }

    // Start listening to first station
    model.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: candidate1_id.clone(),
    };

    // Switch to second station while still in Listening mode
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate2_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Should still be Listening but with new candidate
    assert!(model.is_listening());
    match &model.ui_mode {
        UiMode::Listening {
            playing_candidate_id,
            navigation_index,
            ..
        } => {
            assert_eq!(playing_candidate_id, &candidate2_id);
            assert_eq!(*navigation_index, 0); // Preserves original navigation_index from Listening mode
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_idle() {
    let mut model = Model::new();
    model.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: "88.9-1".to_string(),
    };

    // Simulate exiting browsing mode (Continue scan)
    model.ui_mode = UiMode::Idle;

    assert!(model.is_idle());
    assert!(!model.is_listening());
}

#[test]
fn test_ui_mode_helper_methods() {
    let model_idle = Model::new();
    assert!(model_idle.is_idle());
    assert!(!model_idle.is_interactive());

    let mut model_navigating = Model::new();
    model_navigating.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };
    assert!(model_navigating.is_navigating());
    assert!(model_navigating.is_interactive());

    let mut model_awaiting = Model::new();
    model_awaiting.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };
    assert!(model_awaiting.is_awaiting_tune());
    assert!(model_awaiting.is_interactive());

    let mut model_listening = Model::new();
    model_listening.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: "88.9-1".to_string(),
    };
    assert!(model_listening.is_listening());
    assert!(model_listening.is_interactive());
}

#[test]
fn test_ui_mode_invalid_transitions_prevented() {
    let mut model = Model::new();
    let window_id = 1;
    let candidate_id = "88.9-1".to_string();

    // Create candidate
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // AudioPlaybackStarted in Idle mode - should not transition
    model.ui_mode = UiMode::Idle;

    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Should still be Idle (transition only happens in AwaitingTune/Listening)
    assert!(model.is_idle());
}

#[test]
fn test_browsing_mode_only_true_when_scan_paused() {
    let mut model = Model::new();
    let window_id = 0;

    // Add a candidate
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some("test-candidate".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Idle mode - browsing_mode should be false
    assert!(model.is_idle());
    assert!(!model.browsing_mode());

    // Enter selection mode (NavigatingScanner) - browsing_mode should still be false
    model.enter_selection_mode();
    assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
    assert!(model.selection_mode());
    assert!(!model.browsing_mode()); // Key assertion: browsing_mode is false while navigating

    // Transition to AwaitingTune - browsing_mode should now be true
    if let Some(selected_index) = model.selected_candidate_index() {
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }
    assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(model.browsing_mode()); // Now true because scan is paused

    // Transition to Listening - browsing_mode should remain true
    if let Some(selected_index) = model.selected_candidate_index() {
        model.ui_mode = UiMode::Listening {
            navigation_index: selected_index,
            playing_index: selected_index,
            playing_candidate_id: "test-candidate".to_string(),
        };
    }
    assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
    assert!(model.browsing_mode()); // Still true when listening
}

#[test]
fn test_enter_key_tunes_to_selected_station() {
    let mut model = Model::new();
    let window_id = 0;
    let candidate_id = "test-candidate".to_string();

    // Add a Signal candidate
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::SignalGenerated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: Some(0.8),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Start in Idle mode
    assert!(model.is_idle());

    // User presses UP arrow to enter selection mode (NavigatingScanner)
    model.enter_selection_mode();
    assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
    assert!(model.selection_mode());
    assert!(!model.browsing_mode()); // Not in browsing mode yet

    // User presses ENTER - should transition to AwaitingTune
    // This simulates the ENTER key handler logic
    if let Some(selected_index) = model.selected_candidate_index() {
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }

    // Verify transition to AwaitingTune
    assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(model.browsing_mode()); // Now in browsing mode (scan paused)

    // Verify selected_candidate_info works in AwaitingTune mode
    let info = model.selected_candidate_info();
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.candidate_id, candidate_id);
    assert_eq!(info.candidate_frequency, 88_900_000.0);

    // Simulate receiving AudioPlaybackStarted event
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: Some(0.8),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Should transition to Listening mode
    assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
    if let UiMode::Listening {
        playing_candidate_id,
        ..
    } = &model.ui_mode
    {
        assert_eq!(playing_candidate_id, &candidate_id);
    }
}

#[test]
fn test_navigation_and_highlight_separate_in_listening_mode() {
    let mut model = Model::new();
    let window_id = 0;

    // Add three candidates
    for i in 0..3 {
        let freq = 88_100_000.0 + (i as f64 * 200_000.0); // 88.1, 88.3, 88.5 MHz
        let candidate_id = format!("candidate_{}", i);

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id),
            audio_quality: None,
            signal_strength: Some(0.8),
            timestamp: Instant::now(),
            tuner_id: None,
        });
    }

    // Enter selection mode and select first candidate (index 0)
    model.enter_selection_mode();
    assert_eq!(model.selected_candidate_index(), Some(2)); // Most recent

    // Move to first candidate
    model.select_previous_candidate();
    model.select_previous_candidate();
    assert_eq!(model.selected_candidate_index(), Some(0));

    // Press ENTER on first candidate - transition to AwaitingTune
    model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    // Verify we're tuning to index 0
    if let UiMode::AwaitingTune {
        navigation_index,
        tuning_index,
    } = &model.ui_mode
    {
        assert_eq!(*navigation_index, 0);
        assert_eq!(*tuning_index, 0);
    }

    // Arrow down to navigate to second candidate
    model.select_next_candidate();

    // Verify navigation moved but tuning index stayed the same
    if let UiMode::AwaitingTune {
        navigation_index,
        tuning_index,
    } = &model.ui_mode
    {
        assert_eq!(*navigation_index, 1, "Navigation should move to index 1");
        assert_eq!(*tuning_index, 0, "Tuning should stay at index 0");
    } else {
        panic!("Should still be in AwaitingTune mode");
    }

    // Transition to Listening mode
    model.ui_mode = UiMode::Listening {
        navigation_index: 1,
        playing_index: 0,
        playing_candidate_id: "candidate_0".to_string(),
    };

    // Arrow down again to third candidate
    model.select_next_candidate();

    // Verify navigation moved but playing index stayed the same
    if let UiMode::Listening {
        navigation_index,
        playing_index,
        playing_candidate_id,
    } = &model.ui_mode
    {
        assert_eq!(*navigation_index, 2, "Navigation should move to index 2");
        assert_eq!(*playing_index, 0, "Playing should stay at index 0");
        assert_eq!(playing_candidate_id, "candidate_0");
    } else {
        panic!("Should still be in Listening mode");
    }

    // Arrow up back to second candidate
    model.select_previous_candidate();

    // Verify navigation moved back but playing index still unchanged
    if let UiMode::Listening {
        navigation_index,
        playing_index,
        ..
    } = &model.ui_mode
    {
        assert_eq!(
            *navigation_index, 1,
            "Navigation should move back to index 1"
        );
        assert_eq!(*playing_index, 0, "Playing should still be at index 0");
    }
}

#[test]
fn test_stop_listening_transitions_candidate_to_completed() {
    let mut model = Model::default();
    let window_id = 1;
    let frequency = 88_900_000.0;
    let candidate_id = format!("{:.1}-{}", frequency / 1e6, window_id);

    // Step 1: Create candidate in window 1
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: frequency,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: frequency,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Step 2: Complete audio analysis
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioAnalysisCompleted,
        frequency_hz: frequency,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: frequency,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Step 3: Generate signal
    model.update(ProgressEvent {
        event_type: ProgressEventType::SignalGenerated,
        frequency_hz: frequency,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: frequency,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Good),
        signal_strength: Some(50.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Step 4: Pause scanning and enter interactive mode
    model.enter_selection_mode();
    if let Some(selected_index) = model.selected_candidate_index() {
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }
    assert!(model.browsing_mode());

    // Step 5: Start playing audio from window 1
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: frequency,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: frequency,
            window_id,
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Verify candidate is in Playing state
    let window = model.windows.get(&window_id).unwrap();
    let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
    let candidate = &window.candidates[*candidate_index];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);

    // Step 6: Simulate scanning having progressed to window 2 (making window 1 an "old window")
    // This tests the "old window" filtering bug where AudioPlaybackCompleted was rejected
    // In a real scenario, this could happen if scanning resumed briefly or if there are
    // multiple tuners scanning while one is listening
    model.current_window = 2;

    // Verify current_window has advanced to 2
    assert_eq!(model.current_window, 2);

    // Verify we're still in interactive mode
    assert!(model.is_interactive());

    // Step 7: Stop listening to the station from window 1 (now an "old window")
    // Regression test for TWO bugs:
    // 1. AudioPlaybackCompleted was filtered out in interactive mode by should_process_event()
    // 2. AudioPlaybackCompleted was filtered out for old windows by update_candidate()
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackCompleted,
        frequency_hz: frequency,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: frequency,
            window_id, // window 1 is now "old" since current_window is 2
        },
        candidate_id: Some(candidate_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Good),
        signal_strength: Some(50.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Verify candidate transitioned to Completed state despite being in an old window
    let window = model.windows.get(&window_id).unwrap();
    let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
    let candidate = &window.candidates[*candidate_index];
    assert_eq!(
        candidate.status,
        CandidateStatus::Completed,
        "Candidate should transition to Completed when AudioPlaybackCompleted is sent, \
             even when in interactive mode (bug #1) and from an old window (bug #2)"
    );
    assert_eq!(candidate.completion, 1.0);
}
