use crate::ui::tui::model::{CandidateStatus, Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
