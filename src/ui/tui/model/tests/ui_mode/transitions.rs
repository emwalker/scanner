use crate::ui::tui::model::{Model, UiMode};
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
