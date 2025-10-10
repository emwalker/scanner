use crate::ui::tui::model::{Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
