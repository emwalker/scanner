use crate::ui::tui::model::{Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
