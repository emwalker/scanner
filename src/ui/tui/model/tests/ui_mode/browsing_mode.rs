use crate::ui::tui::model::{Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
