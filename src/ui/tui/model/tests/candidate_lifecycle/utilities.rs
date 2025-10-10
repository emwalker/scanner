use crate::ui::tui::model::Model;
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test that candidates progress through all expected states

/// Test model utility functions
#[test]
fn test_model_utility_functions() {
    let mut model = Model::new();

    // Empty model - all_complete returns false for empty models
    assert!(model.is_empty());
    assert!(!model.all_complete()); // Empty model returns false for all_complete
    assert_eq!(model.candidate_count(), 0);

    // Add some candidates
    let window_id = 1;
    let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

    for (id, freq) in &candidates {
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: *freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: *freq,
                window_id,
            },
            candidate_id: Some(id.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });
    }

    // Model with incomplete candidates
    assert!(!model.is_empty());
    assert!(!model.all_complete());
    assert_eq!(model.candidate_count(), 2);

    // Complete all candidates
    for (id, freq) in &candidates {
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: *freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: *freq,
                window_id,
            },
            candidate_id: Some(id.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });
    }

    // Model with complete candidates
    assert!(!model.is_empty());
    model.total_windows = Some(1);
    assert!(model.all_complete());
    assert_eq!(model.candidate_count(), 2);
}
