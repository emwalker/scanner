use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test that candidates progress through all expected states

/// Test that windows complete sequentially, not overlapping
#[test]
fn test_sequential_window_completion() {
    let mut model = Model::new();

    // Create candidates in window 1
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id: 1,
        },
        candidate_id: Some("88.9-1".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    assert_eq!(model.current_window, 1);
    assert!(!model.windows.get(&1).unwrap().is_complete);

    // Start window 2 - should mark window 1 as complete
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 89_100_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 89_100_000.0,
            window_id: 2,
        },
        candidate_id: Some("89.1-2".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    assert_eq!(model.current_window, 2);
    assert!(model.windows.get(&1).unwrap().is_complete);
    assert!(!model.windows.get(&2).unwrap().is_complete);

    // Start window 3 - should mark window 2 as complete
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 89_300_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 89_300_000.0,
            window_id: 3,
        },
        candidate_id: Some("89.3-3".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    assert_eq!(model.current_window, 3);
    assert!(model.windows.get(&1).unwrap().is_complete);
    assert!(model.windows.get(&2).unwrap().is_complete);
    assert!(!model.windows.get(&3).unwrap().is_complete);
}

/// Test that old window events are ignored after window completion
#[test]
fn test_old_window_events_ignored() {
    let mut model = Model::new();

    // Create candidate in window 1
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id: 1,
        },
        candidate_id: Some("88.9-1".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Start window 2 (marks window 1 complete)
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 89_100_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 89_100_000.0,
            window_id: 2,
        },
        candidate_id: Some("89.1-2".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    let window1_candidate_count = model.windows.get(&1).unwrap().candidates.len();

    // Try to add another candidate to completed window 1 - should be ignored
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 88_700_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_700_000.0,
            window_id: 1,
        },
        candidate_id: Some("88.7-1".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Window 1 should still have the same number of candidates
    assert_eq!(
        model.windows.get(&1).unwrap().candidates.len(),
        window1_candidate_count
    );

    // Try to update existing candidate in window 1 - should be ignored
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioAnalysisStarted,
        frequency_hz: 88_900_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 88_900_000.0,
            window_id: 1,
        },
        candidate_id: Some("88.9-1".to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Candidate should still be in original state
    let candidate = &model.windows.get(&1).unwrap().candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Detected);
    assert_eq!(candidate.completion, 0.3);
}
