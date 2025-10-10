use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test that candidates progress through all expected states

/// Test that rejected candidates reach terminal state correctly
#[test]
fn test_rejected_candidate_lifecycle() {
    let mut model = Model::new();
    let candidate_id = "88.9-1".to_string();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Step 1: Candidate created
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

    // Step 2: Audio analysis started
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioAnalysisStarted,
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

    // Step 3: Candidate rejected (noise)
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateRejected,
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

    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Rejected);
    assert_eq!(candidate.completion, 1.0); // 100% - terminal state
}
