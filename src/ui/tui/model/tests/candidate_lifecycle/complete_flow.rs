use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test that candidates progress through all expected states

/// Test that candidates progress through all expected states
#[test]
fn test_complete_candidate_lifecycle() {
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

    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Detected);
    assert_eq!(candidate.completion, 0.3); // 30%

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

    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Analyzing);
    assert_eq!(candidate.completion, 0.5); // 50%

    // Step 3: Signal generated (good signal path)
    model.update(ProgressEvent {
        event_type: ProgressEventType::SignalGenerated,
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
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6); // 60%

    // Step 4: Audio playback started
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

    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8); // 80%

    // Step 5: Audio playback completed
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackCompleted,
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
    assert_eq!(candidate.status, CandidateStatus::Completed);
    assert_eq!(candidate.completion, 1.0); // 100%
}
