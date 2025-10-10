use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test quit functionality

/// Test AudioAnalysisCompleted event handling preserves Signal status
#[test]
fn test_audio_analysis_completed_preserves_signal() {
    let mut model = Model::new();
    let candidate_id = "88.9-1".to_string();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Create candidate and start analysis
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

    // Generate signal first
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
    assert_eq!(candidate.completion, 0.6);

    // AudioAnalysisCompleted should not override Signal status
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

    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6); // Should remain unchanged
}

/// Test that status text mapping remains exactly the same
#[test]
fn test_status_text_mapping_unchanged() {
    // These exact strings must be preserved across refactoring
    assert_eq!(CandidateStatus::Detected.to_string(), "DETECTED");
    assert_eq!(CandidateStatus::Analyzing.to_string(), "ANALYZING");
    assert_eq!(CandidateStatus::Rejected.to_string(), "NOISE");
    assert_eq!(CandidateStatus::Signal.to_string(), "SIGNAL");
    assert_eq!(CandidateStatus::Playing.to_string(), "PLAYING");
    assert_eq!(CandidateStatus::Completed.to_string(), "DONE");
}

/// Test that progress percentage calculations remain exact
#[test]
fn test_progress_percentages_unchanged() {
    let mut model = Model::new();
    let candidate_id = "88.9-1".to_string();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Test each state's exact completion percentage
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
    assert_eq!(candidate.completion, 0.3); // DETECTED = 30%

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
    assert_eq!(candidate.completion, 0.5); // ANALYZING = 50%

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
    assert_eq!(candidate.completion, 0.6); // SIGNAL = 60%

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
    assert_eq!(candidate.completion, 0.8); // PLAYING = 80%

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
    assert_eq!(candidate.completion, 1.0); // DONE = 100%

    // Test rejected path
    let rejected_id = "89.1-1".to_string();
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: 89_100_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 89_100_000.0,
            window_id,
        },
        candidate_id: Some(rejected_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateRejected,
        frequency_hz: 89_100_000.0,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: 89_100_000.0,
            window_id,
        },
        candidate_id: Some(rejected_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    let window = model.windows.get(&window_id).unwrap();
    let rejected_candidate = &window.candidates[1];
    assert_eq!(rejected_candidate.completion, 1.0); // NOISE = 100%
}
