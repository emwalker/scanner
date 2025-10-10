use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test quit functionality

#[test]
fn test_playing_candidates_remain_playing_when_entering_selection_mode() {
    let mut model = Model::new();

    let window_id = 1;
    let freq = 88_900_000.0;
    let candidate_id = "88.9-1".to_string();

    // Create candidate and advance to Playing state
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
        candidate_id: Some(candidate_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Good),
        signal_strength: Some(50.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
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

    // Set current window to match the candidate's window
    model.current_window = window_id;

    // Verify candidate is Playing
    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);

    // Enter selection mode (simulates pressing Up to browse)
    model.enter_selection_mode();

    // Verify candidate remains Playing (navigation doesn't stop playback)
    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);
}

#[test]
fn test_playing_candidates_remain_when_entering_selection_mode() {
    let mut model = Model::new();

    // Create two windows with candidates
    let window1_id = 1;
    let window2_id = 2;
    let freq1 = 88_900_000.0;
    let freq2 = 89_100_000.0;
    let candidate1_id = "88.9-1".to_string();
    let candidate2_id = "89.1-2".to_string();

    // Window 1 candidate - create and advance to Playing state
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: freq1,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: freq1,
            window_id: window1_id,
        },
        candidate_id: Some(candidate1_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::SignalGenerated,
        frequency_hz: freq1,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: freq1,
            window_id: window1_id,
        },
        candidate_id: Some(candidate1_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Good),
        signal_strength: Some(50.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioPlaybackStarted,
        frequency_hz: freq1,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: freq1,
            window_id: window1_id,
        },
        candidate_id: Some(candidate1_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Verify candidate is Playing
    let window = model.windows.get(&window1_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);

    // Window 2 candidate - create and advance to Signal state
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: freq2,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: freq2,
            window_id: window2_id,
        },
        candidate_id: Some(candidate2_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    model.update(ProgressEvent {
        event_type: ProgressEventType::SignalGenerated,
        frequency_hz: freq2,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: freq2,
            window_id: window2_id,
        },
        candidate_id: Some(candidate2_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Moderate),
        signal_strength: Some(40.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Set current window to window 1 (where the Playing candidate is)
    model.current_window = window1_id;

    // Enter selection mode - candidates should remain in their current state
    model.enter_selection_mode();

    // Verify window 1 candidate remains Playing
    let window = model.windows.get(&window1_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);

    // Verify window 2 candidate remains Signal
    let window = model.windows.get(&window2_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);
}

#[test]
fn test_signal_candidates_remain_signal_when_entering_selection_mode() {
    let mut model = Model::new();

    let window_id = 1;
    let freq = 88_900_000.0;
    let candidate_id = "88.9-1".to_string();

    // Create candidate and advance to Signal state
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
        candidate_id: Some(candidate_id.clone()),
        audio_quality: Some(crate::audio::quality::AudioQuality::Good),
        signal_strength: Some(50.0),
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Set current window to match the candidate's window
    model.current_window = window_id;

    // Verify candidate is Signal
    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);

    // Enter selection mode (simulates pressing Up to browse)
    model.enter_selection_mode();

    // Verify candidate remains Signal (navigation doesn't complete candidates)
    let window = model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);
}
