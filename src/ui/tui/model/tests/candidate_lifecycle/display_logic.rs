use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;
/// Test that candidates progress through all expected states
/// Test window filtering behavior - only non-rejected candidates shown for complete windows
#[test]
fn test_window_candidate_filtering() {
    let mut model = Model::new();
    let window_id = 1;
    // Create multiple candidates
    let candidates = vec![
        ("88.1-1", 88_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("88.5-1", 88_500_000.0),
    ];
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
    // Reject first candidate, complete others
    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateRejected,
        frequency_hz: candidates[0].1,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: candidates[0].1,
            window_id,
        },
        candidate_id: Some(candidates[0].0.to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });
    for (id, freq) in &candidates[1..] {
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
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
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
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
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
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
    // Mark window complete by starting window 2
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
    let window = model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);
    // For complete windows, rejected candidates are always filtered out
    // (even if it's the current window, even if not in selection mode)
    let current_displayable = window.displayable_candidates(true, false);
    assert_eq!(current_displayable.len(), 2); // Only non-rejected
    // Same for non-current complete windows
    let completed_displayable = window.displayable_candidates(false, false);
    assert_eq!(completed_displayable.len(), 2); // Only non-rejected
    // Verify the rejected candidate is filtered out
    for candidate in current_displayable {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
    for candidate in completed_displayable {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}
/// Test that window should_display logic works correctly
#[test]
fn test_window_display_logic() {
    let mut model = Model::new();
    let window_id = 1;
    // Create window with all rejected candidates
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
    // Mark window complete by starting window 2
    model.total_windows = Some(2);
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
    // After window 2 is created, window 1 should be marked complete
    let window = model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);
    // Complete window with only rejected candidates should not display
    assert!(!window.should_display());
}
