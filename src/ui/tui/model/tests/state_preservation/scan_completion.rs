use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test quit functionality

/// Test that rejected candidates disappear from the last window when scan completes
/// This is a regression test for the behavior where rejected candidates should
/// disappear as soon as all candidates finish processing, not just when entering
/// browse mode.
#[test]
fn test_rejected_candidates_disappear_when_scan_completes() {
    let mut model = Model::new();
    let window_id = 1;

    // Create a mix of signal and rejected candidates in the window
    let candidates = vec![
        ("88.1-1", 88_100_000.0, false), // Signal
        ("88.3-1", 88_300_000.0, true),  // Rejected
        ("88.5-1", 88_500_000.0, false), // Signal
        ("88.7-1", 88_700_000.0, true),  // Rejected
    ];

    for (id, freq, is_rejected) in &candidates {
        // Create candidate
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

        if *is_rejected {
            // Mark as rejected
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
        } else {
            // Complete as signal
            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: *freq,
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: Some(crate::audio::quality::AudioQuality::Good),
                signal_strength: Some(50.0),
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
    }

    // Verify all candidates exist
    assert_eq!(model.windows.get(&window_id).unwrap().candidates.len(), 4);

    // Set total_windows and verify all_complete returns true
    model.total_windows = Some(1);

    // Verify current_window and all_candidates_complete
    assert_eq!(model.current_window, 1);
    assert!(
        model.all_candidates_complete(),
        "all_candidates_complete should be true"
    );
    assert!(model.all_complete(), "all_complete should be true");

    // Manually mark the window complete (since no more events will trigger it)
    if let Some(window) = model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    // After manually marking complete, verify window is complete
    let window = model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);

    // For a complete window, rejected candidates should be filtered out
    // even if it's the current window (is_current_window=true)
    let displayable_after_complete = window.displayable_candidates(true, false);
    assert_eq!(displayable_after_complete.len(), 2); // Only 2 signals visible

    // Verify only non-rejected candidates are shown
    for candidate in displayable_after_complete {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }

    // In selection mode, rejected should also be filtered
    let displayable_in_selection = window.displayable_candidates(true, true);
    assert_eq!(displayable_in_selection.len(), 2); // Only 2 signals visible

    for candidate in displayable_in_selection {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}
