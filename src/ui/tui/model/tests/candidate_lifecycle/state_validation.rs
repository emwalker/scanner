use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test that candidates progress through all expected states

/// Test that no candidates remain stuck in intermediate states
#[test]
fn test_no_stuck_intermediate_states() {
    let mut model = Model::new();
    let window_id = 1;

    // Create multiple candidates in different states
    let candidates = vec![
        ("88.1-1", 88_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("88.5-1", 88_500_000.0),
        ("88.7-1", 88_700_000.0),
        ("88.9-1", 88_900_000.0),
    ];

    // Create all candidates
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

    // Start analysis for all
    for (id, freq) in &candidates {
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
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

    // Resolve all candidates to terminal states
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

    model.update(ProgressEvent {
        event_type: ProgressEventType::CandidateRejected,
        frequency_hz: candidates[1].1,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: candidates[1].1,
            window_id,
        },
        candidate_id: Some(candidates[1].0.to_string()),
        audio_quality: None,
        signal_strength: None,
        timestamp: Instant::now(),
        tuner_id: None,
    });

    // Complete signal paths for others
    for (id, freq) in &candidates[2..] {
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

    // Verify no candidates are stuck in intermediate states
    let window = model.windows.get(&window_id).unwrap();
    for candidate in &window.candidates {
        match candidate.status {
            CandidateStatus::Detected | CandidateStatus::Analyzing => {
                panic!(
                    "Candidate at {:.1} MHz stuck in intermediate state: {:?}",
                    candidate.frequency_hz / 1e6,
                    candidate.status
                );
            }
            CandidateStatus::Rejected | CandidateStatus::Completed => {
                // Terminal states are good
                assert_eq!(candidate.completion, 1.0);
            }
            CandidateStatus::Signal | CandidateStatus::Playing => {
                // These are valid but should have progressed to Completed
                panic!(
                    "Candidate at {:.1} MHz should have completed: {:?}",
                    candidate.frequency_hz / 1e6,
                    candidate.status
                );
            }
        }
    }
}
