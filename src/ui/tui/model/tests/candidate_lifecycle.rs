use crate::ui::tui::model::{CandidateStatus, Model};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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

/// Test deterministic candidate ordering within windows
#[test]
fn test_deterministic_candidate_ordering() {
    let mut model = Model::new();
    let window_id = 1;

    // Create candidates in specific order
    let candidates = vec![
        ("89.1-1", 89_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("90.5-1", 90_500_000.0),
        ("87.9-1", 87_900_000.0),
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

    let window = model.windows.get(&window_id).unwrap();

    // Candidates should maintain insertion order
    assert_eq!(window.candidates.len(), 4);
    assert_eq!(window.candidates[0].frequency_hz, 89_100_000.0);
    assert_eq!(window.candidates[1].frequency_hz, 88_300_000.0);
    assert_eq!(window.candidates[2].frequency_hz, 90_500_000.0);
    assert_eq!(window.candidates[3].frequency_hz, 87_900_000.0);

    // displayable_candidates should also maintain this order
    let displayable = window.displayable_candidates(true, false);
    assert_eq!(displayable.len(), 4);
    assert_eq!(displayable[0].frequency_hz, 89_100_000.0);
    assert_eq!(displayable[1].frequency_hz, 88_300_000.0);
    assert_eq!(displayable[2].frequency_hz, 90_500_000.0);
    assert_eq!(displayable[3].frequency_hz, 87_900_000.0);
}

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
