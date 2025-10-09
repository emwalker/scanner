#[cfg(test)]
mod tests {
    use crate::terminal::tui::model::{CandidateStatus, Model, TunerState, UiMode};
    use crate::terminal::{ProgressEvent, ProgressEventType};
    use std::time::Instant;

    fn create_test_pool_status(
        available: Vec<crate::sdr::DeviceId>,
        scanning: Vec<crate::sdr::DeviceId>,
        listening: Vec<crate::sdr::DeviceId>,
    ) -> crate::pool::PoolStatus {
        use crate::pool::{PoolStatus, TunerActivity, TunerId, TunerState, TunerStatus};

        let mut tuners = Vec::new();

        // Add all tuners from available list
        for device_id in available.iter() {
            let is_scanning = scanning.contains(device_id);
            let is_listening = listening.contains(device_id);

            let (state, activity) = if is_scanning {
                (TunerState::Allocated, Some(TunerActivity::Scanning))
            } else if is_listening {
                (TunerState::Allocated, Some(TunerActivity::Listening))
            } else {
                (TunerState::Available, None)
            };

            tuners.push(TunerStatus {
                id: TunerId {
                    device_id: device_id.clone(),
                    channel_index: 0,
                },
                model: "Test Device".to_string(),
                backend: "test".to_string(),
                channel_index: 0,
                state,
                activity,
            });
        }

        let available_count = available
            .iter()
            .filter(|id| !scanning.contains(id) && !listening.contains(id))
            .count();
        let allocated_count = scanning.len() + listening.len();

        PoolStatus {
            tuners,
            available_tuner_count: available_count,
            allocated_tuner_count: allocated_count,
            device_count: available.len(),
        }
    }

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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
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

    /// Test quit functionality
    #[test]
    fn test_quit_functionality() {
        let mut model = Model::new();

        assert!(!model.should_quit);

        model.quit();

        assert!(model.should_quit);
    }

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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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

    #[test]
    fn test_browsing_mode_playing_correct_candidate() {
        let mut model = Model::new();
        let window_id = 1;

        // Create three candidates at different frequencies
        let freq1 = 88_500_000.0;
        let freq2 = 88_900_000.0;
        let freq3 = 89_300_000.0;
        let candidate1_id = "88.5-1".to_string();
        let candidate2_id = "88.9-1".to_string();
        let candidate3_id = "89.3-1".to_string();

        // Create all three candidates in Signal state
        for (freq, candidate_id) in [
            (freq1, candidate1_id.clone()),
            (freq2, candidate2_id.clone()),
            (freq3, candidate3_id.clone()),
        ] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id.clone()),
                audio_quality: Some(crate::audio_quality::AudioQuality::Good),
                signal_strength: Some(50.0),
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        model.current_window = window_id;

        // Verify all three candidates are in Signal state
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(window.candidates.len(), 3);
        assert_eq!(window.candidates[0].frequency_hz, freq1);
        assert_eq!(window.candidates[1].frequency_hz, freq2);
        assert_eq!(window.candidates[2].frequency_hz, freq3);
        assert_eq!(window.candidates[0].status, CandidateStatus::Signal);
        assert_eq!(window.candidates[1].status, CandidateStatus::Signal);
        assert_eq!(window.candidates[2].status, CandidateStatus::Signal);

        // Enter browsing mode and transition to AwaitingTune
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 1,
            tuning_index: 1,
        };

        // Send AudioPlaybackStarted for the middle candidate (88.9)
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify ONLY the middle candidate is Playing
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(
            window.candidates[0].status,
            CandidateStatus::Signal,
            "First candidate should still be Signal"
        );
        assert_eq!(
            window.candidates[1].status,
            CandidateStatus::Playing,
            "Second candidate should be Playing"
        );
        assert_eq!(
            window.candidates[2].status,
            CandidateStatus::Signal,
            "Third candidate should still be Signal"
        );

        // Now switch to a different candidate (89.3)
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 2 };

        // Send AudioPlaybackStarted for the third candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq3,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq3,
                window_id,
            },
            candidate_id: Some(candidate3_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify only the third candidate is Playing - the second should have been auto-completed
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(
            window.candidates[0].status,
            CandidateStatus::Signal,
            "First candidate should still be Signal"
        );
        assert_eq!(
            window.candidates[1].status,
            CandidateStatus::Completed,
            "Second candidate should be Completed (was replaced)"
        );
        assert_eq!(
            window.candidates[2].status,
            CandidateStatus::Playing,
            "Third candidate should be Playing"
        );
    }

    #[test]
    fn test_browsing_mode_allows_old_window_playback() {
        let mut model = Model::new();

        // Create candidate in window 1
        let window1_id = 1;
        let freq1 = 88_900_000.0;
        let candidate1_id = "88.9-1".to_string();

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Create candidate in window 2 (this marks window 1 as complete)
        let window2_id = 2;
        let freq2 = 89_300_000.0;
        let candidate2_id = "89.3-2".to_string();

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify we're now in window 2
        assert_eq!(model.current_window, window2_id);
        assert!(model.windows.get(&window1_id).unwrap().is_complete);

        // In normal scanning mode, events to old windows should be blocked
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Status should still be Signal (event was blocked)
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(window1.candidates[0].status, CandidateStatus::Signal);

        // Now enter browsing mode by transitioning to Navigating mode
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        // Send AudioPlaybackStarted for the old window candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // In browsing mode, AudioPlaybackStarted should work even for old windows
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "AudioPlaybackStarted should work for old windows in browsing mode"
        );
    }

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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Moderate),
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
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
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

    /// Regression test: Navigating between windows with arrow keys should not stop playback
    /// This tests the fix for the bug where a playing station would lose its Playing status
    /// when the user navigated to a different window or candidate using arrow keys.
    #[test]
    fn test_playing_candidate_persists_during_cross_window_navigation() {
        let mut model = Model::new();

        // Create two windows with candidates
        let window1_id = 1;
        let window2_id = 2;
        let freq1 = 88_900_000.0;
        let freq2 = 89_100_000.0;
        let candidate1_id = "88.9-1".to_string();
        let candidate2_id = "89.1-2".to_string();

        // Window 1: Create candidate and set to Playing
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Window 2: Create another candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
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
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Moderate),
            signal_strength: Some(40.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Enter selection mode and set up selection on window 2's candidate
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 1 };

        // Verify window 1 candidate is Playing
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(window1.candidates[0].status, CandidateStatus::Playing);

        // Simulate navigating with arrow keys - move up to window 1's candidate
        model.select_previous_candidate();

        // REGRESSION TEST: Window 1 candidate should STILL be Playing after navigation
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "Playing candidate should remain Playing when navigating with arrow keys"
        );
        assert_eq!(window1.candidates[0].completion, 0.8);

        // Navigate back down to window 2
        model.select_next_candidate();

        // Window 1 candidate should STILL be Playing
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "Playing candidate should persist across multiple navigation actions"
        );
    }

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
                metadata: crate::window::WindowMetadata {
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
                    metadata: crate::window::WindowMetadata {
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
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: *freq,
                        window_id,
                    },
                    candidate_id: Some(id.to_string()),
                    audio_quality: Some(crate::audio_quality::AudioQuality::Good),
                    signal_strength: Some(50.0),
                    timestamp: Instant::now(),
                    tuner_id: None,
                });

                model.update(ProgressEvent {
                    event_type: ProgressEventType::AudioPlaybackCompleted,
                    frequency_hz: *freq,
                    metadata: crate::window::WindowMetadata {
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

    // UiMode State Machine Tests

    #[test]
    fn test_ui_mode_transition_idle_to_navigating() {
        let mut model = Model::new();
        assert!(matches!(model.ui_mode, UiMode::Idle));

        // Simulate pressing Up arrow (first navigation)
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        assert!(model.is_navigating());
        assert!(!model.is_idle());
    }

    #[test]
    fn test_ui_mode_transition_navigating_to_awaiting_tune() {
        let mut model = Model::new();
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        // Simulate pressing Enter
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        assert!(model.is_awaiting_tune());
        assert!(!model.is_navigating());
    }

    #[test]
    fn test_ui_mode_transition_awaiting_tune_to_listening() {
        let mut model = Model::new();
        let window_id = 1;
        let candidate_id = "88.9-1".to_string();

        // Setup: Create a candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        // Simulate AudioPlaybackStarted event
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should transition to Listening
        assert!(model.is_listening());
        match &model.ui_mode {
            UiMode::Listening {
                playing_candidate_id,
                ..
            } => {
                assert_eq!(playing_candidate_id, &candidate_id);
            }
            _ => panic!("Expected Listening mode"),
        }
    }

    #[test]
    fn test_ui_mode_transition_listening_to_listening_switch_station() {
        let mut model = Model::new();
        let window_id = 1;

        // Create two candidates
        let candidate1_id = "88.5-1".to_string();
        let candidate2_id = "88.9-1".to_string();

        for (id, freq) in [
            (candidate1_id.clone(), 88_500_000.0),
            (candidate2_id.clone(), 88_900_000.0),
        ] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(id),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Start listening to first station
        model.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: candidate1_id.clone(),
        };

        // Switch to second station while still in Listening mode
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should still be Listening but with new candidate
        assert!(model.is_listening());
        match &model.ui_mode {
            UiMode::Listening {
                playing_candidate_id,
                navigation_index,
                ..
            } => {
                assert_eq!(playing_candidate_id, &candidate2_id);
                assert_eq!(*navigation_index, 0); // Preserves original navigation_index from Listening mode
            }
            _ => panic!("Expected Listening mode"),
        }
    }

    #[test]
    fn test_ui_mode_transition_listening_to_idle() {
        let mut model = Model::new();
        model.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: "88.9-1".to_string(),
        };

        // Simulate exiting browsing mode (Continue scan)
        model.ui_mode = UiMode::Idle;

        assert!(model.is_idle());
        assert!(!model.is_listening());
    }

    #[test]
    fn test_ui_mode_helper_methods() {
        let model_idle = Model::new();
        assert!(model_idle.is_idle());
        assert!(!model_idle.is_interactive());

        let mut model_navigating = Model::new();
        model_navigating.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };
        assert!(model_navigating.is_navigating());
        assert!(model_navigating.is_interactive());

        let mut model_awaiting = Model::new();
        model_awaiting.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };
        assert!(model_awaiting.is_awaiting_tune());
        assert!(model_awaiting.is_interactive());

        let mut model_listening = Model::new();
        model_listening.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: "88.9-1".to_string(),
        };
        assert!(model_listening.is_listening());
        assert!(model_listening.is_interactive());
    }

    #[test]
    fn test_ui_mode_invalid_transitions_prevented() {
        let mut model = Model::new();
        let window_id = 1;
        let candidate_id = "88.9-1".to_string();

        // Create candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // AudioPlaybackStarted in Idle mode - should not transition
        model.ui_mode = UiMode::Idle;

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should still be Idle (transition only happens in AwaitingTune/Listening)
        assert!(model.is_idle());
    }

    #[test]
    fn test_browsing_mode_only_true_when_scan_paused() {
        let mut model = Model::new();
        let window_id = 0;

        // Add a candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some("test-candidate".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Idle mode - browsing_mode should be false
        assert!(model.is_idle());
        assert!(!model.browsing_mode());

        // Enter selection mode (NavigatingScanner) - browsing_mode should still be false
        model.enter_selection_mode();
        assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
        assert!(model.selection_mode());
        assert!(!model.browsing_mode()); // Key assertion: browsing_mode is false while navigating

        // Transition to AwaitingTune - browsing_mode should now be true
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }
        assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
        assert!(model.browsing_mode()); // Now true because scan is paused

        // Transition to Listening - browsing_mode should remain true
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::Listening {
                navigation_index: selected_index,
                playing_index: selected_index,
                playing_candidate_id: "test-candidate".to_string(),
            };
        }
        assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
        assert!(model.browsing_mode()); // Still true when listening
    }

    #[test]
    fn test_enter_key_tunes_to_selected_station() {
        let mut model = Model::new();
        let window_id = 0;
        let candidate_id = "test-candidate".to_string();

        // Add a Signal candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
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
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: Some(0.8),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Start in Idle mode
        assert!(model.is_idle());

        // User presses UP arrow to enter selection mode (NavigatingScanner)
        model.enter_selection_mode();
        assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
        assert!(model.selection_mode());
        assert!(!model.browsing_mode()); // Not in browsing mode yet

        // User presses ENTER - should transition to AwaitingTune
        // This simulates the ENTER key handler logic
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }

        // Verify transition to AwaitingTune
        assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
        assert!(model.browsing_mode()); // Now in browsing mode (scan paused)

        // Verify selected_candidate_info works in AwaitingTune mode
        let info = model.selected_candidate_info();
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.candidate_id, candidate_id);
        assert_eq!(info.candidate_frequency, 88_900_000.0);

        // Simulate receiving AudioPlaybackStarted event
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: Some(0.8),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should transition to Listening mode
        assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
        if let UiMode::Listening {
            playing_candidate_id,
            ..
        } = &model.ui_mode
        {
            assert_eq!(playing_candidate_id, &candidate_id);
        }
    }

    #[test]
    fn test_navigation_and_highlight_separate_in_listening_mode() {
        let mut model = Model::new();
        let window_id = 0;

        // Add three candidates
        for i in 0..3 {
            let freq = 88_100_000.0 + (i as f64 * 200_000.0); // 88.1, 88.3, 88.5 MHz
            let candidate_id = format!("candidate_{}", i);

            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
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
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: Some(0.8),
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Enter selection mode and select first candidate (index 0)
        model.enter_selection_mode();
        assert_eq!(model.selected_candidate_index(), Some(2)); // Most recent

        // Move to first candidate
        model.select_previous_candidate();
        model.select_previous_candidate();
        assert_eq!(model.selected_candidate_index(), Some(0));

        // Press ENTER on first candidate - transition to AwaitingTune
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        // Verify we're tuning to index 0
        if let UiMode::AwaitingTune {
            navigation_index,
            tuning_index,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 0);
            assert_eq!(*tuning_index, 0);
        }

        // Arrow down to navigate to second candidate
        model.select_next_candidate();

        // Verify navigation moved but tuning index stayed the same
        if let UiMode::AwaitingTune {
            navigation_index,
            tuning_index,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 1, "Navigation should move to index 1");
            assert_eq!(*tuning_index, 0, "Tuning should stay at index 0");
        } else {
            panic!("Should still be in AwaitingTune mode");
        }

        // Transition to Listening mode
        model.ui_mode = UiMode::Listening {
            navigation_index: 1,
            playing_index: 0,
            playing_candidate_id: "candidate_0".to_string(),
        };

        // Arrow down again to third candidate
        model.select_next_candidate();

        // Verify navigation moved but playing index stayed the same
        if let UiMode::Listening {
            navigation_index,
            playing_index,
            playing_candidate_id,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 2, "Navigation should move to index 2");
            assert_eq!(*playing_index, 0, "Playing should stay at index 0");
            assert_eq!(playing_candidate_id, "candidate_0");
        } else {
            panic!("Should still be in Listening mode");
        }

        // Arrow up back to second candidate
        model.select_previous_candidate();

        // Verify navigation moved back but playing index still unchanged
        if let UiMode::Listening {
            navigation_index,
            playing_index,
            ..
        } = &model.ui_mode
        {
            assert_eq!(
                *navigation_index, 1,
                "Navigation should move back to index 1"
            );
            assert_eq!(*playing_index, 0, "Playing should still be at index 0");
        }
    }

    #[test]
    fn test_stop_listening_transitions_candidate_to_completed() {
        let mut model = Model::default();
        let window_id = 1;
        let frequency = 88_900_000.0;
        let candidate_id = format!("{:.1}-{}", frequency / 1e6, window_id);

        // Step 1: Create candidate in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 2: Complete audio analysis
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 3: Generate signal
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 4: Pause scanning and enter interactive mode
        model.enter_selection_mode();
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }
        assert!(model.browsing_mode());

        // Step 5: Start playing audio from window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify candidate is in Playing state
        let window = model.windows.get(&window_id).unwrap();
        let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
        let candidate = &window.candidates[*candidate_index];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8);

        // Step 6: Simulate scanning having progressed to window 2 (making window 1 an "old window")
        // This tests the "old window" filtering bug where AudioPlaybackCompleted was rejected
        // In a real scenario, this could happen if scanning resumed briefly or if there are
        // multiple tuners scanning while one is listening
        model.current_window = 2;

        // Verify current_window has advanced to 2
        assert_eq!(model.current_window, 2);

        // Verify we're still in interactive mode
        assert!(model.is_interactive());

        // Step 7: Stop listening to the station from window 1 (now an "old window")
        // Regression test for TWO bugs:
        // 1. AudioPlaybackCompleted was filtered out in interactive mode by should_process_event()
        // 2. AudioPlaybackCompleted was filtered out for old windows by update_candidate()
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id, // window 1 is now "old" since current_window is 2
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify candidate transitioned to Completed state despite being in an old window
        let window = model.windows.get(&window_id).unwrap();
        let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
        let candidate = &window.candidates[*candidate_index];
        assert_eq!(
            candidate.status,
            CandidateStatus::Completed,
            "Candidate should transition to Completed when AudioPlaybackCompleted is sent, \
             even when in interactive mode (bug #1) and from an old window (bug #2)"
        );
        assert_eq!(candidate.completion, 1.0);
    }

    #[test]
    fn test_only_used_tuner_shows_scanning_state() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery service finds RTL-SDR first (alphabetically or by enumeration order)
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        // RTL-SDR should be Available, not Scanning
        assert_eq!(
            model.tuner_states.get(&rtlsdr_tuner.id),
            Some(&TunerState::Available),
            "First discovered tuner should be Available, not auto-set to Scanning"
        );

        // Discovery service then finds SDRplay
        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available
        assert_eq!(
            model.tuner_states.get(&sdrplay_tuner.id),
            Some(&TunerState::Available)
        );
        assert_eq!(
            model.tuner_states.get(&rtlsdr_tuner.id),
            Some(&TunerState::Available)
        );

        // MainThread starts scan with SDRplay - sends ActiveTunersUpdated event
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // SDRplay should now be Scanning
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "SDRplay should be Scanning when MainThread allocated it for scanning"
        );

        // RTL-SDR should still be Available (regression test for incorrect auto-scanning)
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not in active tuners"
        );

        // Scan continues - active tuners remain unchanged
        // Progress events no longer affect tuner state

        // SDRplay should still be Scanning
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

        // RTL-SDR should STILL be Available
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should never transition to Scanning since it's not in active tuners"
        );
    }

    #[test]
    fn test_only_used_tuner_shows_listening_state() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds both tuners
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available initially
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread starts scan with SDRplay
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // SDRplay is now Scanning
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

        // User presses Enter to tune to the candidate - MainThread moves tuner to listening
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![],
                vec![sdrplay_tuner.id.clone()],
            ),
        });

        // SDRplay should transition to Listening
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Listening,
            "SDRplay should be Listening when MainThread allocated it to listening"
        );

        // RTL-SDR should still be Available (regression test for incorrect listening state)
        // The bug was: update_candidate() set self.tuners.first() to Listening
        // instead of using event.tuner_id
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not in active tuners"
        );

        // Stop listening doesn't change active tuners
        // (MainThread would send new ActiveTunersUpdated when user presses Escape)
        // For this test, we're just verifying state stays as-is

        // SDRplay remains in Listening state (still allocated to listening)
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Listening,
            "SDRplay remains Listening until MainThread reallocates it"
        );

        // RTL-SDR should STILL be Available throughout
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should never transition to Listening since it's not in active tuners"
        );
    }

    #[test]
    fn test_tuner_stays_scanning_during_automatic_audio_playback() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds SDRplay tuner
        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Should be Available initially
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread allocates SDRplay for scanning
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // SDRplay is now Scanning
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should be Scanning when MainThread allocated it for scanning"
        );

        // Model is still in Idle mode (not AwaitingTune) - user has NOT pressed Enter
        assert!(matches!(model.ui_mode, UiMode::Idle));

        // During scanning, audio playback starts automatically for quality analysis
        // Even though audio is playing, MainThread keeps the tuner in scanning list
        // because user has not pressed Enter (no TuneToCandidate command sent)

        // MainThread continues to report tuner as scanning during automatic playback
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // The tuner should remain in Scanning state during automatic audio playback
        // Only when user presses Enter (sends TuneToCandidate) should it go to Listening
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should remain Scanning during automatic audio playback (user has not pressed Enter)"
        );

        // Audio playback completes automatically, tuner still scanning
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // Should still be Scanning after automatic playback completes
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should remain Scanning after automatic audio playback completes"
        );
    }

    #[test]
    fn test_correct_tuner_shows_scanning_when_returning_from_listening() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds both tuners (RTL-SDR first, SDRplay second)
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available initially
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread allocates SDRplay for scanning (not RTL-SDR)
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // SDRplay should be Scanning, RTL-SDR should remain Available
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

        // User presses Enter to listen - MainThread moves SDRplay to listening list
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![],
                vec![sdrplay_tuner.id.clone()],
            ),
        });

        // SDRplay should be Listening, RTL-SDR should remain Available
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Listening);
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

        // User presses Escape to go back to scanning
        // MainThread moves SDRplay back to scanning list
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            status: create_test_pool_status(
                vec![sdrplay_tuner.id.clone()],
                vec![sdrplay_tuner.id.clone()],
                vec![],
            ),
        });

        // After returning from listening to scanning, only SDRplay should be Scanning
        // RTL-SDR should remain Available (never used)
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not being used"
        );

        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "SDRplay should transition back to Scanning when MainThread returns it to scanning list"
        );

        // Verify that exactly one tuner is in Scanning state by checking active_tuners
        if let Some(ref status) = model.pool_status {
            assert_eq!(
                status
                    .tuners
                    .iter()
                    .filter(|t| t.activity == Some(crate::pool::TunerActivity::Scanning))
                    .count(),
                1,
                "Exactly one tuner should be in scanning list"
            );
            assert_eq!(
                status
                    .tuners
                    .iter()
                    .find(|t| t.activity == Some(crate::pool::TunerActivity::Scanning))
                    .unwrap()
                    .id
                    .device_id
                    .clone(),
                sdrplay_tuner.id,
                "Only SDRplay should be in scanning list"
            );
        } else {
            panic!("pool_status should be set");
        }
    }
}
