use crate::ui::tui::model::{CandidateStatus, Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
        metadata: crate::scanning::window::WindowMetadata {
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
        metadata: crate::scanning::window::WindowMetadata {
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

    // Create candidate in window 2 (this marks window 1 as complete)
    let window2_id = 2;
    let freq2 = 89_300_000.0;
    let candidate2_id = "89.3-2".to_string();

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

    // Verify we're now in window 2
    assert_eq!(model.current_window, window2_id);
    assert!(model.windows.get(&window1_id).unwrap().is_complete);

    // In normal scanning mode, events to old windows should be blocked
    model.update(ProgressEvent {
        event_type: ProgressEventType::AudioAnalysisStarted,
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

    // Status should still be Signal (event was blocked)
    let window1 = model.windows.get(&window1_id).unwrap();
    assert_eq!(window1.candidates[0].status, CandidateStatus::Signal);

    // Now enter browsing mode by transitioning to Navigating mode
    model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    // Send AudioPlaybackStarted for the old window candidate
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

    // Window 2: Create another candidate
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

// UiMode State Machine Tests
