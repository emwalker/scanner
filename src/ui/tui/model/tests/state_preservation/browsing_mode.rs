use crate::ui::tui::model::{CandidateStatus, Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

/// Test quit functionality

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
