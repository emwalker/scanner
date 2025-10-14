use crate::ui::tui::model::{CandidateStatus, Model, UiMode};
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;

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
