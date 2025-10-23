use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{
    audio::quality::AudioQuality,
    ui::tui::model::{AnalysisStatus, PlaybackState, UiMode},
};

#[test]
fn test_browsing_mode_playing_correct_signal() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let freq1 = 88_500_000.0;
    let freq2 = 88_900_000.0;
    let freq3 = 89_300_000.0;

    for freq in [freq1, freq2, freq3] {
        ctx.update_signal(freq, window_id, TestSignalState::Detected, None, None);
        ctx.update_signal(
            freq,
            window_id,
            TestSignalState::Signal,
            Some(AudioQuality::Good),
            Some(50.0),
        );
    }
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(window.signals.len(), 3);
    assert_eq!(window.signals[0].frequency_hz, freq1);
    assert_eq!(window.signals[1].frequency_hz, freq2);
    assert_eq!(window.signals[2].frequency_hz, freq3);
    assert_eq!(window.signals[0].status, AnalysisStatus::Signal);
    assert_eq!(window.signals[1].status, AnalysisStatus::Signal);
    assert_eq!(window.signals[2].status, AnalysisStatus::Signal);

    let signal_id = ctx.model.windows.get(&window_id).unwrap().signals[1]
        .signal_id
        .clone();
    ctx.model.ui_mode = UiMode::AwaitingTune {
        signal_index: 1,
        window_id,
        tuning_signal_id: signal_id,
    };

    ctx.update_signal(freq2, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(
        window.signals[0].status,
        AnalysisStatus::Signal,
        "First signal should still be Signal"
    );
    assert_eq!(
        window.signals[1].playback_state,
        PlaybackState::Playing,
        "Second signal should be Playing"
    );
    assert_eq!(
        window.signals[2].status,
        AnalysisStatus::Signal,
        "Third signal should still be Signal"
    );

    ctx.model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 2,
        window_id,
    };

    ctx.update_signal(freq3, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(
        window.signals[0].status,
        AnalysisStatus::Signal,
        "First signal should still be Signal"
    );
    assert_eq!(
        window.signals[1].playback_state,
        PlaybackState::Completed,
        "Second signal should be Completed (was replaced)"
    );
    assert_eq!(
        window.signals[2].playback_state,
        PlaybackState::Playing,
        "Third signal should be Playing"
    );
}

#[test]
fn test_browsing_mode_allows_old_window_playback() {
    let mut ctx = ModelTestContext::new();

    let window1_id = 1;
    let freq1 = 88_900_000.0;

    ctx.update_signal(freq1, window1_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq1,
        window1_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    let window2_id = 2;
    let freq2 = 89_300_000.0;

    ctx.update_signal(freq2, window2_id, TestSignalState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, window2_id);
    assert!(ctx.model.windows.get(&window1_id).unwrap().is_complete);

    ctx.update_signal(freq1, window1_id, TestSignalState::Analyzing, None, None);
    ctx.sync();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(window1.signals[0].status, AnalysisStatus::Signal);

    ctx.model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 0,
        window_id: window1_id,
    };

    ctx.update_signal(freq1, window1_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.signals[0].playback_state,
        PlaybackState::Playing,
        "AudioPlaybackStarted should work for old windows in browsing mode"
    );
}
