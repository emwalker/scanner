use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{
    audio::quality::AudioQuality,
    ui::tui::model::{AnalysisStatus, PlaybackState},
};

#[test]
fn test_playing_signals_remain_playing_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window_id = 1;
    let freq = 88_900_000.0;

    ctx.update_signal(freq, window_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq,
        window_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_signal(freq, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Playing);

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Playing);
    assert_eq!(signal.completion, 0.8);
}

#[test]
fn test_playing_signals_remain_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window1_id = 1;
    let window2_id = 2;
    let freq1 = 88_900_000.0;
    let freq2 = 89_100_000.0;

    ctx.update_signal(freq1, window1_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq1,
        window1_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_signal(freq1, window1_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window1_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Playing);

    ctx.update_signal(freq2, window2_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq2,
        window2_id,
        TestSignalState::Signal,
        Some(AudioQuality::Moderate),
        Some(40.0),
    );
    ctx.sync();

    ctx.model.current_window = window1_id;

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window1_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Playing);
    assert_eq!(signal.completion, 0.8);

    let window = ctx.model.windows.get(&window2_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);
    assert_eq!(signal.completion, 0.6);
}

#[test]
fn test_signal_signals_remain_signal_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window_id = 1;
    let freq = 88_900_000.0;

    ctx.update_signal(freq, window_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq,
        window_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);
    assert_eq!(signal.completion, 0.6);
}
