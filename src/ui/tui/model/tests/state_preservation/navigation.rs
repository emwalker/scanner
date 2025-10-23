use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{
    audio::quality::AudioQuality,
    ui::tui::model::{PlaybackState, UiMode},
};

#[test]
fn test_playing_signal_persists_during_cross_window_navigation() {
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

    ctx.update_signal(freq2, window2_id, TestSignalState::Detected, None, None);
    ctx.update_signal(
        freq2,
        window2_id,
        TestSignalState::Signal,
        Some(AudioQuality::Moderate),
        Some(40.0),
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 1,
        window_id: window2_id,
    };

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(window1.signals[0].playback_state, PlaybackState::Playing);

    ctx.model.select_previous_signal();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.signals[0].playback_state,
        PlaybackState::Playing,
        "Playing signal should remain Playing when navigating with arrow keys"
    );
    assert_eq!(window1.signals[0].completion, 0.8);

    ctx.model.select_next_signal();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.signals[0].playback_state,
        PlaybackState::Playing,
        "Playing signal should persist across multiple navigation actions"
    );
}
