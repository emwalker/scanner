use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{
    audio::quality::AudioQuality,
    ui::tui::model::{PlaybackState, UiMode},
};

#[test]
fn test_enter_key_tunes_to_selected_station() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;
    let signal_id = format!("{:.1}-test-task-{}", 88_900_000.0 / 1e6, window_id);

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Signal,
        None,
        Some(0.8),
    );
    ctx.sync();

    assert!(ctx.model.is_idle());

    ctx.model.enter_selection_mode();
    assert!(matches!(
        ctx.model.ui_mode,
        UiMode::NavigatingScanner { .. }
    ));
    assert!(ctx.model.selection_mode());
    assert!(!ctx.model.browsing_mode());

    if let Some(selected_index) = ctx.model.selected_signal_index() {
        ctx.model.ui_mode = UiMode::AwaitingTune {
            signal_index: selected_index,
            window_id,
            tuning_signal_id: signal_id.clone(),
        };
    }

    assert!(matches!(ctx.model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(ctx.model.browsing_mode());

    let info = ctx.model.selected_signal_info();
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.signal_id, signal_id);
    assert_eq!(info.signal_frequency, 88_900_000.0);

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Playing,
        None,
        Some(0.8),
    );
    ctx.sync();

    assert!(matches!(ctx.model.ui_mode, UiMode::Listening { .. }));
    if let UiMode::Listening {
        playing_signal_id, ..
    } = &ctx.model.ui_mode
    {
        assert_eq!(playing_signal_id, &signal_id);
    }
}

#[test]
fn test_stop_listening_transitions_signal_to_completed() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let frequency = 88_900_000.0;

    ctx.update_signal(frequency, window_id, TestSignalState::Detected, None, None);
    ctx.sync();

    ctx.update_signal(
        frequency,
        window_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal_id = window
        .signals
        .first()
        .map(|c| c.signal_id.clone())
        .expect("Should have at least one signal");

    ctx.model.enter_selection_mode();
    if let Some(selected_index) = ctx.model.selected_signal_index() {
        ctx.model.ui_mode = UiMode::AwaitingTune {
            signal_index: selected_index,
            window_id,
            tuning_signal_id: signal_id.clone(),
        };
    }
    assert!(ctx.model.browsing_mode());

    ctx.update_signal(frequency, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal_index = window.signal_lookup.get(&signal_id).unwrap();
    let signal = &window.signals[*signal_index];
    assert_eq!(signal.playback_state, PlaybackState::Playing);
    assert_eq!(signal.completion, 0.8);

    ctx.model.current_window = 2;

    assert_eq!(ctx.model.current_window, 2);

    assert!(ctx.model.is_interactive());

    ctx.update_signal(
        frequency,
        window_id,
        TestSignalState::Completed,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal_index = window.signal_lookup.get(&signal_id).unwrap();
    let signal = &window.signals[*signal_index];
    assert_eq!(
        signal.playback_state,
        PlaybackState::Completed,
        "Candidate should transition to Completed when AudioPlaybackCompleted is sent, even when \
         in interactive mode (bug #1) and from an old window (bug #2)"
    );
    assert_eq!(signal.completion, 1.0);
}
