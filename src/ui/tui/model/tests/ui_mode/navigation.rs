use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{ecs::components::signal::SignalId, ui::tui::model::UiMode};

#[test]
fn test_navigation_and_highlight_separate_in_listening_mode() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;

    for i in 0..3 {
        let freq = 88_100_000.0 + (i as f64 * 200_000.0);
        ctx.update_signal(freq, window_id, TestSignalState::Detected, None, None);
        ctx.update_signal(freq, window_id, TestSignalState::Signal, None, Some(0.8));
    }
    ctx.sync();

    ctx.model.enter_selection_mode();
    assert_eq!(ctx.model.selected_signal_index(), Some(2));

    ctx.model.select_previous_signal();
    ctx.model.select_previous_signal();
    assert_eq!(ctx.model.selected_signal_index(), Some(0));

    ctx.model.ui_mode = UiMode::AwaitingTune {
        signal_index: 0,
        window_id: 0,
        tuning_signal_id: SignalId::from_string("signal_0".to_string()),
    };

    if let UiMode::AwaitingTune {
        signal_index,
        window_id: mode_window_id,
        tuning_signal_id,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*signal_index, 0);
        assert_eq!(*mode_window_id, 0);
        assert_eq!(
            tuning_signal_id,
            &SignalId::from_string("signal_0".to_string())
        );
    }

    ctx.model.select_next_signal();

    if let UiMode::AwaitingTune {
        signal_index,
        window_id: mode_window_id,
        tuning_signal_id,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*signal_index, 1, "Navigation should move to index 1");
        assert_eq!(*mode_window_id, 0, "Window ID should stay at 0");
        assert_eq!(
            tuning_signal_id,
            &SignalId::from_string("signal_0".to_string()),
            "Tuning signal should stay the same"
        );
    } else {
        panic!("Should still be in AwaitingTune mode");
    }

    ctx.model.ui_mode = UiMode::Listening {
        signal_index: 1,
        window_id: 0,
        playing_signal_id: SignalId::from_string("signal_0".to_string()),
    };

    ctx.model.select_next_signal();

    if let UiMode::Listening {
        signal_index,
        window_id: mode_window_id,
        playing_signal_id,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*signal_index, 2, "Navigation should move to index 2");
        assert_eq!(*mode_window_id, 0, "Window ID should stay at 0");
        assert_eq!(
            playing_signal_id,
            &SignalId::from_string("signal_0".to_string())
        );
    } else {
        panic!("Should still be in Listening mode");
    }

    ctx.model.select_previous_signal();

    if let UiMode::Listening {
        signal_index,
        window_id: mode_window_id,
        ..
    } = &ctx.model.ui_mode
    {
        assert_eq!(*signal_index, 1, "Navigation should move back to index 1");
        assert_eq!(*mode_window_id, 0, "Window ID should still be at 0");
    }
}
