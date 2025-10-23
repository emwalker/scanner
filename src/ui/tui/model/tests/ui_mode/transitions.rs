use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::UiMode;

#[test]
fn test_ui_mode_transition_idle_to_navigating() {
    let ctx = ModelTestContext::new();
    assert!(matches!(ctx.model.ui_mode, UiMode::Idle));

    let mut model = ctx.model;
    model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 0,
        window_id: 0,
    };

    assert!(model.is_navigating());
    assert!(!model.is_idle());
}

#[test]
fn test_ui_mode_transition_navigating_to_awaiting_tune() {
    let ctx = ModelTestContext::new();
    let mut model = ctx.model;
    model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 0,
        window_id: 0,
    };

    model.ui_mode = UiMode::AwaitingTune {
        signal_index: 0,
        window_id: 0,
        tuning_signal_id: "88.9-0".to_string(),
    };

    assert!(model.is_awaiting_tune());
    assert!(!model.is_navigating());
}

#[test]
fn test_ui_mode_transition_awaiting_tune_to_listening() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let signal_id = "88.9-test-task-1".to_string();

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::AwaitingTune {
        signal_index: 0,
        window_id,
        tuning_signal_id: signal_id.clone(),
    };

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Playing,
        None,
        None,
    );
    ctx.sync();

    assert!(ctx.model.is_listening());
    match &ctx.model.ui_mode {
        UiMode::Listening {
            playing_signal_id, ..
        } => {
            assert_eq!(playing_signal_id, &signal_id);
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_listening_switch_station() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let signal1_id = "88.5-test-task-1".to_string();
    let signal2_id = "88.9-test-task-1".to_string();

    ctx.update_signal(
        88_500_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::Listening {
        signal_index: 0,
        window_id,
        playing_signal_id: signal1_id.clone(),
    };

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Playing,
        None,
        None,
    );
    ctx.sync();

    assert!(ctx.model.is_listening());
    match &ctx.model.ui_mode {
        UiMode::Listening {
            playing_signal_id,
            signal_index,
            ..
        } => {
            assert_eq!(playing_signal_id, &signal2_id);
            assert_eq!(*signal_index, 0);
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_idle() {
    let ctx = ModelTestContext::new();
    let mut model = ctx.model;
    model.ui_mode = UiMode::Listening {
        signal_index: 0,
        window_id: 1,
        playing_signal_id: "88.9-1".to_string(),
    };

    model.ui_mode = UiMode::Idle;

    assert!(model.is_idle());
    assert!(!model.is_listening());
}
