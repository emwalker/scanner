use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::UiMode;

#[test]
fn test_ui_mode_helper_methods() {
    let model_idle = ModelTestContext::new().model;
    assert!(model_idle.is_idle());
    assert!(!model_idle.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::NavigatingScanner {
        signal_index: 0,
        window_id: 0,
    };
    assert!(ctx.model.is_navigating());
    assert!(ctx.model.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::AwaitingTune {
        signal_index: 0,
        window_id: 0,
        tuning_signal_id: "88.9-0".to_string(),
    };
    assert!(ctx.model.is_awaiting_tune());
    assert!(ctx.model.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::Listening {
        signal_index: 0,
        window_id: 1,
        playing_signal_id: "88.9-1".to_string(),
    };
    assert!(ctx.model.is_listening());
    assert!(ctx.model.is_interactive());
}

#[test]
fn test_ui_mode_invalid_transitions_prevented() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::Idle;

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Playing,
        None,
        None,
    );
    ctx.sync();

    assert!(ctx.model.is_idle());
}
