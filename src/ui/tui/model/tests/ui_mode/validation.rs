use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::UiMode;

#[test]
fn test_ui_mode_helper_methods() {
    let model_idle = ModelTestContext::new().model;
    assert!(model_idle.is_idle());
    assert!(!model_idle.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };
    assert!(ctx.model.is_navigating());
    assert!(ctx.model.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };
    assert!(ctx.model.is_awaiting_tune());
    assert!(ctx.model.is_interactive());

    let mut ctx = ModelTestContext::new();
    ctx.model.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: "88.9-1".to_string(),
    };
    assert!(ctx.model.is_listening());
    assert!(ctx.model.is_interactive());
}

#[test]
fn test_ui_mode_invalid_transitions_prevented() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let _candidate_id = "88.9-1".to_string();

    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::Idle;

    ctx.update_candidate(88_900_000.0, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    assert!(ctx.model.is_idle());
}
