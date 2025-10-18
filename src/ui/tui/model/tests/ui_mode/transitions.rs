use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::UiMode;

#[test]
fn test_ui_mode_transition_idle_to_navigating() {
    let ctx = ModelTestContext::new();
    assert!(matches!(ctx.model.ui_mode, UiMode::Idle));

    let mut model = ctx.model;
    model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    assert!(model.is_navigating());
    assert!(!model.is_idle());
}

#[test]
fn test_ui_mode_transition_navigating_to_awaiting_tune() {
    let ctx = ModelTestContext::new();
    let mut model = ctx.model;
    model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    assert!(model.is_awaiting_tune());
    assert!(!model.is_navigating());
}

#[test]
fn test_ui_mode_transition_awaiting_tune_to_listening() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let candidate_id = "88.9-1".to_string();

    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    ctx.update_candidate(88_900_000.0, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    assert!(ctx.model.is_listening());
    match &ctx.model.ui_mode {
        UiMode::Listening {
            playing_candidate_id,
            ..
        } => {
            assert_eq!(playing_candidate_id, &candidate_id);
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_listening_switch_station() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let candidate1_id = "88.5-1".to_string();
    let candidate2_id = "88.9-1".to_string();

    ctx.update_candidate(
        88_500_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: candidate1_id.clone(),
    };

    ctx.update_candidate(88_900_000.0, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    assert!(ctx.model.is_listening());
    match &ctx.model.ui_mode {
        UiMode::Listening {
            playing_candidate_id,
            navigation_index,
            ..
        } => {
            assert_eq!(playing_candidate_id, &candidate2_id);
            assert_eq!(*navigation_index, 0);
        }
        _ => panic!("Expected Listening mode"),
    }
}

#[test]
fn test_ui_mode_transition_listening_to_idle() {
    let ctx = ModelTestContext::new();
    let mut model = ctx.model;
    model.ui_mode = UiMode::Listening {
        navigation_index: 0,
        playing_index: 0,
        playing_candidate_id: "88.9-1".to_string(),
    };

    model.ui_mode = UiMode::Idle;

    assert!(model.is_idle());
    assert!(!model.is_listening());
}
