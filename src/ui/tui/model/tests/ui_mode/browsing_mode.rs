use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::UiMode;

#[test]
fn test_browsing_mode_only_true_when_scan_paused() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;

    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.sync();

    assert!(ctx.model.is_idle());
    assert!(!ctx.model.browsing_mode());

    ctx.model.enter_selection_mode();
    assert!(matches!(
        ctx.model.ui_mode,
        UiMode::NavigatingScanner { .. }
    ));
    assert!(ctx.model.selection_mode());
    assert!(!ctx.model.browsing_mode());

    if let Some(selected_index) = ctx.model.selected_candidate_index() {
        ctx.model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }
    assert!(matches!(ctx.model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(ctx.model.browsing_mode());

    if let Some(selected_index) = ctx.model.selected_candidate_index() {
        ctx.model.ui_mode = UiMode::Listening {
            navigation_index: selected_index,
            playing_index: selected_index,
            playing_candidate_id: "test-candidate".to_string(),
        };
    }
    assert!(matches!(ctx.model.ui_mode, UiMode::Listening { .. }));
    assert!(ctx.model.browsing_mode());
}
