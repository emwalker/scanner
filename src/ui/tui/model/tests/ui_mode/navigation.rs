use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::UiMode;

#[test]
fn test_navigation_and_highlight_separate_in_listening_mode() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;

    for i in 0..3 {
        let freq = 88_100_000.0 + (i as f64 * 200_000.0);
        ctx.update_candidate(freq, window_id, CandidateState::Detected, None, None);
        ctx.update_candidate(freq, window_id, CandidateState::Signal, None, Some(0.8));
    }
    ctx.sync();

    ctx.model.enter_selection_mode();
    assert_eq!(ctx.model.selected_candidate_index(), Some(2));

    ctx.model.select_previous_candidate();
    ctx.model.select_previous_candidate();
    assert_eq!(ctx.model.selected_candidate_index(), Some(0));

    ctx.model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 0,
        tuning_index: 0,
    };

    if let UiMode::AwaitingTune {
        navigation_index,
        tuning_index,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*navigation_index, 0);
        assert_eq!(*tuning_index, 0);
    }

    ctx.model.select_next_candidate();

    if let UiMode::AwaitingTune {
        navigation_index,
        tuning_index,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*navigation_index, 1, "Navigation should move to index 1");
        assert_eq!(*tuning_index, 0, "Tuning should stay at index 0");
    } else {
        panic!("Should still be in AwaitingTune mode");
    }

    ctx.model.ui_mode = UiMode::Listening {
        navigation_index: 1,
        playing_index: 0,
        playing_candidate_id: "candidate_0".to_string(),
    };

    ctx.model.select_next_candidate();

    if let UiMode::Listening {
        navigation_index,
        playing_index,
        playing_candidate_id,
    } = &ctx.model.ui_mode
    {
        assert_eq!(*navigation_index, 2, "Navigation should move to index 2");
        assert_eq!(*playing_index, 0, "Playing should stay at index 0");
        assert_eq!(playing_candidate_id, "candidate_0");
    } else {
        panic!("Should still be in Listening mode");
    }

    ctx.model.select_previous_candidate();

    if let UiMode::Listening {
        navigation_index,
        playing_index,
        ..
    } = &ctx.model.ui_mode
    {
        assert_eq!(
            *navigation_index, 1,
            "Navigation should move back to index 1"
        );
        assert_eq!(*playing_index, 0, "Playing should still be at index 0");
    }
}
