use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

#[test]
fn test_sequential_window_completion() {
    let mut ctx = ModelTestContext::new();

    ctx.update_candidate(88_900_000.0, 1, CandidateState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 1);
    assert!(!ctx.model.windows.get(&1).unwrap().is_complete);

    ctx.update_candidate(89_100_000.0, 2, CandidateState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 2);
    assert!(ctx.model.windows.get(&1).unwrap().is_complete);
    assert!(!ctx.model.windows.get(&2).unwrap().is_complete);

    ctx.update_candidate(89_300_000.0, 3, CandidateState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 3);
    assert!(ctx.model.windows.get(&1).unwrap().is_complete);
    assert!(ctx.model.windows.get(&2).unwrap().is_complete);
    assert!(!ctx.model.windows.get(&3).unwrap().is_complete);
}

#[test]
fn test_old_window_events_ignored() {
    let mut ctx = ModelTestContext::new();

    ctx.update_candidate(88_900_000.0, 1, CandidateState::Detected, None, None);
    ctx.sync();

    ctx.update_candidate(89_100_000.0, 2, CandidateState::Detected, None, None);
    ctx.sync();

    let window1_candidate_count = ctx.model.windows.get(&1).unwrap().candidates.len();

    ctx.update_candidate(88_700_000.0, 1, CandidateState::Detected, None, None);
    ctx.sync();

    assert_eq!(
        ctx.model.windows.get(&1).unwrap().candidates.len(),
        window1_candidate_count
    );

    ctx.update_candidate(88_900_000.0, 1, CandidateState::Analyzing, None, None);
    ctx.sync();

    let candidate = &ctx.model.windows.get(&1).unwrap().candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Detected);
    assert_eq!(candidate.completion, 0.3);
}
