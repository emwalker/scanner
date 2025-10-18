use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

#[test]
fn test_window_candidate_filtering() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let candidates = vec![
        ("88.1-1", 88_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("88.5-1", 88_500_000.0),
    ];

    for (_, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);
    }
    ctx.sync();

    ctx.update_candidate(
        candidates[0].1,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    for (_, freq) in &candidates[1..] {
        ctx.update_candidate(*freq, window_id, CandidateState::Signal, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Playing, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Completed, None, None);
    }
    ctx.sync();

    ctx.update_candidate(89_100_000.0, 2, CandidateState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);

    let current_displayable = window.displayable_candidates(true, false);
    assert_eq!(current_displayable.len(), 2);

    let completed_displayable = window.displayable_candidates(false, false);
    assert_eq!(completed_displayable.len(), 2);

    for candidate in current_displayable {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
    for candidate in completed_displayable {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}

#[test]
fn test_window_display_logic() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

    for (_, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Rejected, None, None);
    }
    ctx.sync();

    ctx.model.total_windows = Some(2);
    ctx.update_candidate(89_100_000.0, 2, CandidateState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);
    assert!(!window.should_display());
}
