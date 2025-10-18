use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;

#[test]
fn test_model_utility_functions() {
    let mut ctx = ModelTestContext::new();

    assert!(ctx.model.is_empty());
    assert!(!ctx.model.all_complete());
    assert_eq!(ctx.model.candidate_count(), 0);

    let window_id = 1;
    let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

    for (_, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);
    }
    ctx.sync();

    assert!(!ctx.model.is_empty());
    assert!(!ctx.model.all_complete());
    assert_eq!(ctx.model.candidate_count(), 2);

    for (_, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Rejected, None, None);
    }
    ctx.sync();

    assert!(!ctx.model.is_empty());
    ctx.model.total_windows = Some(1);
    assert!(ctx.model.all_complete());
    assert_eq!(ctx.model.candidate_count(), 2);
}
