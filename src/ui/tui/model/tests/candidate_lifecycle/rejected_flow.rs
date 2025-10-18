use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

/// Test that rejected candidates reach terminal state correctly
#[test]
fn test_rejected_candidate_lifecycle() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Step 1: Candidate created
    ctx.update_candidate(frequency, window_id, CandidateState::Detected, None, None);
    ctx.sync();

    // Step 2: Audio analysis started
    ctx.update_candidate(frequency, window_id, CandidateState::Analyzing, None, None);
    ctx.sync();

    // Step 3: Candidate rejected (noise)
    ctx.update_candidate(frequency, window_id, CandidateState::Rejected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Rejected);
    assert_eq!(candidate.completion, 1.0); // 100% - terminal state
}
