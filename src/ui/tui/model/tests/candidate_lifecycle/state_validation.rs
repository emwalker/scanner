use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

/// Test that no candidates remain stuck in intermediate states
#[test]
fn test_no_stuck_intermediate_states() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create multiple candidates in different states
    let candidates = vec![
        ("88.1-1", 88_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("88.5-1", 88_500_000.0),
        ("88.7-1", 88_700_000.0),
        ("88.9-1", 88_900_000.0),
    ];

    // Create all candidates and start analysis
    for (_id, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Analyzing, None, None);
    }
    ctx.sync();

    // Resolve first two to Rejected
    ctx.update_candidate(
        candidates[0].1,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );
    ctx.update_candidate(
        candidates[1].1,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    // Complete signal paths for others
    for (_id, freq) in &candidates[2..] {
        ctx.update_candidate(*freq, window_id, CandidateState::Signal, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Playing, None, None);
        ctx.update_candidate(*freq, window_id, CandidateState::Completed, None, None);
    }
    ctx.sync();

    // Verify no candidates are stuck in intermediate states
    let window = ctx.model.windows.get(&window_id).unwrap();
    for candidate in &window.candidates {
        match candidate.status {
            CandidateStatus::Detected | CandidateStatus::Analyzing => {
                panic!(
                    "Candidate at {:.1} MHz stuck in intermediate state: {:?}",
                    candidate.frequency_hz / 1e6,
                    candidate.status
                );
            }
            CandidateStatus::Rejected | CandidateStatus::Completed => {
                // Terminal states are good
                assert_eq!(candidate.completion, 1.0);
            }
            CandidateStatus::Signal | CandidateStatus::Playing => {
                // These are valid but should have progressed to Completed
                panic!(
                    "Candidate at {:.1} MHz should have completed: {:?}",
                    candidate.frequency_hz / 1e6,
                    candidate.status
                );
            }
        }
    }
}
