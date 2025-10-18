use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

/// Test that candidates progress through all expected states
#[test]
fn test_complete_candidate_lifecycle() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Step 1: Candidate created (Detected state)
    ctx.update_candidate(frequency, window_id, CandidateState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Detected);
    assert_eq!(candidate.completion, 0.3); // 30%

    // Step 2: Audio analysis started (Analyzing state)
    ctx.update_candidate(frequency, window_id, CandidateState::Analyzing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Analyzing);
    assert_eq!(candidate.completion, 0.5); // 50%

    // Step 3: Signal generated (Signal state)
    ctx.update_candidate(frequency, window_id, CandidateState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6); // 60%

    // Step 4: Audio playback started (Playing state)
    ctx.update_candidate(frequency, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8); // 80%

    // Step 5: Audio playback completed (Completed state)
    ctx.update_candidate(frequency, window_id, CandidateState::Completed, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Completed);
    assert_eq!(candidate.completion, 1.0); // 100%
}
