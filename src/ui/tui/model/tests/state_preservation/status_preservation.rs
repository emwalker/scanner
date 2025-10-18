use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

#[test]
fn test_audio_analysis_completed_preserves_signal() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    ctx.update_candidate(frequency, window_id, CandidateState::Detected, None, None);
    ctx.update_candidate(frequency, window_id, CandidateState::Analyzing, None, None);
    ctx.update_candidate(frequency, window_id, CandidateState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);

    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);
}

#[test]
fn test_status_text_mapping_unchanged() {
    assert_eq!(CandidateStatus::Detected.to_string(), "DETECTED");
    assert_eq!(CandidateStatus::Analyzing.to_string(), "ANALYZING");
    assert_eq!(CandidateStatus::Rejected.to_string(), "NOISE");
    assert_eq!(CandidateStatus::Signal.to_string(), "SIGNAL");
    assert_eq!(CandidateStatus::Playing.to_string(), "PLAYING");
    assert_eq!(CandidateStatus::Completed.to_string(), "DONE");
}

#[test]
fn test_progress_percentages_unchanged() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    ctx.update_candidate(frequency, window_id, CandidateState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.completion, 0.3);

    ctx.update_candidate(frequency, window_id, CandidateState::Analyzing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.completion, 0.5);

    ctx.update_candidate(frequency, window_id, CandidateState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.completion, 0.6);

    ctx.update_candidate(frequency, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.completion, 0.8);

    ctx.update_candidate(frequency, window_id, CandidateState::Completed, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.completion, 1.0);

    let rejected_freq = 89_100_000.0;
    ctx.update_candidate(
        rejected_freq,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        rejected_freq,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let rejected_candidate = &window.candidates[1];
    assert_eq!(rejected_candidate.completion, 1.0);
}
