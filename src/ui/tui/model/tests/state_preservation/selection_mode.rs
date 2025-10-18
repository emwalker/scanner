use super::super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

#[test]
fn test_playing_candidates_remain_playing_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window_id = 1;
    let freq = 88_900_000.0;

    ctx.update_candidate(freq, window_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(freq, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);
}

#[test]
fn test_playing_candidates_remain_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window1_id = 1;
    let window2_id = 2;
    let freq1 = 88_900_000.0;
    let freq2 = 89_100_000.0;

    ctx.update_candidate(freq1, window1_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq1,
        window1_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(freq1, window1_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window1_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);

    ctx.update_candidate(freq2, window2_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq2,
        window2_id,
        CandidateState::Signal,
        Some(AudioQuality::Moderate),
        Some(40.0),
    );
    ctx.sync();

    ctx.model.current_window = window1_id;

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window1_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);

    let window = ctx.model.windows.get(&window2_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);
}

#[test]
fn test_signal_candidates_remain_signal_when_entering_selection_mode() {
    let mut ctx = ModelTestContext::new();

    let window_id = 1;
    let freq = 88_900_000.0;

    ctx.update_candidate(freq, window_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);

    ctx.model.enter_selection_mode();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate = &window.candidates[0];
    assert_eq!(candidate.status, CandidateStatus::Signal);
    assert_eq!(candidate.completion, 0.6);
}
