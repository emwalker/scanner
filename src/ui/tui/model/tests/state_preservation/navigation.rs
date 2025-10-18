use super::super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::{CandidateStatus, UiMode};

#[test]
fn test_playing_candidate_persists_during_cross_window_navigation() {
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

    ctx.update_candidate(freq2, window2_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq2,
        window2_id,
        CandidateState::Signal,
        Some(AudioQuality::Moderate),
        Some(40.0),
    );
    ctx.sync();

    ctx.model.ui_mode = UiMode::NavigatingScanner { selected_index: 1 };

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(window1.candidates[0].status, CandidateStatus::Playing);

    ctx.model.select_previous_candidate();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.candidates[0].status,
        CandidateStatus::Playing,
        "Playing candidate should remain Playing when navigating with arrow keys"
    );
    assert_eq!(window1.candidates[0].completion, 0.8);

    ctx.model.select_next_candidate();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.candidates[0].status,
        CandidateStatus::Playing,
        "Playing candidate should persist across multiple navigation actions"
    );
}
