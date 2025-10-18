use super::super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::{CandidateStatus, UiMode};

#[test]
fn test_browsing_mode_playing_correct_candidate() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let freq1 = 88_500_000.0;
    let freq2 = 88_900_000.0;
    let freq3 = 89_300_000.0;

    for freq in [freq1, freq2, freq3] {
        ctx.update_candidate(freq, window_id, CandidateState::Detected, None, None);
        ctx.update_candidate(
            freq,
            window_id,
            CandidateState::Signal,
            Some(AudioQuality::Good),
            Some(50.0),
        );
    }
    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(window.candidates.len(), 3);
    assert_eq!(window.candidates[0].frequency_hz, freq1);
    assert_eq!(window.candidates[1].frequency_hz, freq2);
    assert_eq!(window.candidates[2].frequency_hz, freq3);
    assert_eq!(window.candidates[0].status, CandidateStatus::Signal);
    assert_eq!(window.candidates[1].status, CandidateStatus::Signal);
    assert_eq!(window.candidates[2].status, CandidateStatus::Signal);

    ctx.model.ui_mode = UiMode::AwaitingTune {
        navigation_index: 1,
        tuning_index: 1,
    };

    ctx.update_candidate(freq2, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(
        window.candidates[0].status,
        CandidateStatus::Signal,
        "First candidate should still be Signal"
    );
    assert_eq!(
        window.candidates[1].status,
        CandidateStatus::Playing,
        "Second candidate should be Playing"
    );
    assert_eq!(
        window.candidates[2].status,
        CandidateStatus::Signal,
        "Third candidate should still be Signal"
    );

    ctx.model.ui_mode = UiMode::NavigatingScanner { selected_index: 2 };

    ctx.update_candidate(freq3, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(
        window.candidates[0].status,
        CandidateStatus::Signal,
        "First candidate should still be Signal"
    );
    assert_eq!(
        window.candidates[1].status,
        CandidateStatus::Completed,
        "Second candidate should be Completed (was replaced)"
    );
    assert_eq!(
        window.candidates[2].status,
        CandidateStatus::Playing,
        "Third candidate should be Playing"
    );
}

#[test]
fn test_browsing_mode_allows_old_window_playback() {
    let mut ctx = ModelTestContext::new();

    let window1_id = 1;
    let freq1 = 88_900_000.0;

    ctx.update_candidate(freq1, window1_id, CandidateState::Detected, None, None);
    ctx.update_candidate(
        freq1,
        window1_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    let window2_id = 2;
    let freq2 = 89_300_000.0;

    ctx.update_candidate(freq2, window2_id, CandidateState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, window2_id);
    assert!(ctx.model.windows.get(&window1_id).unwrap().is_complete);

    ctx.update_candidate(freq1, window1_id, CandidateState::Analyzing, None, None);
    ctx.sync();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(window1.candidates[0].status, CandidateStatus::Signal);

    ctx.model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

    ctx.update_candidate(freq1, window1_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window1 = ctx.model.windows.get(&window1_id).unwrap();
    assert_eq!(
        window1.candidates[0].status,
        CandidateStatus::Playing,
        "AudioPlaybackStarted should work for old windows in browsing mode"
    );
}
