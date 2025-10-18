use super::super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::{CandidateStatus, UiMode};

#[test]
fn test_enter_key_tunes_to_selected_station() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;
    let candidate_id = format!("{:.1}-{}", 88_900_000.0 / 1e6, window_id);

    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Signal,
        None,
        Some(0.8),
    );
    ctx.sync();

    assert!(ctx.model.is_idle());

    ctx.model.enter_selection_mode();
    assert!(matches!(
        ctx.model.ui_mode,
        UiMode::NavigatingScanner { .. }
    ));
    assert!(ctx.model.selection_mode());
    assert!(!ctx.model.browsing_mode());

    if let Some(selected_index) = ctx.model.selected_candidate_index() {
        ctx.model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }

    assert!(matches!(ctx.model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(ctx.model.browsing_mode());

    let info = ctx.model.selected_candidate_info();
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.candidate_id, candidate_id);
    assert_eq!(info.candidate_frequency, 88_900_000.0);

    ctx.update_candidate(
        88_900_000.0,
        window_id,
        CandidateState::Playing,
        None,
        Some(0.8),
    );
    ctx.sync();

    assert!(matches!(ctx.model.ui_mode, UiMode::Listening { .. }));
    if let UiMode::Listening {
        playing_candidate_id,
        ..
    } = &ctx.model.ui_mode
    {
        assert_eq!(playing_candidate_id, &candidate_id);
    }
}

#[test]
fn test_stop_listening_transitions_candidate_to_completed() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let frequency = 88_900_000.0;
    let candidate_id = format!("{:.1}-{}", frequency / 1e6, window_id);

    ctx.update_candidate(frequency, window_id, CandidateState::Detected, None, None);
    ctx.sync();

    ctx.update_candidate(
        frequency,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    ctx.model.enter_selection_mode();
    if let Some(selected_index) = ctx.model.selected_candidate_index() {
        ctx.model.ui_mode = UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };
    }
    assert!(ctx.model.browsing_mode());

    ctx.update_candidate(frequency, window_id, CandidateState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
    let candidate = &window.candidates[*candidate_index];
    assert_eq!(candidate.status, CandidateStatus::Playing);
    assert_eq!(candidate.completion, 0.8);

    ctx.model.current_window = 2;

    assert_eq!(ctx.model.current_window, 2);

    assert!(ctx.model.is_interactive());

    ctx.update_candidate(
        frequency,
        window_id,
        CandidateState::Completed,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
    let candidate = &window.candidates[*candidate_index];
    assert_eq!(
        candidate.status,
        CandidateStatus::Completed,
        "Candidate should transition to Completed when AudioPlaybackCompleted is sent, \
             even when in interactive mode (bug #1) and from an old window (bug #2)"
    );
    assert_eq!(candidate.completion, 1.0);
}
