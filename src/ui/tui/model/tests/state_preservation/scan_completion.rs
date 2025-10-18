use super::super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::CandidateStatus;

#[test]
fn test_rejected_candidates_disappear_when_scan_completes() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let candidates = vec![
        ("88.1-1", 88_100_000.0, false),
        ("88.3-1", 88_300_000.0, true),
        ("88.5-1", 88_500_000.0, false),
        ("88.7-1", 88_700_000.0, true),
    ];

    for (_, freq, is_rejected) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);

        if *is_rejected {
            ctx.update_candidate(*freq, window_id, CandidateState::Rejected, None, None);
        } else {
            ctx.update_candidate(
                *freq,
                window_id,
                CandidateState::Signal,
                Some(AudioQuality::Good),
                Some(50.0),
            );
            ctx.update_candidate(*freq, window_id, CandidateState::Completed, None, None);
        }
    }
    ctx.sync();

    assert_eq!(
        ctx.model.windows.get(&window_id).unwrap().candidates.len(),
        4
    );

    ctx.model.total_windows = Some(1);

    assert_eq!(ctx.model.current_window, 1);
    assert!(
        ctx.model.all_candidates_complete(),
        "all_candidates_complete should be true"
    );
    assert!(ctx.model.all_complete(), "all_complete should be true");

    if let Some(window) = ctx.model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert!(window.is_complete);

    let displayable_after_complete = window.displayable_candidates(true, false);
    assert_eq!(displayable_after_complete.len(), 2);

    for candidate in displayable_after_complete {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }

    let displayable_in_selection = window.displayable_candidates(true, true);
    assert_eq!(displayable_in_selection.len(), 2);

    for candidate in displayable_in_selection {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}
