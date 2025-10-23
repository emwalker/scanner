use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::AnalysisStatus;

#[test]
fn test_sequential_window_completion() {
    let mut ctx = ModelTestContext::new();

    ctx.update_signal(88_900_000.0, 1, TestSignalState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 1);
    assert!(!ctx.model.windows.get(&1).unwrap().is_complete);

    ctx.update_signal(89_100_000.0, 2, TestSignalState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 2);
    assert!(ctx.model.windows.get(&1).unwrap().is_complete);
    assert!(!ctx.model.windows.get(&2).unwrap().is_complete);

    ctx.update_signal(89_300_000.0, 3, TestSignalState::Detected, None, None);
    ctx.sync();

    assert_eq!(ctx.model.current_window, 3);
    assert!(ctx.model.windows.get(&1).unwrap().is_complete);
    assert!(ctx.model.windows.get(&2).unwrap().is_complete);
    assert!(!ctx.model.windows.get(&3).unwrap().is_complete);
}

#[test]
fn test_old_window_events_ignored() {
    let mut ctx = ModelTestContext::new();

    ctx.update_signal(88_900_000.0, 1, TestSignalState::Detected, None, None);
    ctx.sync();

    ctx.update_signal(89_100_000.0, 2, TestSignalState::Detected, None, None);
    ctx.sync();

    let window1_signal_count = ctx.model.windows.get(&1).unwrap().signals.len();

    ctx.update_signal(88_700_000.0, 1, TestSignalState::Detected, None, None);
    ctx.sync();

    assert_eq!(
        ctx.model.windows.get(&1).unwrap().signals.len(),
        window1_signal_count
    );

    ctx.update_signal(88_900_000.0, 1, TestSignalState::Analyzing, None, None);
    ctx.sync();

    let signal = &ctx.model.windows.get(&1).unwrap().signals[0];
    assert_eq!(signal.status, AnalysisStatus::Detected);
    assert_eq!(signal.completion, 0.3);
}
