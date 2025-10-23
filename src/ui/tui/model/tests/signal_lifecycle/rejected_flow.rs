use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::AnalysisStatus;

/// Test that rejected signals reach terminal state correctly
#[test]
fn test_rejected_signal_lifecycle() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Step 1: Candidate created
    ctx.update_signal(frequency, window_id, TestSignalState::Detected, None, None);
    ctx.sync();

    // Step 2: Audio analysis started
    ctx.update_signal(frequency, window_id, TestSignalState::Analyzing, None, None);
    ctx.sync();

    // Step 3: Candidate rejected (noise)
    ctx.update_signal(frequency, window_id, TestSignalState::Rejected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Rejected);
    assert_eq!(signal.completion, 1.0); // 100% - terminal state
}
