use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::{AnalysisStatus, PlaybackState};

/// Test that signals progress through all expected states
#[test]
fn test_complete_signal_lifecycle() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    // Step 1: Candidate created (Detected state)
    ctx.update_signal(frequency, window_id, TestSignalState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Detected);
    assert_eq!(signal.completion, 0.3); // 30%

    // Step 2: Signal generated (Signal state)
    // Note: Analyzing is a transient internal state that we can't simulate without thread handles,
    // so it transitions directly to Signal in tests
    ctx.update_signal(frequency, window_id, TestSignalState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);
    assert_eq!(signal.completion, 0.6); // 60%

    // Step 3: Audio playback started (Playing state)
    ctx.update_signal(frequency, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Playing);
    assert_eq!(signal.completion, 0.8); // 80%

    // Step 4: Audio playback completed (Completed state)
    ctx.update_signal(frequency, window_id, TestSignalState::Completed, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.playback_state, PlaybackState::Completed);
    assert_eq!(signal.completion, 1.0); // 100%
}
