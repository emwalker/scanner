use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::{AnalysisStatus, PlaybackState};

/// Test that no signals remain stuck in intermediate states
#[test]
fn test_no_stuck_intermediate_states() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create multiple signals in different states
    let signals = vec![
        ("88.1-1", 88_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("88.5-1", 88_500_000.0),
        ("88.7-1", 88_700_000.0),
        ("88.9-1", 88_900_000.0),
    ];

    // Create all signals and start analysis
    for (_id, freq) in &signals {
        ctx.update_signal(*freq, window_id, TestSignalState::Detected, None, None);
        ctx.update_signal(*freq, window_id, TestSignalState::Analyzing, None, None);
    }
    ctx.sync();

    // Resolve first two to Rejected
    ctx.update_signal(
        signals[0].1,
        window_id,
        TestSignalState::Rejected,
        None,
        None,
    );
    ctx.update_signal(
        signals[1].1,
        window_id,
        TestSignalState::Rejected,
        None,
        None,
    );

    // Complete signal paths for others
    for (_id, freq) in &signals[2..] {
        ctx.update_signal(*freq, window_id, TestSignalState::Signal, None, None);
        ctx.update_signal(*freq, window_id, TestSignalState::Playing, None, None);
        ctx.update_signal(*freq, window_id, TestSignalState::Completed, None, None);
    }
    ctx.sync();

    // Verify no signals are stuck in intermediate states
    let window = ctx.model.windows.get(&window_id).unwrap();
    for signal in &window.signals {
        // Check analysis state
        match signal.status {
            AnalysisStatus::Detected | AnalysisStatus::Analyzing => {
                panic!(
                    "signal at {:.1} MHz stuck in intermediate analysis state: {:?}",
                    signal.frequency_hz / 1e6,
                    signal.status
                );
            }
            AnalysisStatus::Signal | AnalysisStatus::Rejected | AnalysisStatus::Error => {
                // Terminal analysis states are good
                assert_eq!(signal.completion, 1.0);
            }
        }

        // Check playback state - if analysis completed, playback should also complete
        if matches!(signal.status, AnalysisStatus::Signal)
            && signal.playback_state == PlaybackState::Playing
        {
            panic!(
                "signal at {:.1} MHz has Signal status but still Playing (should be Completed): \
                 analysis={:?}, playback={:?}",
                signal.frequency_hz / 1e6,
                signal.status,
                signal.playback_state
            );
        }
    }
}
