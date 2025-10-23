use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::ui::tui::model::{AnalysisStatus, PlaybackState};

#[test]
fn test_audio_analysis_completed_preserves_signal() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    ctx.update_signal(frequency, window_id, TestSignalState::Detected, None, None);
    ctx.update_signal(frequency, window_id, TestSignalState::Analyzing, None, None);
    ctx.update_signal(frequency, window_id, TestSignalState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);
    assert_eq!(signal.completion, 0.6);

    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.status, AnalysisStatus::Signal);
    assert_eq!(signal.completion, 0.6);
}

#[test]
fn test_status_text_mapping_unchanged() {
    // Analysis status strings
    assert_eq!(AnalysisStatus::Detected.to_string(), "Detected");
    assert_eq!(AnalysisStatus::Analyzing.to_string(), "Analyzing");
    assert_eq!(AnalysisStatus::Rejected.to_string(), "Rejected");
    assert_eq!(AnalysisStatus::Signal.to_string(), "Signal");
    assert_eq!(AnalysisStatus::Error.to_string(), "Error");

    // Playback state strings
    assert_eq!(PlaybackState::NotPlaying.to_string(), "");
    assert_eq!(PlaybackState::Playing.to_string(), "Playing");
    assert_eq!(PlaybackState::Completed.to_string(), "Completed");
}

#[test]
fn test_progress_percentages_unchanged() {
    let mut ctx = ModelTestContext::new();
    let frequency = 88_900_000.0;
    let window_id = 1;

    ctx.update_signal(frequency, window_id, TestSignalState::Detected, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.completion, 0.3);

    // Note: Analyzing is a transient internal state that can't be simulated without thread handles,
    // so we skip it and go directly to Signal
    ctx.update_signal(frequency, window_id, TestSignalState::Signal, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.completion, 0.6);

    ctx.update_signal(frequency, window_id, TestSignalState::Playing, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.completion, 0.8);

    ctx.update_signal(frequency, window_id, TestSignalState::Completed, None, None);
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];
    assert_eq!(signal.completion, 1.0);

    let rejected_freq = 89_100_000.0;
    ctx.update_signal(
        rejected_freq,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.update_signal(
        rejected_freq,
        window_id,
        TestSignalState::Rejected,
        None,
        None,
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let rejected = &window.signals[1];
    assert_eq!(rejected.completion, 1.0);
}
