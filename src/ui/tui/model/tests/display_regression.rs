use super::helpers::{ModelTestContext, TestSignalState};
use crate::{
    audio::quality::AudioQuality,
    ui::tui::model::{AnalysisStatus, PlaybackState},
};

#[test]
fn test_playing_signal_displays_signal_status_not_detected() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;
    let frequency_hz = 88.9e6;

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Signal,
        Some(AudioQuality::Good),
        Some(0.8),
    );
    ctx.sync();

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Playing,
        Some(AudioQuality::Good),
        Some(0.8),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];

    assert_eq!(
        signal.status,
        AnalysisStatus::Signal,
        "BUG REGRESSION: Playing signal showed 'Detected' instead of 'Signal'"
    );
    assert_eq!(
        signal.playback_state,
        PlaybackState::Playing,
        "Candidate should be in Playing state"
    );
}

#[test]
fn test_playing_signal_displays_audio_quality_not_blank() {
    let mut ctx = ModelTestContext::new();
    let window_id = 2;
    let frequency_hz = 89.3e6;

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Signal,
        Some(AudioQuality::Moderate),
        Some(0.65),
    );
    ctx.sync();

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Playing,
        Some(AudioQuality::Moderate),
        Some(0.65),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];

    assert_eq!(
        signal.audio_quality,
        Some(AudioQuality::Moderate),
        "BUG REGRESSION: Playing signal showed blank audio quality instead of 'Moderate'"
    );
    assert_eq!(
        signal.playback_state,
        PlaybackState::Playing,
        "Candidate should be in Playing state"
    );
}

#[test]
fn test_playing_signal_has_both_analysis_and_playback_state() {
    let mut ctx = ModelTestContext::new();
    let window_id = 3;
    let frequency_hz = 90.1e6;

    ctx.update_signal(
        frequency_hz,
        window_id,
        TestSignalState::Playing,
        Some(AudioQuality::Good),
        Some(0.75),
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let signal = &window.signals[0];

    assert_eq!(
        signal.status,
        AnalysisStatus::Signal,
        "Playing signal must have completed analysis (Signal status)"
    );
    assert_eq!(
        signal.playback_state,
        PlaybackState::Playing,
        "Candidate must be in Playing state"
    );
    assert!(
        signal.audio_quality.is_some(),
        "Playing signal must have audio quality data"
    );
    assert_eq!(
        signal.signal_strength,
        Some(0.75),
        "Playing signal must have signal strength"
    );
}
