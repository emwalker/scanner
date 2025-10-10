use super::*;
use crate::{
    audio::quality::AudioAnalyzer,
    core::types::{ScanningConfig, Signal, TEST_FREQUENCY_HZ},
    ui::{MockProgressReporter, NoOpProgressReporter, ProgressEventType},
};
use std::sync::mpsc;
use tokio::sync::broadcast;

fn create_test_config() -> ScanningConfig {
    let mut config = ScanningConfig::default();
    config.signal_processing.frequency_tracking.disabled = true;
    config.audio.analyzer = AudioAnalyzer::mock();
    config.audio.squelch.learning_duration = 0.1;
    config.samp_rate = 1_000_000.0;
    config
}

fn create_mock_sdr_stream() -> broadcast::Receiver<crate::broadcast::SamplePacket> {
    let (tx, rx) = broadcast::channel(100);

    for _ in 0..100 {
        let samples: Vec<_> = (0..1024)
            .map(|_| rustradio::Complex::new(0.1, 0.1))
            .collect();
        let packet = crate::broadcast::SamplePacket::new(samples);
        let _ = tx.send(packet);
    }

    rx
}

fn create_mock_strong_sdr_stream() -> broadcast::Receiver<crate::broadcast::SamplePacket> {
    let (tx, rx) = broadcast::channel(100);

    for _ in 0..100 {
        let samples: Vec<_> = (0..1024)
            .map(|_| rustradio::Complex::new(0.8, 0.8))
            .collect();
        let packet = crate::broadcast::SamplePacket::new(samples);
        let _ = tx.send(packet);
    }

    rx
}

#[test]
fn test_weak_peak_exits_early() {
    let config = create_test_config();
    let sdr_rx = create_mock_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, signal_rx) = mpsc::sync_channel::<Signal>(10);
    let progress_reporter = NoOpProgressReporter;

    let context = AnalysisContext {
        config: &config,
        center_freq,
        progress_reporter: std::sync::Arc::new(progress_reporter),
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ, sdr_rx, signal_tx, &context);

    assert!(result.is_ok(), "Weak peak processing should succeed");

    assert!(
        signal_rx.try_recv().is_err(),
        "Weak peak should not produce signals"
    );
}

#[test]
fn test_strong_peak_produces_signal() {
    let config = create_test_config();
    let sdr_rx = create_mock_strong_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
    let progress_reporter = MockProgressReporter::new();
    let progress_arc = std::sync::Arc::new(progress_reporter.clone());

    let context = AnalysisContext {
        config: &config,
        center_freq,

        progress_reporter: progress_arc,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ + 100_000.0, sdr_rx, signal_tx, &context);

    assert!(result.is_ok(), "Strong peak processing should succeed");

    assert!(
        progress_reporter.event_count() > 0,
        "Should have progress events for strong signal"
    );
}

#[test]
fn test_progress_events_emitted() {
    let config = create_test_config();
    let sdr_rx = create_mock_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
    let progress_reporter = MockProgressReporter::new();
    let progress_arc = std::sync::Arc::new(progress_reporter.clone());

    let context = AnalysisContext {
        config: &config,
        center_freq,

        progress_reporter: progress_arc,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ, sdr_rx, signal_tx, &context);

    assert!(result.is_ok(), "Pipeline should complete successfully");

    let events = progress_reporter.events();
    assert!(
        !events.is_empty(),
        "Should emit at least one progress event"
    );

    let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();

    assert!(
        event_types
            .iter()
            .any(|t| matches!(t, ProgressEventType::PeakDetected)),
        "Should emit PeakDetected event"
    );

    for event in &events {
        assert_eq!(
            event.frequency_hz, TEST_FREQUENCY_HZ,
            "All events should have correct frequency"
        );
    }
}

#[test]
fn test_pipeline_with_frequency_tracking_disabled() {
    let config = create_test_config();
    let sdr_rx = create_mock_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
    let progress_reporter = MockProgressReporter::new();
    let progress_arc = std::sync::Arc::new(progress_reporter.clone());

    let context = AnalysisContext {
        config: &config,
        center_freq,

        progress_reporter: progress_arc,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ + 50_000.0, sdr_rx, signal_tx, &context);

    assert!(
        result.is_ok(),
        "Pipeline should handle disabled frequency tracking"
    );

    let events = progress_reporter.events();
    assert!(!events.is_empty(), "Should have progress events");

    assert!(
        matches!(
            events[0].event_type,
            crate::ui::ProgressEventType::PeakDetected
        ),
        "First event should be PeakDetected"
    );
}
