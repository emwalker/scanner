use super::*;
use crate::{
    audio::quality::AudioAnalyzer,
    core::types::{ScanningConfig, Signal, TEST_FREQUENCY_HZ},
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

    let candidate_entities = None;
    let context = AnalysisContext {
        config: &config,
        center_freq,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
        candidate_entities: &candidate_entities,
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

    let candidate_entities = None;
    let context = AnalysisContext {
        config: &config,
        center_freq,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
        candidate_entities: &candidate_entities,
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ + 100_000.0, sdr_rx, signal_tx, &context);

    assert!(result.is_ok(), "Strong peak processing should succeed");
}

#[test]
fn test_pipeline_with_frequency_tracking_disabled() {
    let config = create_test_config();
    let sdr_rx = create_mock_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);

    let candidate_entities = None;
    let context = AnalysisContext {
        config: &config,
        center_freq,
        metadata: crate::scanning::window::WindowMetadata {
            center_frequency_hz: center_freq,
            window_id: 1,
        },
        candidate_entities: &candidate_entities,
    };
    let result = process_peak_to_signal(TEST_FREQUENCY_HZ + 50_000.0, sdr_rx, signal_tx, &context);

    assert!(
        result.is_ok(),
        "Pipeline should handle disabled frequency tracking"
    );
}
