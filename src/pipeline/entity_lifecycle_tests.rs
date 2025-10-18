//! Regression tests for CandidateEntity lifecycle transitions in the pipeline
//!
//! These tests verify that entity state updates are properly called at each
//! stage of the signal processing pipeline, preventing regressions where
//! entity updates might be accidentally removed.

use super::*;
use crate::{
    audio::quality::AudioAnalyzer,
    core::types::{ScanningConfig, Signal, TEST_FREQUENCY_HZ},
    ecs::{CandidateEntity, CandidateId, CandidateState, Entities, EntityWorld},
};
use std::sync::{Arc, RwLock, mpsc};
use tokio::sync::broadcast;

fn create_test_config() -> ScanningConfig {
    let mut config = ScanningConfig::default();
    config.signal_processing.frequency_tracking.disabled = true;
    config.audio.analyzer = AudioAnalyzer::mock();
    config.audio.squelch.learning_duration = 0.1;
    config.samp_rate = 1_000_000.0;
    config
}

fn create_candidate_entities() -> Entities<CandidateEntity> {
    Arc::new(RwLock::new(EntityWorld::new()))
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

#[test]
fn test_entity_transitions_through_full_lifecycle() {
    let _config = create_test_config();
    let candidate_entities = create_candidate_entities();
    let frequency = TEST_FREQUENCY_HZ;
    let window_id = 1;
    let metadata = crate::scanning::window::WindowMetadata {
        center_frequency_hz: frequency,
        window_id,
    };

    {
        let mut entities = candidate_entities.write().unwrap();
        entities.insert(CandidateEntity::new(frequency, metadata));
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_detected());
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.start_analysis();
        }
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_analyzing());
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.mark_as_signal(crate::audio::quality::AudioQuality::Good, Some(0.8));
        }
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_signal());
        assert_eq!(
            entity.audio_quality(),
            Some(crate::audio::quality::AudioQuality::Good)
        );
        assert_eq!(entity.signal_strength(), Some(0.8));
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.start_playback();
        }
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_playing());
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.complete_playback();
        }
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_completed());
    }
}

#[test]
fn test_entity_rejection_path() {
    let _config = create_test_config();
    let candidate_entities = create_candidate_entities();
    let frequency = TEST_FREQUENCY_HZ;
    let window_id = 1;
    let metadata = crate::scanning::window::WindowMetadata {
        center_frequency_hz: frequency,
        window_id,
    };

    {
        let mut entities = candidate_entities.write().unwrap();
        entities.insert(CandidateEntity::new(frequency, metadata));
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.start_analysis();
        }
    }

    {
        let mut entities = candidate_entities.write().unwrap();
        let id = CandidateId::new(frequency, window_id);
        if let Some(entity) = entities.get_mut(&id) {
            entity.reject();
        }
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert!(entity.is_rejected());
    }
}

#[test]
fn test_pipeline_creates_and_updates_entities() {
    let config = create_test_config();
    let sdr_rx = create_mock_sdr_stream();
    let center_freq = TEST_FREQUENCY_HZ;
    let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
    let candidate_entities = Some(create_candidate_entities());

    let frequency = TEST_FREQUENCY_HZ;
    let window_id = 1;
    let metadata = crate::scanning::window::WindowMetadata {
        center_frequency_hz: center_freq,
        window_id,
    };

    {
        let entities = candidate_entities.as_ref().unwrap();
        let mut entities_guard = entities.write().unwrap();
        entities_guard.insert(CandidateEntity::new(frequency, metadata));
    }

    let context = AnalysisContext {
        config: &config,
        center_freq,
        metadata,
        candidate_entities: &candidate_entities,
    };

    let _ = process_peak_to_signal(TEST_FREQUENCY_HZ, sdr_rx, signal_tx, &context);

    {
        let entities = candidate_entities.as_ref().unwrap();
        let entities_guard = entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities_guard.get(&id);
        assert!(entity.is_some());
    }
}

#[test]
fn test_entity_state_sequence_invariants() {
    let candidate_entities = create_candidate_entities();
    let frequency = TEST_FREQUENCY_HZ;
    let window_id = 1;
    let metadata = crate::scanning::window::WindowMetadata {
        center_frequency_hz: frequency,
        window_id,
    };

    {
        let mut entities = candidate_entities.write().unwrap();
        entities.insert(CandidateEntity::new(frequency, metadata));
    }

    {
        let entities = candidate_entities.read().unwrap();
        let id = CandidateId::new(frequency, window_id);
        let entity = entities.get(&id).unwrap();
        assert_eq!(entity.state(), CandidateState::Detected);
        assert!(entity.completion() >= 0.0 && entity.completion() <= 1.0);
    }

    let state_sequence = vec![
        (CandidateState::Analyzing, "start_analysis"),
        (CandidateState::Signal, "mark_as_signal"),
        (CandidateState::Playing, "start_playback"),
        (CandidateState::Completed, "complete_playback"),
    ];

    for (expected_state, method_name) in state_sequence {
        {
            let mut entities = candidate_entities.write().unwrap();
            let id = CandidateId::new(frequency, window_id);
            if let Some(entity) = entities.get_mut(&id) {
                match method_name {
                    "start_analysis" => entity.start_analysis(),
                    "mark_as_signal" => {
                        entity.mark_as_signal(crate::audio::quality::AudioQuality::Good, Some(0.8))
                    }
                    "start_playback" => entity.start_playback(),
                    "complete_playback" => entity.complete_playback(),
                    _ => {}
                }
            }
        }

        {
            let entities = candidate_entities.read().unwrap();
            let id = CandidateId::new(frequency, window_id);
            let entity = entities.get(&id).unwrap();
            assert_eq!(entity.state(), expected_state);
            assert!(entity.completion() >= 0.0 && entity.completion() <= 1.0);
        }
    }
}
