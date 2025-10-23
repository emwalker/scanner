//! Tune request system - converts tune transitions to queued requests

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::station::{TuneAllocationComponent, TuneRequestComponent, TuneState},
        queue::TunerRequest,
        system::{System, SystemContext},
    },
};

pub struct TuneRequestSystem;

impl TuneRequestSystem {
    pub fn new() -> Self {
        TuneRequestSystem
    }
}

impl System for TuneRequestSystem {
    fn name(&self) -> &'static str {
        "TuneRequestSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let tuner_request_queue = match &context.tuner_request_queue {
            Some(queue) => queue.clone(),
            None => return Ok(()),
        };

        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        for signal in signals.iter_mut() {
            if let TuneState::Transitioning(transition) = &signal.tune_state {
                let window_id = transition.window_id.clone();
                let center_frequency = transition.center_frequency;
                let signal_frequency = signal.frequency();

                // Create components
                let request = TuneRequestComponent::new(window_id.clone());
                let allocation = TuneAllocationComponent::new();

                // Enqueue request
                let tuner_request = TunerRequest {
                    station_id: crate::ecs::StationId::new(), // Placeholder during dual-write
                    frequency: signal_frequency,
                    window_id,
                    center_frequency,
                };

                if let Ok(mut queue) = tuner_request_queue.lock() {
                    queue.push_back(tuner_request);
                    debug!(
                        signal_id = ?signal.id(),
                        frequency_mhz = signal_frequency / 1e6,
                        "TuneRequestSystem: Enqueued tuner request"
                    );
                } else {
                    continue; // Lock poisoned, skip this signal
                }

                // Transition to RequestQueued
                signal.tune_state = TuneState::RequestQueued {
                    request,
                    allocation,
                };
                debug!(
                    signal_id = ?signal.id(),
                    "TuneRequestSystem: Transitioned to RequestQueued"
                );
            }
        }

        Ok(())
    }
}

impl Default for TuneRequestSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = TuneRequestSystem::new();
        assert_eq!(system.name(), "TuneRequestSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = TuneRequestSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_runs_with_idle_station() {
        use std::{
            sync::{Arc, RwLock},
            time::SystemTime,
        };

        use crate::{
            audio::quality::AudioQuality,
            core::types::{ModulationType, Signal},
            ecs::{EntityWorld, StationEntity, TaskId, components::window::WindowId},
        };

        let mut system = TuneRequestSystem::new();
        let mut context = SystemContext::new();

        let signal = Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: AudioQuality::Good,
        };

        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 1);

        let station = StationEntity::from_signal(&signal, window_id);
        let mut station_world = EntityWorld::new();
        station_world.insert(station);
        let entities = Arc::new(RwLock::new(station_world));

        context = context.with_tuner_request_queue(Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::new(),
        )));

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Idle station should not be processed
        let stations = entities.read().unwrap();
        assert!(stations.iter().all(|s| s.tune_state.is_idle()));
    }
}
