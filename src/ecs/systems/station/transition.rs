//! Tune transition system - coordinates multi-step playback transitions

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::station::TuneState,
        queue::TunerRequest,
        system::{System, SystemContext},
    },
};

/// System that coordinates tune transitions
///
/// This system processes stations in Transitioning state and enqueues
/// tuner requests for them.
pub struct TransitionSystem;

impl Default for TransitionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for TransitionSystem {
    fn name(&self) -> &'static str {
        "TuneTransition"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        // Don't process station transitions during global pause
        if context.is_globally_paused() {
            return Ok(());
        }

        let (signal_entities, tuner_request_queue) =
            match (&context.signal_entities, &context.tuner_request_queue) {
                (Some(se), Some(queue)) => (se.clone(), queue.clone()),
                _ => return Ok(()),
            };

        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        let _tasks_opt = context
            .task_entities
            .as_ref()
            .and_then(|te| te.try_read().ok());

        for signal in signals.iter_mut() {
            // Process signals in Transitioning state
            if let TuneState::Transitioning(transition) = &signal.tune_state {
                let window_id = transition.window_id.clone();
                let center_freq = transition.center_frequency;

                // Enqueue tuner request (station_id field used for backwards compatibility)
                let request = TunerRequest {
                    station_id: crate::ecs::StationId::new(), // Placeholder during dual-write
                    frequency: signal.frequency(),
                    window_id,
                    center_frequency: center_freq,
                };

                if let Ok(mut queue) = tuner_request_queue.try_lock() {
                    queue.push_back(request);
                    debug!(
                        signal_id = ?signal.id(),
                        signal_frequency_mhz = signal.frequency() / 1e6,
                        window_frequency_mhz = center_freq / 1e6,
                        queue_length = queue.len(),
                        "TuneTransition: Enqueued tuner request"
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, RwLock},
        time::SystemTime,
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, Signal},
        ecs::{EntityWorld, StationEntity, TaskId, components::window::WindowId},
    };

    fn create_test_signal(frequency: f64) -> Signal {
        Signal {
            frequency_hz: frequency,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: frequency,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_system_with_no_entities() {
        let mut system = TransitionSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transition_enqueues_tuner_request() {
        let mut system = TransitionSystem::new();

        let signal = create_test_signal(88.9e6);
        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        let station = StationEntity::from_signal(&signal, window_id.clone());

        let mut station_world = EntityWorld::new();
        station_world.insert(station);

        // Add SignalEntity in Transitioning state - this is what the TransitionSystem processes
        let mut signal_world = EntityWorld::new();
        let mut signal_entity = crate::ecs::SignalEntity::new(88.9e6, window_id.clone());
        signal_entity.tune_state = TuneState::transitioning(window_id, 88.9e6);
        signal_world.insert(signal_entity);

        let _station_entities = Arc::new(RwLock::new(station_world));
        let signal_entities = Arc::new(RwLock::new(signal_world));
        let tuner_request_queue =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities)
            .with_tuner_request_queue(tuner_request_queue.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(queue.len(), 1, "Should have 1 request in queue");
    }
}
