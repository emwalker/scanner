//! Tuner allocation system - acquires tuners and spawns audio entities

use tracing::debug;

use crate::{
    core::types::{ModulationType, Result, Signal},
    ecs::{
        Entity,
        components::station::{TuneAllocationState, TuneState},
        queue::TunerRequest,
        system::{System, SystemContext},
    },
    hardware::pool::SegmentTrait,
    scanning::window::spawn_audio_entity,
};

pub struct TunerAllocationSystem;

impl TunerAllocationSystem {
    pub fn new() -> Self {
        TunerAllocationSystem
    }

    fn process_request(request: &TunerRequest, context: &mut SystemContext) -> bool {
        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return true,
        };

        let pool = match &context.pool {
            Some(p) => p.clone(),
            None => return true,
        };

        let config = match &context.config {
            Some(c) => c.clone(),
            None => return true,
        };

        let shutdown_coordinator = match &context.shutdown_coordinator {
            Some(s) => s.clone(),
            None => return true,
        };

        // Find signal and verify it's in RequestQueued state
        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return false,
        };

        const FREQ_TOLERANCE_HZ: f64 = 1000.0;
        let signal = match signals
            .iter_mut()
            .find(|s| (s.frequency() - request.frequency).abs() < FREQ_TOLERANCE_HZ)
        {
            Some(s) => s,
            None => {
                debug!(
                    frequency_mhz = request.frequency / 1e6,
                    "TunerAllocationSystem: Signal not found"
                );
                return true; // Signal gone, remove from queue
            }
        };

        if !matches!(signal.tune_state, TuneState::RequestQueued { .. }) {
            debug!(
                signal_id = ?signal.id(),
                "TunerAllocationSystem: Signal not in RequestQueued state"
            );
            return true; // Wrong state, remove from queue
        }

        // Try to acquire tuner
        let requirements = crate::hardware::pool::TaskRequirements {
            frequency_hz: request.center_frequency,
            bandwidth_hz: config.samp_rate,
            required_sample_rate: config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let tuner = match pool.acquire(
            &requirements,
            crate::hardware::pool::TunerActivity::Listening,
        ) {
            Ok(t) => {
                debug!(
                    tuner_id = ?t.id(),
                    station_id = ?request.station_id,
                    "TunerAllocationSystem: Acquired tuner"
                );
                t
            }
            Err(_) => {
                return false; // Tuner unavailable, retry later
            }
        };

        // Create segment
        let segment = match crate::hardware::pool::Segment::from_tuner(
            tuner,
            request.center_frequency,
            &config,
            shutdown_coordinator.token(),
            context.global_pause_resource.clone(),
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!(
                    error = %e,
                    station_id = ?request.station_id,
                    "TunerAllocationSystem: Failed to create segment"
                );
                return false;
            }
        };

        // Spawn audio entity
        let signal_data = Signal {
            frequency_hz: request.frequency,
            signal_strength: signal.info.signal_strength().unwrap_or(0.0) as f32,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: config.audio.sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: request.center_frequency,
            audio_quality: crate::audio::quality::AudioQuality::Unknown,
        };

        let sdr_rx = segment.audio_subscriber();

        match spawn_audio_entity(signal_data, sdr_rx, &config, request.center_frequency) {
            Ok((audio_entity, stream)) => {
                let audio_id = audio_entity.id;

                // Store audio resources
                if let Some(ref audio_entities) = context.audio_entities {
                    if let Ok(mut audios) = audio_entities.try_write() {
                        audios.insert(audio_entity);
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }

                if let Some(ref streams) = context.audio_streams {
                    streams.lock().unwrap().insert(audio_id, stream);
                }

                if let Some(ref segments) = context.audio_segments {
                    segments.lock().unwrap().insert(audio_id, segment);
                }

                // Update signal state
                if let TuneState::RequestQueued { allocation, .. } = &signal.tune_state {
                    let mut new_allocation = allocation.clone();
                    new_allocation.transition(TuneAllocationState::Active);
                    signal.tune_state = TuneState::Active {
                        allocation: new_allocation,
                    };
                    signal
                        .playback
                        .transition_to(crate::ecs::components::signal::PlaybackState::Playing);
                    signal.playback.set_audio_id(Some(audio_id));

                    debug!(
                        signal_id = ?signal.id(),
                        audio_id = ?audio_id,
                        frequency_mhz = request.frequency / 1e6,
                        "TunerAllocationSystem: Spawned audio entity and transitioned to Active"
                    );
                }

                true // Request processed successfully
            }
            Err(e) => {
                debug!(
                    error = %e,
                    frequency_mhz = request.frequency / 1e6,
                    "TunerAllocationSystem: Failed to spawn audio entity"
                );
                false // Spawn failed, retry
            }
        }
    }
}

impl System for TunerAllocationSystem {
    fn name(&self) -> &'static str {
        "TunerAllocationSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let tuner_request_queue = match &context.tuner_request_queue {
            Some(queue) => queue.clone(),
            None => return Ok(()),
        };

        loop {
            // Peek at front of queue without removing
            let request = {
                let queue = match tuner_request_queue.lock() {
                    Ok(q) => q,
                    Err(poisoned) => poisoned.into_inner(),
                };
                queue.front().cloned()
            };

            let request = match request {
                Some(r) => r,
                None => break, // Queue empty
            };

            // Process request
            if Self::process_request(&request, context) {
                // Success, remove from queue
                if let Ok(mut queue) = tuner_request_queue.lock() {
                    queue.pop_front();
                }
            } else {
                // Failure, leave in queue and stop processing
                break;
            }
        }

        Ok(())
    }
}

impl Default for TunerAllocationSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = TunerAllocationSystem::new();
        assert_eq!(system.name(), "TunerAllocationSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = TunerAllocationSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_runs_with_empty_queue() {
        use std::{
            collections::VecDeque,
            sync::{Arc, Mutex},
        };

        let mut system = TunerAllocationSystem::new();
        let mut context = SystemContext::new();
        context = context.with_tuner_request_queue(Arc::new(Mutex::new(VecDeque::new())));

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
