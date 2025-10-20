//! Tune transition system - coordinates multi-step playback transitions

use crate::core::types::Result;
use crate::ecs::Entity;
use crate::ecs::components::station::TuneStage;
use crate::ecs::queue::TunerRequest;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

/// System that coordinates tune transitions through multiple stages
///
/// This system progresses stations through the transition lifecycle:
/// 1. AwaitingTunerRelease: Wait for scan to pause and release tuner
/// 2. AcquiringResources: Enqueue tuner request for AudioPlaybackSystem
/// 3. AwaitingPlayback: Wait for audio playback to start
///
/// The queue-based approach provides deterministic resource acquisition
/// without retries or timeouts.
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
        let (station_entities, tuner_request_queue) =
            match (&context.station_entities, &context.tuner_request_queue) {
                (Some(se), Some(queue)) => (se.clone(), queue.clone()),
                _ => return Ok(()),
            };

        let mut stations = match station_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        let scans_opt = context
            .scan_entities
            .as_ref()
            .and_then(|se| se.try_read().ok());

        for station in stations.iter_mut() {
            // Check if station has transition and extract stage
            let (current_stage, should_timeout) = match &station.transition {
                Some(t) => (t.stage, t.should_timeout()),
                None => continue,
            };

            if should_timeout {
                debug!(
                    station_id = ?station.id(),
                    stage = ?current_stage,
                    "TuneTransition: Transition timed out"
                );
                station.transition = None;
                continue;
            }

            match current_stage {
                TuneStage::AwaitingTunerRelease => {
                    if let Some(ref scans) = scans_opt {
                        let scan_paused = scans.iter().any(|s| s.is_paused() || s.is_listening());

                        if scan_paused {
                            if let Some(ref mut t) = station.transition {
                                t.stage = TuneStage::AcquiringResources;
                            }
                            debug!(
                                station_id = ?station.id(),
                                "TuneTransition: Tuner released, acquiring resources"
                            );
                        }
                    }
                }

                TuneStage::AcquiringResources => {
                    // Extract transition data
                    let (window_id, center_freq) = match &station.transition {
                        Some(t) => (t.window_id, t.center_frequency),
                        None => continue,
                    };

                    // Enqueue tuner request for AudioPlaybackSystem
                    let request = TunerRequest {
                        station_id: *station.id(),
                        frequency: station.frequency(),
                        window_id,
                        center_frequency: center_freq,
                    };

                    if let Ok(mut queue) = tuner_request_queue.try_lock() {
                        queue.push_back(request);
                        debug!(
                            station_id = ?station.id(),
                            station_frequency_mhz = station.frequency() / 1e6,
                            window_frequency_mhz = center_freq / 1e6,
                            queue_length = queue.len(),
                            "TuneTransition: Enqueued tuner request"
                        );

                        // Clear transition - AudioPlaybackSystem will handle the rest
                        station.transition = None;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioQuality;
    use crate::core::types::{ModulationType, Signal};
    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::components::station::TuneTransitionComponent;
    use crate::ecs::{EntityWorld, ScanEntity, ScanId, StationEntity};
    use crate::scanning::window::WindowMetadata;
    use std::sync::{Arc, RwLock};
    use std::time::SystemTime;

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

    fn create_test_scan(freq_min: f64, freq_max: f64) -> ScanEntity {
        let config = ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            1.0e6,
            2.4e6,
            40.0,
            1.0,
            3,
        );
        ScanEntity::new(config)
    }

    #[test]
    fn test_system_with_no_entities() {
        let mut system = TransitionSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transition_from_awaiting_tuner_release_to_acquiring_resources() {
        let mut system = TransitionSystem::new();

        let signal = create_test_signal(88.9e6);
        let mut station = StationEntity::from_signal(
            &signal,
            ScanId::new(),
            WindowMetadata {
                window_id: 0,
                center_frequency_hz: 88.9e6,
            },
        );

        station.transition = Some(TuneTransitionComponent::new(0, 88.9e6));

        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_listening(0);

        let mut station_world = EntityWorld::new();
        let mut scan_world = EntityWorld::new();
        station_world.insert(station);
        scan_world.insert(scan);

        let station_entities = Arc::new(RwLock::new(station_world));
        let scan_entities = Arc::new(RwLock::new(scan_world));
        let tuner_request_queue =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities.clone())
            .with_scan_entities(scan_entities)
            .with_tuner_request_queue(tuner_request_queue.clone());

        // First run: AwaitingTunerRelease -> AcquiringResources
        let result = system.run(&mut context);
        assert!(result.is_ok());

        {
            let stations = station_entities.read().unwrap();
            for station in stations.iter() {
                if let Some(ref transition) = station.transition {
                    assert_eq!(transition.stage, TuneStage::AcquiringResources);
                }
            }
        }

        // Second run: AcquiringResources -> enqueue and clear
        let result = system.run(&mut context);
        assert!(result.is_ok());

        // After enqueuing, transition should be cleared
        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(
                station.transition.is_none(),
                "Transition should be cleared after enqueuing"
            );
        }

        // Verify request was enqueued
        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(queue.len(), 1, "Should have 1 request in queue");
    }

    #[test]
    fn test_timeout_clears_transition() {
        let mut system = TransitionSystem::new();

        let signal = create_test_signal(88.9e6);
        let mut station = StationEntity::from_signal(
            &signal,
            ScanId::new(),
            WindowMetadata {
                window_id: 0,
                center_frequency_hz: 88.9e6,
            },
        );

        let mut transition = TuneTransitionComponent::new(0, 88.9e6);
        transition.requested_at = std::time::Instant::now() - std::time::Duration::from_secs(11);
        station.transition = Some(transition);

        let mut station_world = EntityWorld::new();
        station_world.insert(station);

        let station_entities = Arc::new(RwLock::new(station_world));
        let tuner_request_queue =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

        let mut context = SystemContext::new()
            .with_station_entities(station_entities.clone())
            .with_tuner_request_queue(tuner_request_queue);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(
                station.transition.is_none(),
                "Transition should be cleared on timeout"
            );
        }
    }
}
