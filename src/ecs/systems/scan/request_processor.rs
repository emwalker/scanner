//! Scan request processor system - processes pause/resume request components

use crate::core::types::Result;
use crate::ecs::Entity;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

/// System that processes pause and resume request components on ScanEntity
///
/// This system:
/// - Queries for ScanEntity with pause_request.is_some()
/// - Processes pause requests by updating scan state
/// - Clears pause_request after processing
/// - Queries for ScanEntity with resume_request.is_some()
/// - Processes resume requests by updating scan state
/// - Clears resume_request after processing
pub struct RequestProcessorSystem;

impl Default for RequestProcessorSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestProcessorSystem {
    pub fn new() -> Self {
        Self
    }

    fn stop_previous_playback(context: &SystemContext, scan_id: &crate::ecs::ScanId) {
        // Stop any currently playing stations
        if let Some(ref station_entities) = context.station_entities {
            let mut stations = station_entities.write().unwrap();
            for station in stations.iter_mut() {
                if station.playback.is_playing() {
                    station.playback.stop_playing();
                    debug!(
                        scan_id = ?scan_id,
                        station_id = ?station.id(),
                        "ScanRequestProcessor: Stopped previous station playback"
                    );
                }
            }
        }

        // Stop any audio entities from previous playback
        if let Some(ref audio_entities) = context.audio_entities
            && let Ok(mut audios) = audio_entities.try_write()
        {
            let count = audios.len();
            // Cancel all audio graphs before clearing to prevent busy-wait
            for audio in audios.iter() {
                if let Some(ref cancel) = audio.allocation.graph_cancel {
                    cancel.cancel();
                }
            }
            audios.clear();
            if count > 0 {
                debug!(
                    scan_id = ?scan_id,
                    audio_count = count,
                    "ScanRequestProcessor: Stopped and cleared previous audio entities"
                );
            }
        }
    }

    fn start_tune_transition(
        context: &SystemContext,
        scan_id: &crate::ecs::ScanId,
        station_freq: f64,
        window_num: usize,
        window_center_freq: f64,
    ) {
        if let Some(ref station_entities) = context.station_entities {
            let mut stations = station_entities.write().unwrap();
            for station in stations.iter_mut() {
                if (station.frequency() - station_freq).abs() < 1000.0 {
                    station.transition = Some(
                        crate::ecs::components::station::TuneTransitionComponent::new(
                            window_num,
                            window_center_freq,
                        ),
                    );
                    debug!(
                        scan_id = ?scan_id,
                        station_id = ?station.id(),
                        station_frequency_mhz = station_freq / 1e6,
                        stage = "AwaitingTunerRelease",
                        "ScanRequestProcessor: Started tune transition"
                    );
                    break;
                }
            }
        }
    }
}

impl System for RequestProcessorSystem {
    fn name(&self) -> &'static str {
        "ScanRequestProcessor"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let scan_entities = match &context.scan_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let tuner_request_queue = context.tuner_request_queue.clone();

        // Process pause request queue: pop requests and set component on ScanEntity
        if let Some(ref pause_request_queue) = context.pause_request_queue {
            let mut queue = match pause_request_queue.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    debug!("Pause request queue lock poisoned, recovering");
                    poisoned.into_inner()
                }
            };
            while let Some(request) = queue.pop_front() {
                let mut scans = scan_entities.write().map_err(|e| {
                    crate::core::types::ScannerError::LockPoisoned(format!("scan_entities: {}", e))
                })?;
                if let Some(scan) = scans.iter_mut().find(|s| s.id() == &request.scan_id) {
                    if let Some(station_freq) = request.station_frequency_hz {
                        let window_center_freq = request.window_center_frequency_hz.unwrap();
                        scan.request_pause_with_station(
                            request.window_num,
                            station_freq,
                            window_center_freq,
                        );
                        debug!(
                            scan_id = ?request.scan_id,
                            window_num = request.window_num,
                            station_frequency_mhz = station_freq / 1e6,
                            "ScanRequestProcessor: Set pause_request component from queue (with station)"
                        );
                    } else {
                        scan.request_pause(request.window_num);
                        debug!(
                            scan_id = ?request.scan_id,
                            window_num = request.window_num,
                            "ScanRequestProcessor: Set pause_request component from queue"
                        );
                    }
                }
            }
        }

        let mut scans = scan_entities.write().map_err(|e| {
            crate::core::types::ScannerError::LockPoisoned(format!("scan_entities: {}", e))
        })?;

        for scan in scans.iter_mut() {
            // Process pause request component
            if let Some(ref pause_request) = scan.pause_request {
                debug!(
                    scan_id = ?scan.id(),
                    window_num = pause_request.window_num,
                    has_station = pause_request.station_frequency_hz.is_some(),
                    "ScanRequestProcessor: Processing pause request"
                );

                // If pause request includes station info, transition to Listening state
                if let Some(station_freq) = pause_request.station_frequency_hz
                    && let Some(window_center_freq) = pause_request.window_center_frequency_hz
                {
                    scan.progress.start_listening(pause_request.window_num);
                    scan.lifecycle.pause();

                    Self::stop_previous_playback(context, scan.id());
                    Self::start_tune_transition(
                        context,
                        scan.id(),
                        station_freq,
                        pause_request.window_num,
                        window_center_freq,
                    );
                } else {
                    // Regular pause without station
                    scan.progress.pause(pause_request.window_num);
                    scan.lifecycle.pause();
                }

                scan.clear_pause_request();
            }

            // Process resume request component
            if let Some(ref resume_request) = scan.resume_request {
                debug!(
                    scan_id = ?scan.id(),
                    window_num = resume_request.window_num,
                    is_listening = scan.progress.is_listening(),
                    "ScanRequestProcessor: Processing resume request"
                );

                // If we were listening, set station playback state to Idle to stop audio
                if scan.progress.is_listening()
                    && let Some(ref station_entities) = context.station_entities
                {
                    let mut stations = station_entities.write().unwrap();
                    for station in stations.iter_mut() {
                        if station.playback.is_playing() {
                            station.playback.stop_playing();
                            debug!(
                                scan_id = ?scan.id(),
                                station_id = ?station.id(),
                                "ScanRequestProcessor: Set playback state to Idle to stop audio"
                            );
                        }
                    }
                }

                // Clear tuner request queue - resuming scan means canceling any pending playback
                if let Some(ref queue) = tuner_request_queue
                    && let Ok(mut q) = queue.try_lock()
                {
                    let cleared_count = q.len();
                    q.clear();
                    if cleared_count > 0 {
                        debug!(
                            scan_id = ?scan.id(),
                            cleared_count = cleared_count,
                            "ScanRequestProcessor: Cleared tuner request queue on resume"
                        );
                    }
                }

                scan.progress.resume();
                scan.clear_resume_request();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::{EntityWorld, ScanEntity};
    use std::sync::{Arc, RwLock};

    fn create_test_scan(freq_min: f64, freq_max: f64) -> ScanEntity {
        let config = ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            1.0e6,
            2.0e6,
            40.0,
            0.5,
            10,
        );
        ScanEntity::new(config)
    }

    #[test]
    fn test_no_requests() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_scan(88.0e6, 108.0e6));

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(scan.pause_request.is_none());
            assert!(scan.resume_request.is_none());
        }
    }

    #[test]
    fn test_processes_pause_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_window(0);
        scan.request_pause(5);
        assert!(scan.pause_request.is_some());
        assert!(scan.is_scanning());

        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(
                scan.pause_request.is_none(),
                "Pause request should be cleared"
            );
            assert!(scan.is_paused(), "Scan should be paused");
        }
    }

    #[test]
    fn test_processes_resume_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.pause(5);
        assert!(scan.is_paused());

        scan.request_resume(5);
        assert!(scan.resume_request.is_some());

        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(
                scan.resume_request.is_none(),
                "Resume request should be cleared"
            );
            assert!(scan.is_scanning(), "Scan should be scanning");
        }
    }

    #[test]
    fn test_processes_both_requests_in_sequence() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);

        scan.request_pause(3);
        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        system.run(&mut context).unwrap();

        {
            let entities = scan_entities.read().unwrap();
            for scan in entities.iter() {
                assert!(scan.is_paused());
            }
        }

        {
            let mut entities = scan_entities.write().unwrap();
            for scan in entities.iter_mut() {
                scan.request_resume(3);
            }
        }

        system.run(&mut context).unwrap();

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(scan.is_scanning());
        }
    }

    #[test]
    fn test_pause_with_station_starts_tune_transition() {
        let mut system = RequestProcessorSystem::new();

        let mut scan_world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_window(0);
        scan.request_pause_with_station(0, 88.9e6, 88.9e6);
        scan_world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(scan_world));

        let signal = crate::core::types::Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let mut station_world = EntityWorld::new();
        let station = crate::ecs::StationEntity::from_signal(
            &signal,
            crate::ecs::ScanId::new(),
            crate::scanning::window::WindowMetadata {
                window_id: 0,
                center_frequency_hz: 88.9e6,
            },
        );
        station_world.insert(station);

        let station_entities = Arc::new(RwLock::new(station_world));
        let mut context = SystemContext::new()
            .with_scan_entities(scan_entities.clone())
            .with_station_entities(station_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let scans = scan_entities.read().unwrap();
        for scan in scans.iter() {
            assert!(scan.is_listening(), "Scan should be in listening mode");
            assert!(
                scan.pause_request.is_none(),
                "Pause request should be cleared"
            );
        }

        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(
                station.transition.is_some(),
                "Station should have tune transition started"
            );
        }
    }

    #[test]
    fn test_pause_with_station_stops_previous_playback() {
        let mut system = RequestProcessorSystem::new();

        let mut scan_world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_listening(0);
        scan.request_pause_with_station(1, 89.7e6, 89.7e6);
        scan_world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(scan_world));

        let signal1 = crate::core::types::Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let signal2 = crate::core::types::Signal {
            frequency_hz: 89.7e6,
            signal_strength: 0.7,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 89.7e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let mut station_world = EntityWorld::new();
        let mut station1 = crate::ecs::StationEntity::from_signal(
            &signal1,
            crate::ecs::ScanId::new(),
            crate::scanning::window::WindowMetadata {
                window_id: 0,
                center_frequency_hz: 88.9e6,
            },
        );
        station1.playback.start_playing(crate::ecs::AudioId::new());

        let station2 = crate::ecs::StationEntity::from_signal(
            &signal2,
            crate::ecs::ScanId::new(),
            crate::scanning::window::WindowMetadata {
                window_id: 1,
                center_frequency_hz: 89.7e6,
            },
        );

        station_world.insert(station1);
        station_world.insert(station2);

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new()
            .with_scan_entities(scan_entities.clone())
            .with_station_entities(station_entities.clone())
            .with_audio_entities(audio_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let stations = station_entities.read().unwrap();
        for station in stations.iter() {
            assert!(
                !station.playback.is_playing(),
                "Previous playback should be stopped"
            );
        }
    }

    #[test]
    fn test_resume_clears_tuner_request_queue() {
        let mut system = RequestProcessorSystem::new();

        let mut scan_world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_listening(0);
        scan.request_resume(0);
        scan_world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(scan_world));

        let tuner_request_queue =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            queue.push_back(crate::ecs::queue::TunerRequest {
                station_id: crate::ecs::StationId::new(),
                frequency: 88.9e6,
                window_id: 0,
                center_frequency: 88.9e6,
            });
        }

        let mut context = SystemContext::new()
            .with_scan_entities(scan_entities.clone())
            .with_tuner_request_queue(tuner_request_queue.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(queue.len(), 0, "Queue should be cleared on resume");

        let scans = scan_entities.read().unwrap();
        for scan in scans.iter() {
            assert!(scan.is_scanning(), "Scan should be in scanning mode");
        }
    }

    #[test]
    fn test_audio_graphs_canceled_before_clearing() {
        let mut system = RequestProcessorSystem::new();

        let mut scan_world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_listening(0);
        scan.request_pause_with_station(1, 89.7e6, 89.7e6);
        scan_world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(scan_world));

        let signal1 = crate::core::types::Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let signal2 = crate::core::types::Signal {
            frequency_hz: 89.7e6,
            signal_strength: 0.7,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 89.7e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let mut station_world = EntityWorld::new();
        let station1 = crate::ecs::StationEntity::from_signal(
            &signal1,
            crate::ecs::ScanId::new(),
            crate::scanning::window::WindowMetadata {
                window_id: 0,
                center_frequency_hz: 88.9e6,
            },
        );
        let station2 = crate::ecs::StationEntity::from_signal(
            &signal2,
            crate::ecs::ScanId::new(),
            crate::scanning::window::WindowMetadata {
                window_id: 1,
                center_frequency_hz: 89.7e6,
            },
        );
        station_world.insert(station1);
        station_world.insert(station2);

        let station_entities = Arc::new(RwLock::new(station_world));

        let mut audio_world = EntityWorld::new();
        let mut audio = crate::ecs::AudioEntity::new(signal1, 88.9e6, None);

        let cancel_token = rustradio::graph::CancellationToken::new();
        let cancel_clone = cancel_token.clone();
        audio.allocation.graph_cancel = Some(cancel_token);
        audio_world.insert(audio);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_scan_entities(scan_entities.clone())
            .with_station_entities(station_entities)
            .with_audio_entities(audio_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        assert!(
            cancel_clone.is_canceled(),
            "Audio graph should be canceled before clearing"
        );

        let audios = audio_entities.read().unwrap();
        assert_eq!(audios.len(), 0, "Audio entities should be cleared");
    }

    #[test]
    fn test_multiple_audio_graphs_all_canceled() {
        let mut system = RequestProcessorSystem::new();

        let mut scan_world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_listening(0);
        scan.request_pause_with_station(2, 90.5e6, 90.5e6);
        scan_world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(scan_world));

        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut audio_world = EntityWorld::new();

        let signal1 = crate::core::types::Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let signal2 = crate::core::types::Signal {
            frequency_hz: 89.7e6,
            signal_strength: 0.7,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 89.7e6,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        let cancel1 = rustradio::graph::CancellationToken::new();
        let cancel2 = rustradio::graph::CancellationToken::new();
        let cancel1_clone = cancel1.clone();
        let cancel2_clone = cancel2.clone();

        let mut audio1 = crate::ecs::AudioEntity::new(signal1, 88.9e6, None);
        audio1.allocation.graph_cancel = Some(cancel1);

        let mut audio2 = crate::ecs::AudioEntity::new(signal2, 89.7e6, None);
        audio2.allocation.graph_cancel = Some(cancel2);

        audio_world.insert(audio1);
        audio_world.insert(audio2);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_scan_entities(scan_entities)
            .with_station_entities(station_entities)
            .with_audio_entities(audio_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        assert!(
            cancel1_clone.is_canceled(),
            "First audio graph should be canceled"
        );
        assert!(
            cancel2_clone.is_canceled(),
            "Second audio graph should be canceled"
        );

        let audios = audio_entities.read().unwrap();
        assert_eq!(audios.len(), 0, "All audio entities should be cleared");
    }
}
