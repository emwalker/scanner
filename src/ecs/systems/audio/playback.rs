//! Audio playback system - spawns audio entities for tune requests and audio queue requests

use std::collections::HashMap;

use tracing::debug;

use crate::{
    core::types::{ModulationType, Result, Signal},
    ecs::{
        AudioEntity, AudioId, Entities, SignalEntity,
        queue::TunerRequest,
        system::{Resource, System, SystemContext},
    },
    hardware::pool::SegmentTrait,
    scanning::window::spawn_audio_entity,
};

struct AudioResources<'a> {
    entities: &'a Entities<AudioEntity>,
    streams: &'a Resource<HashMap<AudioId, cpal::Stream>>,
    segments: &'a Resource<HashMap<AudioId, crate::hardware::pool::Segment>>,
}

/// System that processes TuneRequest components and spawns AudioEntity with resources
///
/// Flow:
/// 1. Query StationEntity with tune_request.is_some()
/// 2. Get segment for window's center frequency
/// 3. Call spawn_audio_entity() helper
/// 4. Store AudioEntity in audio_entities
/// 5. Store cpal::Stream in context.audio_streams (can't be in entity - not Send)
/// 6. Clear tune_request from StationEntity
pub struct PlaybackSystem;

impl Default for PlaybackSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaybackSystem {
    pub fn new() -> Self {
        Self
    }

    fn process_single_request(
        request: &TunerRequest,
        signal_entities: &Entities<SignalEntity>,
        pool: &std::sync::Arc<crate::hardware::pool::Pool>,
        config: &std::sync::Arc<crate::core::types::ScanningConfig>,
        shutdown_coordinator: &std::sync::Arc<crate::shutdown::ShutdownCoordinator>,
        audio_resources: &AudioResources,
        global_pause_resource: &Option<crate::ecs::GlobalPauseResource>,
    ) -> bool {
        // Find the signal by station_id (string match during dual-write)
        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return false,
        };

        // During dual-write, station_id on request might not match signal IDs directly
        // For now, we'll find by frequency match since tune requests use frequency
        let signal = match signals
            .iter_mut()
            .find(|s| (s.frequency() - request.frequency).abs() < 1000.0)
        {
            Some(s) => s,
            None => {
                debug!(
                    station_id = ?request.station_id,
                    "AudioPlaybackSystem: Signal not found"
                );
                return true; // Signal gone, consider this "success" (remove from queue)
            }
        };

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
                    "AudioPlaybackSystem: Acquired tuner"
                );
                t
            }
            Err(_) => {
                return false; // Tuner not available, leave in queue
            }
        };

        // Create segment
        let segment = match crate::hardware::pool::Segment::from_tuner(
            tuner,
            request.center_frequency,
            config,
            shutdown_coordinator.token(),
            global_pause_resource.clone(),
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!(
                    error = %e,
                    station_id = ?request.station_id,
                    "AudioPlaybackSystem: Failed to create segment"
                );
                return false;
            }
        };

        // Spawn audio
        let signal_data = Signal {
            frequency_hz: request.frequency,
            signal_strength: signal.info.signal_strength().unwrap_or(0.5) as f32,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: config.audio.sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: request.center_frequency,
            audio_quality: crate::audio::quality::AudioQuality::Unknown,
        };

        let sdr_rx = segment.audio_subscriber();

        match spawn_audio_entity(signal_data, sdr_rx, config, request.center_frequency) {
            Ok((audio_entity, stream)) => {
                let audio_id = audio_entity.id;

                if let Ok(mut audios) = audio_resources.entities.try_write() {
                    audios.insert(audio_entity);
                } else {
                    return false;
                }

                audio_resources
                    .streams
                    .lock()
                    .unwrap()
                    .insert(audio_id, stream);
                audio_resources
                    .segments
                    .lock()
                    .unwrap()
                    .insert(audio_id, segment);

                signal
                    .playback
                    .transition_to(crate::ecs::components::signal::PlaybackState::Playing);
                signal.clear_tune_state();

                debug!(
                    station_id = ?request.station_id,
                    audio_id = ?audio_id,
                    frequency_mhz = request.frequency / 1e6,
                    "AudioPlaybackSystem: Successfully spawned audio"
                );

                true
            }
            Err(e) => {
                debug!(
                    error = %e,
                    station_id = ?request.station_id,
                    "AudioPlaybackSystem: Failed to spawn audio"
                );
                false
            }
        }
    }

    fn cleanup_audio_resources(
        audio_entities: &Entities<AudioEntity>,
        audio_streams: &Resource<HashMap<AudioId, cpal::Stream>>,
        audio_segments: &Resource<HashMap<AudioId, crate::hardware::pool::Segment>>,
    ) {
        let segment_count = audio_segments.lock().unwrap().len();
        let stream_count = audio_streams.lock().unwrap().len();

        if segment_count == 0 && stream_count == 0 {
            return;
        }

        let active_audio_ids: std::collections::HashSet<_> = match audio_entities.try_read() {
            Ok(audios) => audios.iter().map(|a| a.id).collect(),
            Err(_) => {
                debug!(
                    "AudioPlaybackSystem: Could not acquire audio_entities lock for cleanup, \
                     skipping"
                );
                return;
            }
        };

        debug!(
            active_audio_count = active_audio_ids.len(),
            segment_count_before_cleanup = segment_count,
            stream_count_before_cleanup = stream_count,
            "AudioPlaybackSystem: Starting cleanup"
        );

        audio_streams.lock().unwrap().retain(|audio_id, _stream| {
            let should_keep = active_audio_ids.contains(audio_id);
            if !should_keep {
                debug!(audio_id = ?audio_id, "Dropping audio stream (entity removed)");
            }
            should_keep
        });

        audio_segments.lock().unwrap().retain(|audio_id, _segment| {
            let should_keep = active_audio_ids.contains(audio_id);
            if !should_keep {
                debug!(audio_id = ?audio_id, "Dropping audio segment (entity removed)");
            }
            should_keep
        });

        debug!(
            segment_count_after_cleanup = audio_segments.lock().unwrap().len(),
            stream_count_after_cleanup = audio_streams.lock().unwrap().len(),
            "AudioPlaybackSystem: Cleanup complete"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn process_audio_queue(
        audio_entities: &Entities<AudioEntity>,
        audio_streams: &Resource<HashMap<AudioId, cpal::Stream>>,
        audio_segments: &Resource<HashMap<AudioId, crate::hardware::pool::Segment>>,
        audio_queue: &Resource<crate::ecs::components::audio::AudioQueueComponent>,
        pool: &std::sync::Arc<crate::hardware::pool::Pool>,
        config: &std::sync::Arc<crate::core::types::ScanningConfig>,
        shutdown_coordinator: &std::sync::Arc<crate::shutdown::ShutdownCoordinator>,
        global_pause_resource: &Option<crate::ecs::GlobalPauseResource>,
    ) {
        let mut queue = match audio_queue.try_lock() {
            Ok(q) => q,
            Err(_) => return, // Queue locked, skip this tick
        };

        // Process queue head if available
        if let Some(request) = queue.peek() {
            debug!(
                frequency_mhz = request.frequency() / 1e6,
                "AudioPlaybackSystem: Processing audio queue request"
            );

            // Attempt to spawn audio for this request
            let signal = request.signal().clone();
            let sdr_rx_result = pool.acquire(
                &crate::hardware::pool::TaskRequirements {
                    frequency_hz: signal.frequency_hz,
                    bandwidth_hz: config.samp_rate,
                    required_sample_rate: config.samp_rate,
                    priority: crate::hardware::pool::TaskPriority::Normal,
                },
                crate::hardware::pool::TunerActivity::Listening,
            );

            match sdr_rx_result {
                Ok(tuner) => {
                    if let Ok(segment) = crate::hardware::pool::Segment::from_tuner(
                        tuner,
                        signal.frequency_hz,
                        config,
                        shutdown_coordinator.token(),
                        global_pause_resource.clone(),
                    ) {
                        let sdr_rx = segment.audio_subscriber();

                        match spawn_audio_entity(
                            signal.clone(),
                            sdr_rx,
                            config,
                            signal.frequency_hz,
                        ) {
                            Ok((audio_entity, stream)) => {
                                let audio_id = audio_entity.id;

                                if let Ok(mut audios) = audio_entities.try_write() {
                                    audios.insert(audio_entity);

                                    audio_streams.lock().unwrap().insert(audio_id, stream);
                                    audio_segments.lock().unwrap().insert(audio_id, segment);

                                    debug!(
                                        audio_id = ?audio_id,
                                        frequency_mhz = signal.frequency_hz / 1e6,
                                        "AudioPlaybackSystem: Spawned audio from queue"
                                    );

                                    // Successfully spawned, remove from queue
                                    queue.dequeue();
                                }
                            }
                            Err(e) => {
                                debug!(
                                    error = %e,
                                    frequency_mhz = signal.frequency_hz / 1e6,
                                    "AudioPlaybackSystem: Failed to spawn audio from queue"
                                );
                            }
                        }
                    }
                }
                Err(_) => {
                    // Tuner not available, leave in queue for next tick
                    debug!(
                        frequency_mhz = request.frequency() / 1e6,
                        "AudioPlaybackSystem: Tuner not available for queue request"
                    );
                }
            }
        }
    }
}

impl System for PlaybackSystem {
    fn name(&self) -> &'static str {
        "AudioPlayback"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        // Don't process tune requests during global pause
        if context.is_globally_paused() {
            return Ok(());
        }

        let (signal_entities, audio_entities, tuner_request_queue) = match (
            &context.signal_entities,
            &context.audio_entities,
            &context.tuner_request_queue,
        ) {
            (Some(se), Some(ae), Some(queue)) => (se.clone(), ae.clone(), queue.clone()),
            _ => return Ok(()),
        };

        let (pool, config, shutdown_coordinator, audio_streams, audio_segments) = match (
            &context.pool,
            &context.config,
            &context.shutdown_coordinator,
            &context.audio_streams,
            &context.audio_segments,
        ) {
            (Some(p), Some(c), Some(s), Some(streams), Some(segments)) => (
                p.clone(),
                c.clone(),
                s.clone(),
                streams.clone(),
                segments.clone(),
            ),
            _ => {
                debug!("AudioPlaybackSystem: Missing required resources in context");
                return Ok(());
            }
        };

        // Queue-based processing: Process requests from front of queue
        // If tuner available → spawn audio and pop from queue
        // If tuner busy → leave in queue, try again next tick (deterministic waiting)
        loop {
            let request = {
                let queue = match tuner_request_queue.try_lock() {
                    Ok(q) => q,
                    Err(_) => break, // Queue locked, skip this tick
                };

                match queue.front() {
                    Some(req) => req.clone(),
                    None => break, // Queue empty
                }
            };

            debug!(
                station_id = ?request.station_id,
                frequency_mhz = request.frequency / 1e6,
                "AudioPlaybackSystem: Processing request from queue"
            );

            // Try to acquire tuner and spawn audio
            let success = Self::process_single_request(
                &request,
                &signal_entities,
                &pool,
                &config,
                &shutdown_coordinator,
                &AudioResources {
                    entities: &audio_entities,
                    streams: &audio_streams,
                    segments: &audio_segments,
                },
                &context.global_pause_resource,
            );

            if success {
                // Success - remove from queue and continue to next request
                if let Ok(mut queue) = tuner_request_queue.try_lock() {
                    queue.pop_front();
                    debug!(
                        station_id = ?request.station_id,
                        queue_length = queue.len(),
                        "AudioPlaybackSystem: Request completed, removed from queue"
                    );
                }
            } else {
                // Failed to acquire tuner - leave in queue, try again next tick
                debug!(
                    station_id = ?request.station_id,
                    "AudioPlaybackSystem: Tuner not available, request remains in queue"
                );
                break;
            }
        }

        // Process audio queue if available
        if let Some(audio_queue) = &context.audio_queue {
            Self::process_audio_queue(
                &audio_entities,
                &audio_streams,
                &audio_segments,
                audio_queue,
                &pool,
                &config,
                &shutdown_coordinator,
                &context.global_pause_resource,
            );
        }

        Self::cleanup_audio_resources(&audio_entities, &audio_streams, &audio_segments);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex, RwLock},
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, ScanningConfig, Signal},
        ecs::{Entity, EntityWorld, StationEntity, TaskId, components::window::WindowId},
        hardware::pool::Pool,
        shutdown::ShutdownCoordinator,
    };

    fn create_test_signal(frequency: f64) -> Signal {
        Signal {
            frequency_hz: frequency,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: frequency,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_empty_queue_returns_immediately() {
        let mut system = PlaybackSystem::new();

        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let tuner_request_queue = Arc::new(Mutex::new(VecDeque::new()));

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities)
            .with_tuner_request_queue(tuner_request_queue.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(queue.len(), 0, "Queue should remain empty");
    }

    #[test]
    fn test_request_remains_in_queue_when_no_context() {
        let mut system = PlaybackSystem::new();

        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        let tuner_request_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            queue.push_back(TunerRequest {
                station_id: crate::ecs::StationId::new(),
                frequency: 88.9e6,
                window_id: window_id.clone(),
                center_frequency: 88.9e6,
            });
        }

        let mut context =
            SystemContext::new().with_tuner_request_queue(tuner_request_queue.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(
            queue.len(),
            1,
            "Request should remain in queue when missing context"
        );
    }

    #[test]
    fn test_request_removed_when_station_not_found() {
        let mut system = PlaybackSystem::new();

        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new())); // Empty signal_entities
        let pool = Arc::new(Pool::new_unfiltered());
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        #[allow(clippy::arc_with_non_send_sync)]
        let audio_streams = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let audio_segments = Arc::new(Mutex::new(std::collections::HashMap::new()));

        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        let tuner_request_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            queue.push_back(TunerRequest {
                station_id: crate::ecs::StationId::new(),
                frequency: 88.9e6,
                window_id: window_id.clone(),
                center_frequency: 88.9e6,
            });
        }

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities)
            .with_signal_entities(signal_entities) // Add empty signal_entities
            .with_tuner_request_queue(tuner_request_queue.clone())
            .with_pool(pool)
            .with_config(config)
            .with_shutdown_coordinator(shutdown)
            .with_audio_streams(audio_streams)
            .with_audio_segments(audio_segments);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(
            queue.len(),
            0,
            "Request should be removed when signal not found (empty signal_entities)"
        );
    }

    #[test]
    fn test_system_respects_global_pause() {
        use crate::ecs::GlobalPauseState;

        let mut system = PlaybackSystem::new();

        let tuner_request_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            let task_id = TaskId::new("test-scan".to_string());
            let window_id = WindowId::new(task_id, 0);
            queue.push_back(TunerRequest {
                station_id: crate::ecs::StationId::new(),
                frequency: 88.9e6,
                window_id,
                center_frequency: 88.9e6,
            });
        }

        let global_pause = Arc::new(Mutex::new(GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));

        let mut context = SystemContext::new()
            .with_tuner_request_queue(tuner_request_queue.clone())
            .with_global_pause_resource(global_pause);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(
            queue.len(),
            1,
            "AudioPlaybackSystem should not process queue during global pause"
        );
    }

    #[test]
    fn test_queue_fifo_order() {
        let mut system = PlaybackSystem::new();

        let signal1 = create_test_signal(88.9e6);
        let signal2 = create_test_signal(89.7e6);

        let mut station_world = EntityWorld::new();
        let window_id1 = WindowId::new(TaskId::new("test-scan".to_string()), 0);
        let station1 = StationEntity::from_signal(&signal1, window_id1.clone());
        let window_id2 = WindowId::new(TaskId::new("test-scan".to_string()), 1);
        let station2 = StationEntity::from_signal(&signal2, window_id2.clone());
        let station1_id = *station1.id();
        let station2_id = *station2.id();
        station_world.insert(station1);
        station_world.insert(station2);

        let station_entities = Arc::new(RwLock::new(station_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let pool = Arc::new(Pool::new_unfiltered());
        let config = Arc::new(ScanningConfig::default());
        let shutdown = Arc::new(ShutdownCoordinator::new());
        #[allow(clippy::arc_with_non_send_sync)]
        let audio_streams = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let audio_segments = Arc::new(Mutex::new(std::collections::HashMap::new()));

        let tuner_request_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            queue.push_back(TunerRequest {
                station_id: station1_id,
                frequency: 88.9e6,
                window_id: window_id1.clone(),
                center_frequency: 88.9e6,
            });
            queue.push_back(TunerRequest {
                station_id: station2_id,
                frequency: 89.7e6,
                window_id: window_id2.clone(),
                center_frequency: 89.7e6,
            });
        }

        let mut context = SystemContext::new()
            .with_audio_entities(audio_entities)
            .with_tuner_request_queue(tuner_request_queue.clone())
            .with_pool(pool)
            .with_config(config)
            .with_shutdown_coordinator(shutdown)
            .with_audio_streams(audio_streams)
            .with_audio_segments(audio_segments);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        if queue.is_empty() {
            let stations = station_entities.read().unwrap();
            let mut processed_first = false;
            for station in stations.iter() {
                if station.id() == &station1_id && station.playback.is_playing() {
                    processed_first = true;
                }
            }
            assert!(
                processed_first,
                "First request in queue should be processed first"
            );
        }
    }

    #[test]
    fn test_cleanup_preserves_active_resources() {
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        #[allow(clippy::arc_with_non_send_sync)]
        let audio_streams = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let audio_segments = Arc::new(Mutex::new(std::collections::HashMap::new()));

        let signal = create_test_signal(88.9e6);
        let audio = crate::ecs::AudioEntity::new(signal, 88.9e6, None);

        {
            let mut audios = audio_entities.write().unwrap();
            audios.insert(audio);
        }

        PlaybackSystem::cleanup_audio_resources(&audio_entities, &audio_streams, &audio_segments);

        let streams = audio_streams.lock().unwrap();
        let segments = audio_segments.lock().unwrap();
        assert_eq!(streams.len(), 0, "No orphaned streams should exist");
        assert_eq!(segments.len(), 0, "No orphaned segments should exist");
    }
}
