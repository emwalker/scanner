//! Scan request processor system - processes pause/resume request components

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::window::WindowId,
        system::{System, SystemContext},
    },
};

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

    fn stop_previous_playback(context: &SystemContext, task_id: &crate::ecs::TaskId) {
        // Stop any currently playing signals
        if let Some(ref signal_entities) = context.signal_entities {
            let mut signals = signal_entities.write().unwrap();
            for signal in signals.iter_mut() {
                if signal.playback.is_playing() {
                    signal
                        .playback
                        .transition_to(crate::ecs::components::signal::PlaybackState::NotPlaying);
                    debug!(
                        task_id = ?task_id,
                        signal_id = ?signal.id(),
                        "ScanRequestProcessor: Stopped previous signal playback"
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
                    task_id = ?task_id,
                    audio_count = count,
                    "ScanRequestProcessor: Stopped and cleared previous audio entities"
                );
            }
        }
    }

    fn start_tune_transition(
        context: &SystemContext,
        task_id: &crate::ecs::TaskId,
        signal_freq: f64,
        window_num: usize,
        window_center_freq: f64,
    ) {
        let window_id = WindowId::new(task_id.clone(), window_num);
        if let Some(ref signal_entities) = context.signal_entities {
            let mut signals = signal_entities.write().unwrap();
            for signal in signals.iter_mut() {
                if (signal.frequency() - signal_freq).abs() < 1000.0 {
                    match signal.request_tune_transition(window_id.clone(), window_center_freq) {
                        Ok(()) => {
                            debug!(
                                task_id = ?task_id,
                                signal_id = ?signal.id(),
                                signal_frequency_mhz = signal_freq / 1e6,
                                window_center_frequency_mhz = window_center_freq / 1e6,
                                "ScanRequestProcessor: Successfully requested tune transition"
                            );
                        }
                        Err(err) => {
                            debug!(
                                task_id = ?task_id,
                                signal_id = ?signal.id(),
                                signal_frequency_mhz = signal_freq / 1e6,
                                error = %err,
                                "ScanRequestProcessor: Failed to request tune transition"
                            );
                        }
                    }
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
        let task_entities = match &context.task_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let tuner_request_queue = context.tuner_request_queue.clone();

        // Process pause request queue: pop requests and set component on TaskEntity
        if let Some(ref pause_request_queue) = context.pause_request_queue {
            let mut queue = match pause_request_queue.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    debug!("Pause request queue lock poisoned, recovering");
                    poisoned.into_inner()
                }
            };
            while let Some(request) = queue.pop_front() {
                let mut tasks = task_entities.write().map_err(|e| {
                    crate::core::types::ScannerError::LockPoisoned(format!("task_entities: {}", e))
                })?;
                if let Some(task) = tasks.iter_mut().find(|t| t.id() == &request.task_id) {
                    if let Some(signal_freq) = request.station_frequency_hz {
                        let window_center_freq = request.window_center_frequency_hz.unwrap();
                        task.request_pause_with_station(
                            request.window_num,
                            signal_freq,
                            window_center_freq,
                        );
                        debug!(
                            task_id = ?request.task_id,
                            window_num = request.window_num,
                            signal_frequency_mhz = signal_freq / 1e6,
                            "ScanRequestProcessor: Set pause_request component from queue (with signal)"
                        );
                    } else {
                        task.request_pause(request.window_num);
                        debug!(
                            task_id = ?request.task_id,
                            window_num = request.window_num,
                            "ScanRequestProcessor: Set pause_request component from queue"
                        );
                    }
                }
            }
        }

        let mut tasks = task_entities.write().map_err(|e| {
            crate::core::types::ScannerError::LockPoisoned(format!("task_entities: {}", e))
        })?;

        for task in tasks.iter_mut() {
            let task_id = task.id().clone();

            let crate::ecs::TaskComponents::Scan {
                pause_request,
                resume_request,
                progress,
                lifecycle,
                ..
            } = &mut task.components;
            // Process pause request component
            if let Some(pause_req) = pause_request {
                debug!(
                    task_id = ?task_id,
                    window_num = pause_req.window_num,
                    has_signal = pause_req.station_frequency_hz.is_some(),
                    "ScanRequestProcessor: Processing pause request"
                );

                // If pause request includes signal info, transition to Listening state
                if let Some(signal_freq) = pause_req.station_frequency_hz
                    && let Some(window_center_freq) = pause_req.window_center_frequency_hz
                {
                    let window_id = WindowId::new(task_id.clone(), pause_req.window_num);
                    progress.start_listening(window_id);
                    lifecycle.pause();

                    Self::stop_previous_playback(context, &task_id);
                    Self::start_tune_transition(
                        context,
                        &task_id,
                        signal_freq,
                        pause_req.window_num,
                        window_center_freq,
                    );
                } else {
                    // Regular pause without signal
                    let window_id = WindowId::new(task_id.clone(), pause_req.window_num);
                    progress.pause(window_id);
                    lifecycle.pause();
                }

                *pause_request = None;
            }

            // Process resume request component
            if let Some(resume_req) = resume_request {
                debug!(
                    task_id = ?task_id,
                    window_num = resume_req.window_num,
                    is_listening = progress.is_listening(),
                    "ScanRequestProcessor: Processing resume request"
                );

                // If we were listening, set signal playback state to NotPlaying to stop audio
                if progress.is_listening()
                    && let Some(ref signal_entities) = context.signal_entities
                {
                    let mut signals = signal_entities.write().unwrap();
                    for signal in signals.iter_mut() {
                        if signal.playback.is_playing() {
                            signal.playback.transition_to(
                                crate::ecs::components::signal::PlaybackState::NotPlaying,
                            );
                            debug!(
                                task_id = ?task_id,
                                signal_id = ?signal.id(),
                                "ScanRequestProcessor: Set playback state to NotPlaying to stop audio"
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
                            task_id = ?task_id,
                            cleared_count = cleared_count,
                            "ScanRequestProcessor: Cleared tuner request queue on resume"
                        );
                    }
                }

                progress.resume();
                *resume_request = None;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::ecs::{
        EntityWorld, ScanTaskData, SignalEntity, TaskEntity, TaskId, components::window::WindowId,
    };

    fn create_test_task(task_id: &str, total_windows: usize) -> TaskEntity {
        TaskEntity::new_scan_with_defaults(
            TaskId::new(task_id.to_string()),
            ScanTaskData::Placeholder,
            total_windows,
        )
    }

    #[test]
    fn test_no_requests() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_task("test-scan", 10));

        let task_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(task_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = task_entities.read().unwrap();
        for task in entities.iter() {
            let crate::ecs::TaskComponents::Scan {
                pause_request,
                resume_request,
                ..
            } = &task.components;
            assert!(pause_request.is_none());
            assert!(resume_request.is_none());
        }
    }

    #[test]
    fn test_processes_pause_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id.clone(), 0);
        progress.start_window(window_id);
        assert!(progress.is_scanning());

        task.request_pause(5);

        let crate::ecs::TaskComponents::Scan { pause_request, .. } = &task.components;
        assert!(pause_request.is_some());

        world.insert(task);

        let task_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(task_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = task_entities.read().unwrap();
        for task in entities.iter() {
            let crate::ecs::TaskComponents::Scan {
                pause_request,
                progress,
                ..
            } = &task.components;
            assert!(pause_request.is_none(), "Pause request should be cleared");
            assert!(progress.is_paused(), "Scan should be paused");
        }
    }

    #[test]
    fn test_processes_resume_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 5);
        progress.pause(window_id);
        assert!(progress.is_paused());

        task.request_resume(5);

        let crate::ecs::TaskComponents::Scan { resume_request, .. } = &task.components;
        assert!(resume_request.is_some());

        world.insert(task);

        let task_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(task_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = task_entities.read().unwrap();
        for task in entities.iter() {
            let crate::ecs::TaskComponents::Scan {
                resume_request,
                progress,
                ..
            } = &task.components;
            assert!(resume_request.is_none(), "Resume request should be cleared");
            assert!(progress.is_scanning(), "Scan should be scanning");
        }
    }

    #[test]
    fn test_processes_both_requests_in_sequence() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);

        task.request_pause(3);
        world.insert(task);

        let task_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(task_entities.clone());

        system.run(&mut context).unwrap();

        {
            let entities = task_entities.read().unwrap();
            for task in entities.iter() {
                let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
                assert!(progress.is_paused());
            }
        }

        {
            let mut entities = task_entities.write().unwrap();
            for task in entities.iter_mut() {
                task.request_resume(3);
            }
        }

        system.run(&mut context).unwrap();

        let entities = task_entities.read().unwrap();
        for task in entities.iter() {
            let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
            assert!(progress.is_scanning());
        }
    }

    #[test]
    fn test_pause_with_station_starts_tune_transition() {
        let mut system = RequestProcessorSystem::new();

        let mut task_world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 0);
        progress.start_window(window_id);

        task.request_pause_with_station(0, 88.9e6, 88.9e6);
        task_world.insert(task);

        let task_entities = Arc::new(RwLock::new(task_world));

        let _signal = crate::core::types::Signal {
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

        let mut signal_world = EntityWorld::new();
        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        let mut signal_entity = SignalEntity::new(88.9e6, window_id);

        // Confirm the signal so it can be tuned
        signal_entity
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);

        signal_world.insert(signal_entity);

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let mut context = SystemContext::new()
            .with_task_entities(task_entities.clone())
            .with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let tasks = task_entities.read().unwrap();
        for task in tasks.iter() {
            let crate::ecs::TaskComponents::Scan {
                progress,
                pause_request,
                ..
            } = &task.components;
            assert!(progress.is_listening(), "Scan should be in listening mode");
            assert!(pause_request.is_none(), "Pause request should be cleared");
        }

        let signals = signal_entities.read().unwrap();
        for signal in signals.iter() {
            assert!(
                signal.tune_state.is_transitioning(),
                "Signal should have tune transition started"
            );
        }
    }

    #[test]
    fn test_pause_with_station_stops_previous_playback() {
        let mut system = RequestProcessorSystem::new();

        let mut task_world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 0);
        progress.start_listening(window_id);

        task.request_pause_with_station(1, 89.7e6, 89.7e6);
        task_world.insert(task);

        let task_entities = Arc::new(RwLock::new(task_world));

        let _signal1 = crate::core::types::Signal {
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

        let _signal2 = crate::core::types::Signal {
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

        let mut signal_world = EntityWorld::new();
        let task_id = TaskId::new("test-scan".to_string());
        let window_id1 = WindowId::new(task_id.clone(), 0);
        let mut _signal1_entity = SignalEntity::new(88.9e6, window_id1);

        // Confirm signals so they can be tuned
        _signal1_entity
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);
        _signal1_entity
            .playback
            .transition_to(crate::ecs::components::signal::PlaybackState::Playing);

        let window_id2 = WindowId::new(task_id, 1);
        let mut _signal2_entity = SignalEntity::new(89.7e6, window_id2);
        _signal2_entity
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);

        signal_world.insert(_signal1_entity);
        signal_world.insert(_signal2_entity);

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new()
            .with_task_entities(task_entities.clone())
            .with_signal_entities(signal_entities.clone())
            .with_audio_entities(audio_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let signals = signal_entities.read().unwrap();
        for signal in signals.iter() {
            assert!(
                signal.playback.state() != crate::ecs::components::signal::PlaybackState::Playing,
                "Previous playback should be stopped"
            );
        }
    }

    #[test]
    fn test_resume_clears_tuner_request_queue() {
        let mut system = RequestProcessorSystem::new();

        let mut task_world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 0);
        progress.start_listening(window_id);

        task.request_resume(0);
        task_world.insert(task);

        let task_entities = Arc::new(RwLock::new(task_world));

        let task_id = crate::ecs::TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id, 0);
        let tuner_request_queue =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        {
            let mut queue = tuner_request_queue.lock().unwrap();
            queue.push_back(crate::ecs::queue::TunerRequest {
                station_id: crate::ecs::StationId::new(),
                frequency: 88.9e6,
                window_id,
                center_frequency: 88.9e6,
            });
        }

        let mut context = SystemContext::new()
            .with_task_entities(task_entities.clone())
            .with_tuner_request_queue(tuner_request_queue.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let queue = tuner_request_queue.lock().unwrap();
        assert_eq!(queue.len(), 0, "Queue should be cleared on resume");

        let tasks = task_entities.read().unwrap();
        for task in tasks.iter() {
            let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
            assert!(progress.is_scanning(), "Scan should be in scanning mode");
        }
    }

    #[test]
    fn test_audio_graphs_canceled_before_clearing() {
        let mut system = RequestProcessorSystem::new();

        let mut task_world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 0);
        progress.start_listening(window_id);

        task.request_pause_with_station(1, 89.7e6, 89.7e6);
        task_world.insert(task);

        let task_entities = Arc::new(RwLock::new(task_world));

        let _signal1 = crate::core::types::Signal {
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

        let _signal2 = crate::core::types::Signal {
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

        let mut signal_world = EntityWorld::new();
        let task_id = TaskId::new("test-scan".to_string());
        let window_id1 = WindowId::new(task_id.clone(), 0);
        let mut _signal1_entity = SignalEntity::new(88.9e6, window_id1);
        _signal1_entity
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);

        let window_id2 = WindowId::new(task_id, 1);
        let mut _signal2_entity = SignalEntity::new(89.7e6, window_id2);
        _signal2_entity
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.7);

        signal_world.insert(_signal1_entity);
        signal_world.insert(_signal2_entity);

        let signal_entities = Arc::new(RwLock::new(signal_world));

        let mut audio_world = EntityWorld::new();
        let mut audio = crate::ecs::AudioEntity::new(_signal1, 88.9e6, None);

        let cancel_token = rustradio::graph::CancellationToken::new();
        let cancel_clone = cancel_token.clone();
        audio.allocation.graph_cancel = Some(cancel_token);
        audio_world.insert(audio);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_task_entities(task_entities.clone())
            .with_signal_entities(signal_entities)
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

        let mut task_world = EntityWorld::new();
        let mut task = create_test_task("test-scan", 10);
        let task_id = task.id().clone();

        let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
        let window_id = WindowId::new(task_id, 0);
        progress.start_listening(window_id);

        task.request_pause_with_station(2, 90.5e6, 90.5e6);
        task_world.insert(task);

        let task_entities = Arc::new(RwLock::new(task_world));

        let mut audio_world = EntityWorld::new();

        let _signal1 = crate::core::types::Signal {
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

        let _signal2 = crate::core::types::Signal {
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

        let mut audio1 = crate::ecs::AudioEntity::new(_signal1, 88.9e6, None);
        audio1.allocation.graph_cancel = Some(cancel1);

        let mut audio2 = crate::ecs::AudioEntity::new(_signal2, 89.7e6, None);
        audio2.allocation.graph_cancel = Some(cancel2);

        audio_world.insert(audio1);
        audio_world.insert(audio2);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_task_entities(task_entities)
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
