use std::sync::{Arc, RwLock};

use tracing::debug;

use crate::{
    audio::quality::AudioAnalyzer,
    core::{config::ScanningConfig, types::Result},
    ecs::{
        Entity, EntityWorld, SignalEntity, WindowEntity,
        system::{System, SystemContext},
    },
};

pub struct SignalAnalysisSystem {
    #[allow(dead_code)]
    analyzer: Arc<AudioAnalyzer>,
}

impl SignalAnalysisSystem {
    pub fn new(analyzer: Arc<AudioAnalyzer>) -> Self {
        SignalAnalysisSystem { analyzer }
    }
}

impl SignalAnalysisSystem {
    fn handle_signal_signal(
        signal: &mut SignalEntity,
        window_id: &crate::ecs::components::window::WindowId,
        quality: crate::audio::quality::AudioQuality,
        signal_strength: f64,
        window_entities: &Arc<RwLock<EntityWorld<WindowEntity>>>,
    ) {
        signal.analysis.confirm_analysis(quality, signal_strength);
        signal.info.set_audio_quality(Some(quality));
        signal.info.set_signal_strength(Some(signal_strength));

        debug!(
            signal_id = ?signal.id(),
            frequency_mhz = signal.frequency() / 1e6,
            audio_quality = ?quality,
            signal_strength = signal_strength,
            "SignalAnalysisSystem: Transitioned to Signal"
        );

        if signal.coordination.audio_request_enqueued() {
            debug!(
                signal_id = ?signal.id(),
                "SignalAnalysisSystem: Playback already queued, skipping re-enqueue"
            );
            return;
        }

        match window_entities.try_write() {
            Ok(mut windows) => {
                if let Some(window) = windows.iter_mut().find(|w| w.id() == window_id) {
                    window.lifecycle.complete_signal();
                    window.allocation.queue_for_playback(signal.id().clone());
                    signal.coordination.set_audio_request_enqueued(true);
                } else {
                    debug!(
                        signal_id = ?signal.id(),
                        window_id = ?window_id,
                        "SignalAnalysisSystem: Window not found for queuing playback"
                    );
                }
            }
            Err(_) => {
                debug!(
                    signal_id = ?signal.id(),
                    "SignalAnalysisSystem: Failed to acquire window write lock for queuing playback"
                );
            }
        }
    }

    fn handle_rejected_signal(
        signal: &mut SignalEntity,
        window_id: &crate::ecs::components::window::WindowId,
        quality: crate::audio::quality::AudioQuality,
        signal_strength: f64,
        window_entities: &Arc<RwLock<EntityWorld<WindowEntity>>>,
    ) {
        signal.analysis.reject_analysis(quality, signal_strength);

        let reason = if quality.is_audio() {
            "below squelch threshold"
        } else {
            "non-audio quality"
        };

        debug!(
            signal_id = ?signal.id(),
            frequency_mhz = signal.frequency() / 1e6,
            audio_quality = ?quality,
            signal_strength = signal_strength,
            reason = reason,
            "SignalAnalysisSystem: Transitioned to Rejected"
        );

        match window_entities.try_write() {
            Ok(mut windows) => {
                if let Some(window) = windows.iter_mut().find(|w| w.id() == window_id) {
                    window.lifecycle.complete_signal();
                    window.allocation.complete_analysis();
                } else {
                    debug!(
                        signal_id = ?signal.id(),
                        window_id = ?window_id,
                        "SignalAnalysisSystem: Window not found for complete_analysis"
                    );
                }
            }
            Err(_) => {
                debug!(
                    signal_id = ?signal.id(),
                    "SignalAnalysisSystem: Failed to acquire window write lock for complete_analysis"
                );
            }
        }
    }

    fn process_finished_threads(
        signal_entities: &Arc<RwLock<EntityWorld<SignalEntity>>>,
    ) -> Result<()> {
        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => {
                debug!("SignalAnalysisSystem: Failed to acquire write lock, skipping tick");
                return Ok(());
            }
        };

        let in_progress_count = signals
            .iter()
            .filter(|s| s.analysis.is_in_progress())
            .count();
        if in_progress_count > 0 {
            debug!(
                in_progress_count = in_progress_count,
                "SignalAnalysisSystem: Processing signals"
            );
        }

        for signal in signals.iter_mut() {
            if !signal.analysis.is_in_progress() {
                continue;
            }

            if let Some(results) = signal.analysis.try_receive_results() {
                signal
                    .analysis
                    .confirm_analysis(results.quality, results.strength);
                signal.info.set_audio_quality(Some(results.quality));
                signal.info.set_signal_strength(Some(results.strength));

                debug!(
                    signal_id = ?signal.id(),
                    frequency_mhz = signal.frequency() / 1e6,
                    quality = ?results.quality,
                    strength = results.strength,
                    "SignalAnalysisSystem: Thread completed successfully"
                );
            }
        }

        Ok(())
    }

    fn process_completed_signals(
        signal_entities: &Arc<RwLock<EntityWorld<SignalEntity>>>,
        window_entities: &Arc<RwLock<EntityWorld<WindowEntity>>>,
        config: &Arc<ScanningConfig>,
    ) -> Result<()> {
        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        for signal in signals.iter_mut() {
            if !signal.analysis.is_done() {
                continue;
            }

            let window_id = signal.window_id().clone();

            if signal.analysis.is_rejected() || signal.analysis.is_error() {
                if signal.coordination.rejection_reason().is_some() {
                    continue;
                }

                match window_entities.try_write() {
                    Ok(mut windows) => {
                        if let Some(window) = windows.iter_mut().find(|w| w.id() == &window_id) {
                            window.lifecycle.complete_signal();
                            window.allocation.complete_analysis();

                            let reason = if signal.analysis.is_rejected() {
                                "Analysis failed - below quality threshold".to_string()
                            } else {
                                "Analysis error".to_string()
                            };
                            signal.coordination.set_rejection_reason(Some(reason));

                            debug!(
                                signal_id = ?signal.id(),
                                "SignalAnalysisSystem: Notified window of failed/error signal"
                            );
                        }
                    }
                    Err(_) => {
                        continue;
                    }
                }
                continue;
            }

            let quality = match signal.info.audio_quality() {
                Some(q) => q,
                None => {
                    debug!(
                        signal_id = ?signal.id(),
                        "SignalAnalysisSystem: analysis marked complete but no quality set"
                    );
                    continue;
                }
            };

            let signal_strength = signal.info.signal_strength().unwrap_or(0.0);
            let threshold = config.audio.squelch.threshold;

            if quality.is_audio() && quality.meets_threshold(threshold) {
                Self::handle_signal_signal(
                    signal,
                    &window_id,
                    quality,
                    signal_strength,
                    window_entities,
                );
            } else {
                Self::handle_rejected_signal(
                    signal,
                    &window_id,
                    quality,
                    signal_strength,
                    window_entities,
                );
            }
        }

        Ok(())
    }
}

impl System for SignalAnalysisSystem {
    fn name(&self) -> &'static str {
        "SignalAnalysisSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let window_entities = match &context.window_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let config = match &context.config {
            Some(c) => c.clone(),
            None => return Ok(()),
        };

        let task_id = match &context.task_entities {
            Some(entities) => {
                let entities = entities.try_read().ok();
                entities.and_then(|e| e.iter().next().map(|t| t.id().clone()))
            }
            None => None,
        };

        if task_id.is_none() {
            return Ok(());
        }

        Self::process_finished_threads(&signal_entities)?;
        Self::process_completed_signals(&signal_entities, &window_entities, &config)?;

        Ok(())
    }
}

impl Default for SignalAnalysisSystem {
    fn default() -> Self {
        let classifier = Box::new(crate::audio::quality::heuristic2::Classifier::new(48000.0));
        let analyzer = AudioAnalyzer::new(classifier);
        Self::new(Arc::new(analyzer))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::RwLock;

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::signals::ModulationType,
        ecs::{EntityWorld, SignalEntity, TaskId, components::window::WindowId},
    };

    #[test]
    fn test_system_name() {
        let system = SignalAnalysisSystem::default();
        assert_eq!(system.name(), "SignalAnalysisSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = SignalAnalysisSystem::default();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_skips_signals_without_completed_analysis() {
        let mut system = SignalAnalysisSystem::default();
        let mut context = SystemContext::new();

        let task_id = TaskId::new("test".to_string());
        let window_id = WindowId::new(task_id, 0);
        let signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        let world = Arc::new(RwLock::new(EntityWorld::new()));
        world.write().unwrap().insert(signal);

        context = context.with_signal_entities(world.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Signal should still be in Detected state
        let signals = world.read().unwrap();
        let signal = signals.iter().next().unwrap();
        assert!(signal.analysis.is_not_started());
    }

    /// RED TEST: Reproduces bug where signals are re-processed every tick
    /// This causes the "Transitioned to Signal" log to repeat and audio to restart
    #[test]
    fn test_does_not_reprocess_completed_signals() {
        use crate::{
            core::config::ScanningConfig,
            ecs::{WindowEntity, system::System},
            hardware::{DeviceId, pool::TunerId},
        };

        let mut system = SignalAnalysisSystem::default();

        let task_id = TaskId::new("scan_1".to_string());
        let window_id = WindowId::new(task_id.clone(), 0);

        // Create a signal that has already completed analysis with Good quality
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 26.4);
        signal.info.set_audio_quality(Some(AudioQuality::Good));
        signal.info.set_signal_strength(Some(26.4));

        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
        signal_entities.write().unwrap().insert(signal);

        // Create window entity
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);
        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(1);
        window.allocation.start_active(tuner_id, 1);
        window.allocation.mark_all_spawned();

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        window_entities.write().unwrap().insert(window);

        let mut task_world = EntityWorld::new();
        task_world.insert(crate::ecs::TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            crate::ecs::ScanTaskData::Placeholder,
            10,
        ));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_window_entities(window_entities.clone())
            .with_task_entities(Arc::new(RwLock::new(task_world)))
            .with_config(Arc::new(ScanningConfig::default()));

        // First run: should process the signal (no longer creates station)
        system.run(&mut context).unwrap();

        // Get initial window lifecycle count
        let lifecycle_count_first = {
            let windows = window_entities.read().unwrap();
            let window = windows.iter().next().unwrap();
            format!("{:?}", window.lifecycle)
        };

        // Second run: should NOT re-process the same signal
        system.run(&mut context).unwrap();

        // Verify signal isn't re-processed and no stations are created
        let lifecycle_count_second = {
            let windows = window_entities.read().unwrap();
            let window = windows.iter().next().unwrap();
            format!("{:?}", window.lifecycle)
        };

        assert_eq!(
            lifecycle_count_first, lifecycle_count_second,
            "BUG: Window lifecycle changed on second run! complete_signal() was called again."
        );
    }

    #[test]
    fn test_signal_enqueued_only_once() {
        use crate::{
            core::config::ScanningConfig,
            ecs::{WindowEntity, components::window::WindowAllocationComponent, system::System},
            hardware::{DeviceId, pool::TunerId},
        };

        let mut system = SignalAnalysisSystem::default();

        let task_id = TaskId::new("scan_guard".to_string());
        let window_id = WindowId::new(task_id.clone(), 0);

        let mut signal = SignalEntity::new(102.5e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 12.5);
        signal.info.set_audio_quality(Some(AudioQuality::Good));
        signal.info.set_signal_strength(Some(12.5));

        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
        signal_entities.write().unwrap().insert(signal);

        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 101.8e6);
        let device_id = DeviceId::from_serial("sdrplay", "queue-test");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(1);
        window.allocation.start_active(tuner_id, 1);
        window.allocation.mark_all_spawned();

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        window_entities.write().unwrap().insert(window);

        let mut task_world = EntityWorld::new();
        task_world.insert(crate::ecs::TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            crate::ecs::ScanTaskData::Placeholder,
            1,
        ));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_window_entities(window_entities.clone())
            .with_task_entities(Arc::new(RwLock::new(task_world)))
            .with_config(Arc::new(ScanningConfig::default()));

        system.run(&mut context).unwrap();

        {
            let windows = window_entities.read().unwrap();
            let window = windows.iter().next().unwrap();
            if let WindowAllocationComponent::Active { playback_queue, .. } = &window.allocation {
                assert_eq!(
                    playback_queue.len(),
                    1,
                    "Expected exactly one signal in playback queue after first run"
                );
            } else {
                panic!("Window allocation should be Active after processing signal");
            }
        }

        {
            let signals = signal_entities.read().unwrap();
            let signal = signals.iter().next().unwrap();
            assert!(
                signal.coordination.audio_request_enqueued(),
                "Signal should be marked as having an enqueued audio request"
            );
        }

        system.run(&mut context).unwrap();

        let windows = window_entities.read().unwrap();
        let window = windows.iter().next().unwrap();
        if let WindowAllocationComponent::Active { playback_queue, .. } = &window.allocation {
            assert_eq!(
                playback_queue.len(),
                1,
                "Playback queue should not accumulate duplicate entries across runs"
            );
        } else {
            panic!("Window allocation should remain Active after second run");
        }
    }

    #[test]
    fn test_rejects_non_audio_quality() {
        let mut system = SignalAnalysisSystem::default();
        let mut context = SystemContext::new();

        let task_id = TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Static, 0.5);
        signal.info.set_audio_quality(Some(AudioQuality::Static));
        signal.info.set_signal_strength(Some(0.5));

        let world = Arc::new(RwLock::new(EntityWorld::new()));
        world.write().unwrap().insert(signal);

        let mut task_world = EntityWorld::new();
        task_world.insert(crate::ecs::TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            crate::ecs::ScanTaskData::Placeholder,
            10,
        ));

        let window_world = Arc::new(RwLock::new(EntityWorld::new()));
        let mut window = crate::ecs::WindowEntity::new(window_id.clone(), task_id, 88.9e6);
        window.lifecycle.start_analyzing(1);
        window_world.write().unwrap().insert(window);

        context = context.with_signal_entities(world.clone());
        context = context.with_window_entities(window_world.clone());
        context = context.with_task_entities(Arc::new(RwLock::new(task_world)));
        context = context.with_config(Arc::new(crate::core::config::ScanningConfig::default()));

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Signal should be transitioned to Rejected
        let signals = world.read().unwrap();
        let signal = signals.iter().next().unwrap();
        assert!(
            signal.analysis.is_rejected(),
            "Signal rejected. quality: {:?}, analysis_complete: {}",
            signal.info.audio_quality(),
            signal.analysis.is_done()
        );
    }

    #[test]
    fn test_joins_finished_analysis_threads() {
        let mut system = SignalAnalysisSystem::default();
        let mut context = SystemContext::new();

        let task_id = TaskId::new("test".to_string());
        let window_id = WindowId::new(task_id, 0);
        let mut signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        // Create result channel and barrier for synchronization
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = barrier.clone();

        // Spawn a thread that returns analysis results
        let handle = std::thread::spawn(move || {
            use crate::ecs::components::AnalysisResults;
            let results = AnalysisResults {
                quality: AudioQuality::Good,
                strength: 0.8,
            };
            // Send results through channel
            let _ = result_tx.send(results.clone());
            // Signal that result has been sent
            barrier_clone.wait();
            Ok(results)
        });

        // Start analysis (transition to InProgress)
        signal.analysis.start_analysis(handle, result_rx);
        assert!(signal.analysis.is_in_progress());

        // Wait for thread to send result through channel
        barrier.wait();

        let world = Arc::new(RwLock::new(EntityWorld::new()));
        world.write().unwrap().insert(signal);

        let mut task_world = EntityWorld::new();
        task_world.insert(crate::ecs::TaskEntity::new_scan_with_defaults(
            crate::ecs::TaskId::new("test-scan".to_string()),
            crate::ecs::ScanTaskData::Placeholder,
            10,
        ));

        let window_world = Arc::new(RwLock::new(EntityWorld::new()));
        let mut window = crate::ecs::WindowEntity::new(
            crate::ecs::components::window::WindowId::new(
                crate::ecs::TaskId::new("test-scan".to_string()),
                0,
            ),
            crate::ecs::TaskId::new("test-scan".to_string()),
            88.9e6,
        );
        window.lifecycle.start_analyzing(1);
        window_world.write().unwrap().insert(window);

        context = context.with_signal_entities(world.clone());
        context = context.with_window_entities(window_world.clone());
        context = context.with_task_entities(Arc::new(RwLock::new(task_world)));
        context = context.with_config(Arc::new(crate::core::config::ScanningConfig::default()));

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Signal should be transitioned to Complete from InProgress
        let signals = world.read().unwrap();
        let signal = signals.iter().next().unwrap();
        assert!(
            signal.analysis.is_done(),
            "Expected signal analysis to be Complete after joining thread"
        );

        use crate::ecs::components::AnalysisStatus;
        let status = signal.status();
        assert!(
            matches!(status, AnalysisStatus::Signal {
                quality: AudioQuality::Good,
                ..
            }),
            "Expected analysis results to be extracted and stored with Good quality"
        );
    }

    /// REGRESSION TEST: Verifies Failed signals decrement window counter
    /// This reproduces the bug where Failed signals were skipped in Phase 2,
    /// preventing the window's signals_analyzing counter from decrementing,
    /// which blocked window completion and prevented next windows from starting.
    #[test]
    fn test_failed_signals_decrement_window_counter() {
        use crate::{
            core::config::ScanningConfig,
            ecs::{WindowEntity, system::System},
            hardware::{DeviceId, pool::TunerId},
        };

        let mut system = SignalAnalysisSystem::default();

        let task_id = TaskId::new("scan_1".to_string());
        let window_id = WindowId::new(task_id.clone(), 0);

        // Create a signal that has FAILED analysis (Poor quality)
        let mut signal = SignalEntity::new(87.1e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.reject_analysis(AudioQuality::Poor, 0.2);
        signal.info.set_audio_quality(Some(AudioQuality::Poor));
        signal.info.set_signal_strength(Some(0.2));

        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
        signal_entities.write().unwrap().insert(signal);

        // Create window entity with counter initialized to 1
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);
        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(1);
        window.allocation.start_active(tuner_id, 1);
        window.allocation.mark_all_spawned();

        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
        window_entities.write().unwrap().insert(window);

        let mut task_world = EntityWorld::new();
        task_world.insert(crate::ecs::TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            crate::ecs::ScanTaskData::Placeholder,
            10,
        ));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_window_entities(window_entities.clone())
            .with_task_entities(Arc::new(RwLock::new(task_world)))
            .with_config(Arc::new(ScanningConfig::default()));

        // Run the system - should process Failed signal
        let result = system.run(&mut context);
        assert!(result.is_ok());

        // REGRESSION CHECK: Window counter should be decremented
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).expect("Window should exist");

        // Before the fix, signals_analyzing would stay at 1 because Failed
        // signals were skipped in Phase 2. After the fix, it should be 0.
        let segment_exists = window.segment.is_some();
        assert!(
            window.allocation.is_ready_to_complete(segment_exists),
            "BUG REGRESSION: Failed signal should decrement window counter! Before fix, Phase 2 \
             only processed Complete signals, so Failed signals never called complete_analysis() \
             and counter stayed stuck. This prevented windows from completing and blocked scan \
             progress."
        );

        // Verify rejection_reason is set to prevent reprocessing
        let signals = signal_entities.read().unwrap();
        let signal = signals.iter().next().unwrap();
        assert!(
            signal.coordination.rejection_reason().is_some(),
            "Rejection reason should be set to mark signal as processed"
        );
    }
}
