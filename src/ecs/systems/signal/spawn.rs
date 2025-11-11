//! System that spawns signal analysis threads
//!
//! This system finds SignalEntity in NotStarted state with analysis_input
//! components, spawns analysis threads, and transitions them to InProgress state.
//!
//! This system uses mpsc channels internally to communicate between the
//! streaming graph (SquelchBlock) and the analysis thread. These channels
//! are an implementation detail and not exposed in the ECS public API.

use std::{sync::mpsc, thread, time::Duration};

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::signal::AnalysisResults,
        system::{System, SystemContext},
    },
    pipeline::AnalysisContext,
};

pub struct SignalAnalysisSpawnSystem;

impl SignalAnalysisSpawnSystem {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SignalAnalysisSpawnSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for SignalAnalysisSpawnSystem {
    fn name(&self) -> &'static str {
        "CandidateAnalysisSpawnSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        // Phase 1: Collect signals to spawn (short read lock)
        let signals_to_spawn = {
            let entities = match signal_entities.try_read() {
                Ok(e) => e,
                Err(_) => return Ok(()), // Shutdown or lock contention
            };

            entities
                .iter()
                .filter(|e| e.analysis.is_not_started() && e.analysis_input.is_some())
                .map(|e| (e.id().clone(), e.frequency()))
                .collect::<Vec<_>>()
        }; // Lock released here

        if signals_to_spawn.is_empty() {
            return Ok(());
        }

        // Phase 2: Extract inputs and spawn threads (no lock held)
        let mut spawned_threads = Vec::new();

        for (signal_id, frequency_hz) in signals_to_spawn {
            // Extract inputs with minimal lock time
            let inputs = {
                let mut entities = match signal_entities.try_write() {
                    Ok(e) => e,
                    Err(_) => continue, // Skip if lock contention
                };

                let entity = match entities.get_mut(&signal_id) {
                    Some(e) => e,
                    None => continue,
                };

                match entity.take_analysis_input() {
                    Some(inputs) => inputs,
                    None => continue,
                }
            }; // Lock released here

            let (result_tx, result_rx) = mpsc::channel();
            let signal_id_for_thread = signal_id.clone();

            // Spawn analysis thread (NO LOCK HELD)
            let handle = thread::spawn(move || -> Result<AnalysisResults> {
                // Check pause signal before starting
                if let Some(ref signal) = inputs.pause_signal
                    && signal.is_paused()
                {
                    debug!("Candidate thread exiting early due to pause signal");
                    return Err(crate::core::types::ScannerError::Custom(
                        "Paused".to_string(),
                    ));
                }

                // Create internal signal channel
                let (signal_tx, signal_rx) = mpsc::channel();

                let context = AnalysisContext {
                    config: &inputs.config,
                    center_freq: inputs.center_freq,
                    window_id: inputs.window_id.clone(),
                };

                // Run pipeline (signal_tx is internal implementation detail)
                crate::pipeline::process_peak_to_signal(
                    frequency_hz,
                    inputs.sdr_rx_refining,
                    inputs.sdr_rx_detection,
                    signal_tx,
                    &context,
                )?;

                // Wait for result from pipeline
                let results = if let Ok(signal) = signal_rx.recv_timeout(Duration::from_secs(2)) {
                    debug!(
                        signal_id = ?signal_id_for_thread,
                        frequency_mhz = frequency_hz / 1e6,
                        quality = ?signal.audio_quality,
                        strength = signal.signal_strength,
                        "Analysis thread: signal analysis complete"
                    );

                    AnalysisResults {
                        quality: signal.audio_quality,
                        strength: signal.signal_strength as f64,
                    }
                } else {
                    debug!(
                        signal_id = ?signal_id_for_thread,
                        frequency_mhz = frequency_hz / 1e6,
                        "Analysis thread: no signal produced (noise/timeout)"
                    );

                    AnalysisResults {
                        quality: crate::audio::quality::AudioQuality::NoAudio,
                        strength: 0.0,
                    }
                };

                // Send results through channel (non-blocking communication)
                let _ = result_tx.send(results.clone());

                Ok(results)
            });

            debug!(
                signal_id = ?signal_id,
                frequency_mhz = frequency_hz / 1e6,
                "CandidateAnalysisSpawnSystem: Spawned analysis thread"
            );

            // Verify channel is connected before storing
            if result_rx.try_recv().is_err() {
                debug!(
                    signal_id = ?signal_id,
                    "Channel connected and ready"
                );
            }

            spawned_threads.push((signal_id, handle, result_rx, frequency_hz));
        }

        // Phase 3: Store handles (short write lock)
        if !spawned_threads.is_empty() {
            let mut entities = match signal_entities.try_write() {
                Ok(e) => e,
                Err(_) => return Ok(()), // Shutdown, threads will finish on their own
            };

            let mut windows_to_mark = std::collections::HashSet::new();

            for (signal_id, handle, result_rx, _frequency_hz) in spawned_threads {
                if let Some(entity) = entities.get_mut(&signal_id) {
                    // Check if thread already finished
                    if handle.is_finished() {
                        debug!(
                            signal_id = ?signal_id,
                            "WARNING: Thread finished before handle was stored"
                        );
                    }
                    entity.analysis.start_analysis(handle, result_rx);
                    debug!(
                        signal_id = ?signal_id,
                        "Stored analysis handle and receiver"
                    );

                    // Track which window this signal belongs to
                    windows_to_mark.insert(entity.window_id().clone());
                }
            }

            // Phase 4: Mark windows as all_spawned if no more NotStarted signals
            for window_id in windows_to_mark {
                let has_more_not_started = entities
                    .iter()
                    .any(|e| e.window_id() == &window_id && e.analysis.is_not_started());

                if !has_more_not_started {
                    // All signals for this window have been spawned
                    if let Some(window_entities) = &context.window_entities
                        && let Ok(mut windows) = window_entities.try_write()
                        && let Some(window) = windows.get_mut(&window_id)
                    {
                        window.allocation.mark_all_spawned();
                        debug!(
                            window_id = ?window_id,
                            "CandidateAnalysisSpawnSystem: All signals spawned for window"
                        );
                    }
                }
            }
        } // Lock released here

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::{
        core::signals::ModulationType,
        ecs::{EntityWorld, SignalEntity, TaskId, WindowId},
    };

    #[test]
    fn test_spawn_system_ignores_entities_without_input() {
        let mut system = SignalAnalysisSpawnSystem::new();

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id, 0);
        let entity = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
        let mut world = EntityWorld::new();
        world.insert(entity);

        let signal_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = signal_entities.read().unwrap();
        let entity = entities.iter().next().unwrap();
        assert!(entity.analysis.is_not_started());
    }
}
