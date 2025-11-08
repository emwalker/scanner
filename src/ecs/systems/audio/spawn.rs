use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        system::{System, SystemContext},
    },
    hardware::pool::SegmentTrait,
    scanning::window::spawn_audio_entity,
};

/// System that spawns audio playback for validated signal signals
///
/// Flow:
/// 1. Windows queue signals after analysis when they pass quality checks
/// 2. This system pops one signal per window when no audio is playing
/// 3. An audio graph is created for that signal and playback begins
/// 4. Stations and signal playback state are updated to reflect the change
pub struct AudioSpawnSystem;

impl Default for AudioSpawnSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSpawnSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for AudioSpawnSystem {
    fn name(&self) -> &'static str {
        "AudioSpawn"
    }

    #[allow(clippy::cognitive_complexity)]
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        // Don't spawn audio during global pause
        if context.is_globally_paused() {
            return Ok(());
        }

        let (signal_entities, audio_entities, window_entities) = match (
            &context.signal_entities,
            &context.audio_entities,
            &context.window_entities,
        ) {
            (Some(se), Some(ae), Some(we)) => (se.clone(), ae.clone(), we.clone()),
            _ => return Ok(()),
        };

        let config = match &context.config {
            Some(c) => c.clone(),
            None => return Ok(()),
        };

        // Phase 1: Dequeue playback requests from window allocation state
        let playback_requests = {
            let mut requests = Vec::new();
            let mut windows = match window_entities.try_write() {
                Ok(w) => w,
                Err(_) => return Ok(()), // Shutdown or lock contention
            };

            for window in windows.iter_mut() {
                if !window.allocation.is_active() {
                    continue;
                }

                if let Some(signal_id) = window.allocation.start_playing_next() {
                    requests.push((window.id().clone(), signal_id));
                }
            }

            requests
        };

        if playback_requests.is_empty() {
            return Ok(());
        }

        debug!(
            count = playback_requests.len(),
            "AudioSpawnSystem: Dequeued signals for audio playback"
        );

        // Phase 2: Prepare signal data (single lock on signal entities)
        let mut signals_guard = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        let mut spawn_jobs = Vec::new();
        let mut failed_requests = Vec::new();

        for (window_id, signal_id) in &playback_requests {
            // Find signal by matching signal ID string (during dual-write phase)
            let signal = signals_guard
                .iter_mut()
                .find(|s| s.id().as_str() == signal_id.as_str());

            match signal {
                Some(signal) => {
                    if !signal.analysis.is_confirmed() {
                        debug!(
                            signal_id = ?signal_id,
                            "AudioSpawnSystem: Signal not confirmed for playback"
                        );
                        failed_requests.push((window_id.clone(), signal_id.clone()));
                        continue;
                    }

                    let (quality, strength) =
                        match (signal.info.audio_quality(), signal.info.signal_strength()) {
                            (Some(q), Some(s)) => (q, s),
                            _ => {
                                debug!(
                                    signal_id = ?signal_id,
                                    "AudioSpawnSystem: Signal missing quality/strength"
                                );
                                failed_requests.push((window_id.clone(), signal_id.clone()));
                                continue;
                            }
                        };

                    let center_freq = {
                        let windows = match window_entities.try_read() {
                            Ok(w) => w,
                            Err(_) => {
                                debug!(
                                    signal_id = ?signal_id,
                                    "AudioSpawnSystem: Failed to read window entities for center frequency"
                                );
                                failed_requests.push((window_id.clone(), signal_id.clone()));
                                continue;
                            }
                        };

                        match windows.iter().find(|w| w.id() == window_id) {
                            Some(w) => w.center_frequency_hz(),
                            None => {
                                debug!(
                                    signal_id = ?signal_id,
                                    window_id = ?window_id,
                                    "AudioSpawnSystem: Window not found for center frequency"
                                );
                                failed_requests.push((window_id.clone(), signal_id.clone()));
                                continue;
                            }
                        }
                    };

                    let job = (
                        window_id.clone(),
                        signal_id.clone(),
                        signal.frequency(),
                        center_freq,
                        quality,
                        strength,
                    );

                    signal
                        .playback
                        .transition_to(crate::ecs::components::signal::PlaybackState::Playing);
                    spawn_jobs.push(job);
                }
                None => {
                    debug!(
                        signal_id = ?signal_id,
                        "AudioSpawnSystem: signal not found for playback"
                    );
                    failed_requests.push((window_id.clone(), signal_id.clone()));
                }
            }
        }

        drop(signals_guard);

        if spawn_jobs.is_empty() && failed_requests.is_empty() {
            return Ok(());
        }

        let mut signal_reverts = Vec::new();

        // Phase 3: Spawn audio for prepared jobs
        for (window_id, signal_id, freq, center_freq, quality, strength) in spawn_jobs {
            let sdr_rx = {
                let windows = match window_entities.try_read() {
                    Ok(w) => w,
                    Err(_) => {
                        signal_reverts.push(signal_id.clone());
                        failed_requests.push((window_id.clone(), signal_id.clone()));
                        continue;
                    }
                };

                windows
                    .iter()
                    .find(|w| w.id() == &window_id)
                    .and_then(|w| w.segment.as_ref())
                    .map(|seg| seg.segment().audio_subscriber())
            };

            let sdr_rx = match sdr_rx {
                Some(rx) => rx,
                None => {
                    debug!(
                        signal_id = ?signal_id,
                        window_id = ?window_id,
                        "AudioSpawnSystem: Window segment not available"
                    );
                    signal_reverts.push(signal_id.clone());
                    failed_requests.push((window_id.clone(), signal_id.clone()));
                    continue;
                }
            };

            let signal = crate::core::types::Signal::new_fm(
                freq,
                strength as f32,
                200_000.0,
                48000,
                100,
                center_freq,
                quality,
            );

            match spawn_audio_entity(signal, sdr_rx, &config, center_freq) {
                Ok((audio_entity, stream)) => {
                    let audio_id = *audio_entity.id();

                    if let Ok(mut audio) = audio_entities.try_write() {
                        audio.insert(audio_entity);
                    }

                    if let Some(streams_resource) = &context.audio_streams
                        && let Ok(mut streams) = streams_resource.try_lock()
                    {
                        streams.insert(audio_id, stream);
                    }

                    debug!(
                        signal_id = ?signal_id,
                        frequency_mhz = freq / 1e6,
                        audio_id = ?audio_id,
                        "AudioSpawnSystem: Spawned audio for signal"
                    );
                }
                Err(e) => {
                    debug!(
                        signal_id = ?signal_id,
                        frequency_mhz = freq / 1e6,
                        error = %e,
                        "AudioSpawnSystem: Failed to spawn audio"
                    );
                    signal_reverts.push(signal_id.clone());
                    failed_requests.push((window_id.clone(), signal_id.clone()));
                }
            }
        }

        // Phase 4: Revert signal playback state for failed jobs
        if !signal_reverts.is_empty()
            && let Ok(mut signals) = signal_entities.try_write()
        {
            for signal_id in signal_reverts {
                if let Some(signal) = signals
                    .iter_mut()
                    .find(|s| s.id().as_str() == signal_id.as_str())
                {
                    signal
                        .playback
                        .transition_to(crate::ecs::components::signal::PlaybackState::NotPlaying);
                }
            }
        }

        // Phase 5: Return failed signals to their playback queue
        if !failed_requests.is_empty()
            && let Ok(mut windows) = window_entities.try_write()
        {
            for (window_id, signal_id) in failed_requests {
                if let Some(window) = windows.iter_mut().find(|w| w.id() == &window_id) {
                    window.allocation.return_playback_signal(signal_id.clone());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::signals::ModulationType,
        ecs::{EntityWorld, SignalEntity, TaskId, WindowId},
    };

    #[test]
    fn test_system_creation() {
        let system = AudioSpawnSystem::new();
        assert_eq!(system.name(), "AudioSpawn");
    }

    #[test]
    fn test_run_with_no_signals() {
        let mut system = AudioSpawnSystem::new();
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities)
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_complete_signal() {
        let mut system = AudioSpawnSystem::new();

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id.clone(), 1);
        let mut signal = SignalEntity::new(88.5e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.8);

        let mut signal_world = EntityWorld::new();
        signal_world.insert(signal);

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify signal was found (skeleton doesn't modify yet)
        let signals = signal_entities.read().unwrap();
        assert_eq!(signals.len(), 1);
    }

    #[test]
    fn test_system_respects_global_pause() {
        use std::sync::Mutex;

        use crate::ecs::{GlobalPauseState, SignalId, WindowEntity};

        let mut system = AudioSpawnSystem::new();

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id.clone(), 1);

        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.5e6);
        window.lifecycle.complete_signal();
        let signal_id = SignalId::new(88.5e6, ModulationType::WFM);
        window.allocation.queue_for_playback(signal_id.clone());

        let mut window_world = EntityWorld::new();
        window_world.insert(window);

        let mut signal = SignalEntity::new(88.5e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.8);

        let mut signal_world = EntityWorld::new();
        signal_world.insert(signal);

        let window_entities = Arc::new(RwLock::new(window_world));
        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let global_pause = Arc::new(Mutex::new(GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));

        let mut context = SystemContext::new()
            .with_window_entities(window_entities)
            .with_signal_entities(signal_entities)
            .with_audio_entities(audio_entities.clone())
            .with_global_pause_resource(global_pause);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let audio_world = audio_entities.read().unwrap();
        assert_eq!(
            audio_world.len(),
            0,
            "AudioSpawnSystem should not spawn audio during global pause"
        );
    }

    #[test]
    fn test_system_handles_missing_window_entities() {
        use std::time::SystemTime;

        use crate::{
            core::types::{ModulationType, Signal},
            ecs::{StationEntity, TaskId},
        };

        let mut system = AudioSpawnSystem::new();

        let task_id = TaskId::new("test");
        let window_id = WindowId::new(task_id, 1);
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.8);

        let mut signal_world = EntityWorld::new();
        signal_world.insert(signal);

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
        let station = StationEntity::from_signal(&signal, window_id);
        let mut station_world = EntityWorld::new();
        station_world.insert(station);

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let _station_entities = Arc::new(RwLock::new(station_world));

        let config = Arc::new(crate::core::types::ScanningConfig::default());

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_audio_entities(audio_entities.clone())
            .with_config(config);

        let result = system.run(&mut context);
        assert!(
            result.is_ok(),
            "System should gracefully handle missing window entities"
        );

        let audio_world = audio_entities.read().unwrap();
        assert_eq!(
            audio_world.len(),
            0,
            "No AudioEntity created without window segment"
        );
    }
}
