use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use indexmap::IndexMap;

use crate::{
    ecs::{
        AudioEntity, Entities, Entity, EntityWorld, SignalEntity, System, SystemContext,
        components::window::WindowId,
    },
    hardware::pool::{PoolStatus, TunerActivity, TunerId, TunerState, TunerStatus},
    ui::tui::model::Model,
};

/// Backward compatibility enum for tests that use old combined state model
#[derive(Debug, Clone, Copy)]
pub enum TestSignalState {
    Detected,
    Analyzing,
    Signal,
    Playing,
    Completed,
    Rejected,
}

pub fn create_test_pool_status(
    available: Vec<crate::hardware::DeviceId>,
    scanning: Vec<crate::hardware::DeviceId>,
    listening: Vec<crate::hardware::DeviceId>,
) -> PoolStatus {
    let mut tuners = Vec::new();

    for device_id in available.iter() {
        let is_scanning = scanning.contains(device_id);
        let is_listening = listening.contains(device_id);

        let (state, activity) = if is_scanning {
            (TunerState::Allocated, Some(TunerActivity::Scanning))
        } else if is_listening {
            (TunerState::Allocated, Some(TunerActivity::Listening))
        } else {
            (TunerState::Available, None)
        };

        tuners.push(TunerStatus {
            id: TunerId {
                device_id: device_id.clone(),
                channel_index: 0,
            },
            state,
            activity,
        });
    }

    let available_count = available
        .iter()
        .filter(|id| !scanning.contains(id) && !listening.contains(id))
        .count();
    let allocated_count = scanning.len() + listening.len();

    PoolStatus {
        tuners,
        available_tuner_count: available_count,
        allocated_tuner_count: allocated_count,
        device_count: available.len(),
    }
}

/// Test context that manages ECS entities and syncs them to Model
pub struct ModelTestContext {
    pub model: Model,
    pub audio_entities: Entities<AudioEntity>,
    pub signal_entities: Entities<SignalEntity>,
    ui_update_system: crate::ecs::systems::UIUpdateSystem,
}

impl ModelTestContext {
    pub fn new() -> Self {
        Self {
            model: Model::new(),
            audio_entities: Arc::new(RwLock::new(EntityWorld::new())),
            signal_entities: Arc::new(RwLock::new(EntityWorld::new())),
            ui_update_system: crate::ecs::systems::UIUpdateSystem::new(),
        }
    }

    /// Create or update a signal entity with the given state (backward compatible with old state
    /// model)
    #[allow(dead_code)]
    pub fn update_signal(
        &mut self,
        frequency_hz: f64,
        window_id: usize,
        state: TestSignalState,
        audio_quality: Option<crate::audio::quality::AudioQuality>,
        signal_strength: Option<f64>,
    ) {
        use crate::{
            audio::quality::AudioQuality,
            ecs::{SignalId, TaskId, WindowId, components::PlaybackState},
        };

        let task_id = TaskId::new("test-task");
        let window_identifier = WindowId::new(task_id.clone(), window_id);
        let id = SignalId::new(frequency_hz, window_identifier.clone());
        let mut entities = self.signal_entities.write().unwrap();

        // Create entity if it doesn't exist
        if entities.get(&id).is_none() {
            entities.insert(SignalEntity::new(frequency_hz, window_identifier.clone()));
        }

        if let Some(entity) = entities.get_mut(&id) {
            // Map old combined state to new orthogonal states
            match state {
                TestSignalState::Detected => {
                    // NotStarted analysis, NotPlaying playback
                    // (no changes needed - default state)
                }
                TestSignalState::Analyzing => {
                    // InProgress analysis, NotPlaying playback
                    // Simulate in-progress by creating a completed state for now
                    // (proper implementation would need thread handle, but tests don't need it)
                    entity.analysis.confirm_analysis(AudioQuality::Good, 0.5);
                }
                TestSignalState::Signal => {
                    // Complete analysis, NotPlaying playback
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    entity.analysis.confirm_analysis(quality, strength);
                    entity.info.set_audio_quality(Some(quality));
                    entity.info.set_signal_strength(Some(strength));
                }
                TestSignalState::Playing => {
                    // Complete analysis, Playing playback - will handle audio entity after dropping
                    // lock
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    entity.analysis.confirm_analysis(quality, strength);
                    entity.info.set_audio_quality(Some(quality));
                    entity.info.set_signal_strength(Some(strength));
                    entity.playback.transition_to(PlaybackState::Playing);
                }
                TestSignalState::Completed => {
                    // Complete analysis, Completed playback
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    entity.analysis.confirm_analysis(quality, strength);
                    entity.info.set_audio_quality(Some(quality));
                    entity.info.set_signal_strength(Some(strength));
                    entity.playback.transition_to(PlaybackState::Completed);

                    let mut audio_entities = self.audio_entities.write().unwrap();
                    let ids_to_remove: Vec<_> = audio_entities
                        .iter()
                        .filter(|e| (e.frequency() - frequency_hz).abs() < 1000.0)
                        .map(|e| *e.id())
                        .collect();
                    for id in ids_to_remove {
                        audio_entities.remove(&id);
                    }
                }
                TestSignalState::Rejected => {
                    // Failed analysis, NotPlaying playback
                    let quality =
                        audio_quality.unwrap_or(crate::audio::quality::AudioQuality::Poor);
                    let strength = signal_strength.unwrap_or(0.1);
                    entity.analysis.reject_analysis(quality, strength);
                }
            }
        }
        drop(entities);

        // Mirror changes to SignalEntity
        let mut signal_entities = self.signal_entities.write().unwrap();
        let signal_id = SignalId::new(frequency_hz, window_identifier.clone());

        if signal_entities.get(&signal_id).is_none() {
            signal_entities.insert(SignalEntity::new(frequency_hz, window_identifier.clone()));
        }

        if let Some(signal) = signal_entities.get_mut(&signal_id) {
            use crate::ecs::components::signal::PlaybackState as SignalPlaybackState;

            match state {
                TestSignalState::Detected => {}
                TestSignalState::Analyzing => {
                    signal.analysis.confirm_analysis(AudioQuality::Good, 0.5);
                }
                TestSignalState::Signal => {
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    signal.analysis.confirm_analysis(quality, strength);
                    signal.info.set_audio_quality(Some(quality));
                    signal.info.set_signal_strength(Some(strength));
                }
                TestSignalState::Playing => {
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    signal.analysis.confirm_analysis(quality, strength);
                    signal.info.set_audio_quality(Some(quality));
                    signal.info.set_signal_strength(Some(strength));
                    signal.playback.transition_to(SignalPlaybackState::Playing);
                }
                TestSignalState::Completed => {
                    let quality = audio_quality.unwrap_or(AudioQuality::Good);
                    let strength = signal_strength.unwrap_or(0.5);
                    signal.analysis.confirm_analysis(quality, strength);
                    signal.info.set_audio_quality(Some(quality));
                    signal.info.set_signal_strength(Some(strength));
                    signal
                        .playback
                        .transition_to(SignalPlaybackState::Completed);
                }
                TestSignalState::Rejected => {
                    let quality = audio_quality.unwrap_or(AudioQuality::Poor);
                    let strength = signal_strength.unwrap_or(0.1);
                    signal.analysis.reject_analysis(quality, strength);
                }
            }
        }
        drop(signal_entities);

        // Handle audio entity creation/removal after dropping signal_entities lock
        if matches!(state, TestSignalState::Playing) {
            let quality = audio_quality.unwrap_or(crate::audio::quality::AudioQuality::Good);
            let strength = signal_strength.unwrap_or(0.5);

            let signal = crate::core::types::Signal {
                frequency_hz,
                signal_strength: strength as f32,
                bandwidth_hz: 200_000.0,
                modulation: crate::core::types::ModulationType::WFM,
                audio_sample_rate: 48000,
                detected_at: std::time::SystemTime::now(),
                analysis_duration_ms: 100,
                detection_center_freq: frequency_hz,
                audio_quality: quality,
            };

            let mut audio_entities = self.audio_entities.write().unwrap();
            let previously_playing_freqs: Vec<f64> =
                audio_entities.iter().map(|e| e.frequency()).collect();
            audio_entities.clear();
            audio_entities.insert(AudioEntity::new(signal, frequency_hz, None));
            drop(audio_entities);

            // Transition previously playing signals to Completed
            let mut signal_entities = self.signal_entities.write().unwrap();
            for prev_freq in &previously_playing_freqs {
                if (prev_freq - frequency_hz).abs() >= 1000.0 {
                    for signal in signal_entities.iter_mut() {
                        if (signal.frequency() - prev_freq).abs() < 1000.0 {
                            signal
                                .playback
                                .transition_to(crate::ecs::components::PlaybackState::Completed);
                        }
                    }
                }
            }
            drop(signal_entities);

            // Mirror to signal entities
            let mut signal_entities = self.signal_entities.write().unwrap();
            for prev_freq in &previously_playing_freqs {
                if (prev_freq - frequency_hz).abs() >= 1000.0 {
                    for signal in signal_entities.iter_mut() {
                        if (signal.frequency() - prev_freq).abs() < 1000.0 {
                            use crate::ecs::components::signal::PlaybackState as SignalPlaybackState;
                            signal
                                .playback
                                .transition_to(SignalPlaybackState::Completed);
                        }
                    }
                }
            }
        }
    }

    /// Sync entities to model (simulates what TUI does)
    pub fn sync(&mut self) {
        let mut context = SystemContext::new()
            .with_audio_entities(Arc::clone(&self.audio_entities))
            .with_signal_entities(Arc::clone(&self.signal_entities));

        if self.ui_update_system.run(&mut context).is_ok() {
            let signals_by_window = self.ui_update_system.signals_by_window().clone();
            self.sync_signals_to_model(&signals_by_window);
            self.update_window_completion(&signals_by_window);
        }
    }

    fn update_window_completion(
        &mut self,
        signals_by_window: &IndexMap<WindowId, Vec<crate::ecs::systems::ui::SignalData>>,
    ) {
        if let Some(max_window_id) = signals_by_window.keys().max_by_key(|w| w.window_index)
            && max_window_id.window_index > self.model.current_window
        {
            self.model.current_window = max_window_id.window_index;
            for (window_id, window) in self.model.windows.iter_mut() {
                if *window_id < self.model.current_window {
                    window.is_complete = true;
                }
            }
        }
    }

    fn sync_signals_to_model(
        &mut self,
        signals_by_window: &IndexMap<WindowId, Vec<crate::ecs::systems::ui::SignalData>>,
    ) {
        use std::time::Instant;

        use crate::ui::tui::model::{
            UiMode,
            types::{AnalysisStatus, PlaybackState, SignalProgress, WindowProgress},
        };

        let mut playing_signal_id: Option<String> = None;
        let is_browsing = matches!(
            self.model.ui_mode,
            UiMode::NavigatingScanner { .. } | UiMode::Listening { .. }
        );

        for (window_id, signal_data_list) in signals_by_window {
            let window_index = window_id.window_index;
            let window = self
                .model
                .windows
                .entry(window_index)
                .or_insert_with(|| WindowProgress {
                    window_id: window_index,
                    signals: Vec::new(),
                    is_complete: false,
                    signal_lookup: HashMap::new(),
                });

            for signal_data in signal_data_list {
                use crate::ecs::components::signal::PlaybackState as EcsPlaybackState;

                // Ignore updates to old windows unless:
                // - We're in browsing mode with Playing status
                // - The signal is transitioning to Completed
                let is_old_window = window_index < self.model.current_window;
                let is_playing_event = signal_data.playback_state == EcsPlaybackState::Playing;
                let is_completed_event = signal_data.playback_state == EcsPlaybackState::Completed;

                if is_old_window && !(is_browsing && is_playing_event) && !is_completed_event {
                    continue;
                }

                // Track playing signal
                if is_playing_event {
                    playing_signal_id = Some(signal_data.signal_id.clone());
                }

                // Map ECS AnalysisStatus to TUI AnalysisStatus (keeping analysis and playback
                // separate)
                use crate::ecs::components::AnalysisStatus as EcsStatus;
                let (status, audio_quality, signal_strength) = match signal_data.status {
                    EcsStatus::Detected => (AnalysisStatus::Detected, None, None),
                    EcsStatus::Analyzing => (AnalysisStatus::Analyzing, None, None),
                    EcsStatus::Signal { quality, strength } => {
                        (AnalysisStatus::Signal, Some(quality), Some(strength))
                    }
                    EcsStatus::Rejected { quality, strength } => {
                        (AnalysisStatus::Rejected, Some(quality), Some(strength))
                    }
                    EcsStatus::Error => (AnalysisStatus::Error, None, None),
                };

                let playback_state = match signal_data.playback_state {
                    EcsPlaybackState::NotPlaying => PlaybackState::NotPlaying,
                    EcsPlaybackState::Playing => PlaybackState::Playing,
                    EcsPlaybackState::Completed => PlaybackState::Completed,
                };

                let signal_progress = SignalProgress {
                    signal_id: signal_data.signal_id.clone(),
                    frequency_hz: signal_data.frequency_hz,
                    window_id: window_index,
                    center_frequency_hz: signal_data.frequency_hz,
                    completion: signal_data.completion,
                    status,
                    playback_state,
                    audio_quality,
                    signal_strength,
                    last_update: Instant::now(),
                };

                if let Some(&index) = window.signal_lookup.get(&signal_data.signal_id) {
                    window.signals[index] = signal_progress;
                } else {
                    let index = window.signals.len();
                    window
                        .signal_lookup
                        .insert(signal_data.signal_id.clone(), index);
                    window.signals.push(signal_progress);
                }
            }
        }

        if let Some(playing_id) = playing_signal_id {
            self.mark_old_playing_as_completed(&playing_id);
            self.transition_to_listening_if_needed(&playing_id);
        }
    }

    fn mark_old_playing_as_completed(&mut self, new_playing_id: &str) {
        use crate::ui::tui::model::types::PlaybackState;

        for window in self.model.windows.values_mut() {
            for signal in &mut window.signals {
                if signal.signal_id != new_playing_id
                    && signal.playback_state == PlaybackState::Playing
                {
                    signal.playback_state = PlaybackState::Completed;
                }
            }
        }
    }

    fn transition_to_listening_if_needed(&mut self, playing_signal_id: &str) {
        use crate::ui::tui::model::UiMode;

        match &self.model.ui_mode {
            UiMode::AwaitingTune {
                signal_index,
                window_id: _,
                tuning_signal_id: _,
            } => {
                for (window_id, window) in &self.model.windows {
                    for signal in &window.signals {
                        if signal.signal_id == playing_signal_id {
                            self.model.ui_mode = UiMode::Listening {
                                signal_index: *signal_index,
                                window_id: *window_id,
                                playing_signal_id: playing_signal_id.to_string(),
                            };
                            return;
                        }
                    }
                }
            }
            UiMode::Listening { signal_index, .. } => {
                for (window_id, window) in &self.model.windows {
                    for signal in &window.signals {
                        if signal.signal_id == playing_signal_id {
                            self.model.ui_mode = UiMode::Listening {
                                signal_index: *signal_index,
                                window_id: *window_id,
                                playing_signal_id: playing_signal_id.to_string(),
                            };
                            return;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Default for ModelTestContext {
    fn default() -> Self {
        Self::new()
    }
}
