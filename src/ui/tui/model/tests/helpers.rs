use crate::ecs::{
    AudioEntity, CandidateEntity, CandidateState, Entities, EntityWorld, StationEntity, System,
    SystemContext,
};
use crate::hardware::pool::{PoolStatus, TunerActivity, TunerId, TunerState, TunerStatus};
use crate::ui::tui::model::Model;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
    pub station_entities: Entities<StationEntity>,
    pub audio_entities: Entities<AudioEntity>,
    pub candidate_entities: Entities<CandidateEntity>,
    ui_update_system: crate::ecs::systems::UIUpdateSystem,
}

impl ModelTestContext {
    pub fn new() -> Self {
        Self {
            model: Model::new(),
            station_entities: Arc::new(RwLock::new(EntityWorld::new())),
            audio_entities: Arc::new(RwLock::new(EntityWorld::new())),
            candidate_entities: Arc::new(RwLock::new(EntityWorld::new())),
            ui_update_system: crate::ecs::systems::UIUpdateSystem::new(),
        }
    }

    /// Create or update a candidate entity with the given state
    pub fn update_candidate(
        &mut self,
        frequency_hz: f64,
        window_id: usize,
        state: CandidateState,
        audio_quality: Option<crate::audio::quality::AudioQuality>,
        signal_strength: Option<f64>,
    ) {
        use crate::ecs::CandidateId;
        use crate::scanning::window::WindowMetadata;

        let metadata = WindowMetadata {
            window_id,
            center_frequency_hz: frequency_hz,
        };

        let id = CandidateId::new(frequency_hz, window_id);
        let mut entities = self.candidate_entities.write().unwrap();

        if let Some(entity) = entities.get_mut(&id) {
            entity.lifecycle.transition_to(state);
            if let Some(quality) = audio_quality {
                entity.info.set_audio_quality(quality);
            }
            if let Some(strength) = signal_strength {
                entity.info.set_signal_strength(strength);
            }
        } else {
            let mut entity = CandidateEntity::new(frequency_hz, metadata);
            entity.lifecycle.transition_to(state);
            if let Some(quality) = audio_quality {
                entity.info.set_audio_quality(quality);
            }
            if let Some(strength) = signal_strength {
                entity.info.set_signal_strength(strength);
            }
            entities.insert(entity);
        }
    }

    /// Sync entities to model (simulates what TUI does)
    pub fn sync(&mut self) {
        let mut context = SystemContext::new()
            .with_station_entities(Arc::clone(&self.station_entities))
            .with_audio_entities(Arc::clone(&self.audio_entities))
            .with_candidate_entities(Arc::clone(&self.candidate_entities));

        if self.ui_update_system.run(&mut context).is_ok() {
            let candidates_by_window = self.ui_update_system.candidates_by_window().clone();
            self.sync_candidates_to_model(&candidates_by_window);
            self.update_window_completion(&candidates_by_window);
        }
    }

    fn update_window_completion(
        &mut self,
        candidates_by_window: &IndexMap<usize, Vec<crate::ecs::systems::ui::CandidateData>>,
    ) {
        if let Some(max_window) = candidates_by_window.keys().max()
            && *max_window > self.model.current_window
        {
            self.model.current_window = *max_window;
            for (window_id, window) in self.model.windows.iter_mut() {
                if *window_id < self.model.current_window {
                    window.is_complete = true;
                }
            }
        }
    }

    fn sync_candidates_to_model(
        &mut self,
        candidates_by_window: &IndexMap<usize, Vec<crate::ecs::systems::ui::CandidateData>>,
    ) {
        use crate::ui::tui::model::UiMode;
        use crate::ui::tui::model::types::{CandidateProgress, CandidateStatus, WindowProgress};
        use std::time::Instant;

        let mut playing_candidate_id: Option<String> = None;
        let is_browsing = matches!(
            self.model.ui_mode,
            UiMode::NavigatingScanner { .. } | UiMode::Listening { .. }
        );

        for (window_id, candidate_data_list) in candidates_by_window {
            let window = self
                .model
                .windows
                .entry(*window_id)
                .or_insert_with(|| WindowProgress {
                    window_id: *window_id,
                    candidates: Vec::new(),
                    is_complete: false,
                    candidate_lookup: HashMap::new(),
                });

            for candidate_data in candidate_data_list {
                // Ignore updates to old windows unless:
                // - We're in browsing mode with Playing status
                // - The candidate is transitioning to Completed
                let is_old_window = *window_id < self.model.current_window;
                let is_playing_event = matches!(candidate_data.state, CandidateState::Playing);
                let is_completed_event = matches!(candidate_data.state, CandidateState::Completed);

                if is_old_window && !(is_browsing && is_playing_event) && !is_completed_event {
                    continue;
                }

                let status = match candidate_data.state {
                    CandidateState::Detected => CandidateStatus::Detected,
                    CandidateState::Analyzing => CandidateStatus::Analyzing,
                    CandidateState::Signal => CandidateStatus::Signal,
                    CandidateState::Playing => {
                        playing_candidate_id = Some(candidate_data.candidate_id.clone());
                        CandidateStatus::Playing
                    }
                    CandidateState::Rejected => CandidateStatus::Rejected,
                    CandidateState::Completed => CandidateStatus::Completed,
                };

                let candidate_progress = CandidateProgress {
                    candidate_id: candidate_data.candidate_id.clone(),
                    frequency_hz: candidate_data.frequency_hz,
                    metadata: crate::scanning::window::WindowMetadata {
                        window_id: *window_id,
                        center_frequency_hz: candidate_data.frequency_hz,
                    },
                    completion: candidate_data.completion,
                    status,
                    audio_quality: candidate_data.audio_quality,
                    signal_strength: candidate_data.signal_strength,
                    last_update: Instant::now(),
                };

                if let Some(&index) = window.candidate_lookup.get(&candidate_data.candidate_id) {
                    window.candidates[index] = candidate_progress;
                } else {
                    let index = window.candidates.len();
                    window
                        .candidate_lookup
                        .insert(candidate_data.candidate_id.clone(), index);
                    window.candidates.push(candidate_progress);
                }
            }
        }

        if let Some(playing_id) = playing_candidate_id {
            self.mark_old_playing_as_completed(&playing_id);
            self.transition_to_listening_if_needed(&playing_id);
        }
    }

    fn mark_old_playing_as_completed(&mut self, new_playing_id: &str) {
        use crate::ui::tui::model::types::CandidateStatus;

        for window in self.model.windows.values_mut() {
            for candidate in &mut window.candidates {
                if candidate.candidate_id != new_playing_id
                    && candidate.status == CandidateStatus::Playing
                {
                    candidate.status = CandidateStatus::Completed;
                }
            }
        }
    }

    fn transition_to_listening_if_needed(&mut self, playing_candidate_id: &str) {
        use crate::ui::tui::model::UiMode;

        match &self.model.ui_mode {
            UiMode::AwaitingTune {
                navigation_index,
                tuning_index: _,
            } => {
                let all_candidates = self.model.displayable_candidates();
                if let Some((playing_index, _)) = all_candidates
                    .iter()
                    .enumerate()
                    .find(|(_, (_, c))| c.candidate_id == playing_candidate_id)
                {
                    self.model.ui_mode = UiMode::Listening {
                        navigation_index: *navigation_index,
                        playing_index,
                        playing_candidate_id: playing_candidate_id.to_string(),
                    };
                }
            }
            UiMode::Listening {
                navigation_index, ..
            } => {
                let all_candidates = self.model.displayable_candidates();
                if let Some((playing_index, _)) = all_candidates
                    .iter()
                    .enumerate()
                    .find(|(_, (_, c))| c.candidate_id == playing_candidate_id)
                {
                    self.model.ui_mode = UiMode::Listening {
                        navigation_index: *navigation_index,
                        playing_index,
                        playing_candidate_id: playing_candidate_id.to_string(),
                    };
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
