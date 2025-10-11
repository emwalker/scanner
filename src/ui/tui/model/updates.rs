//! Event processing and state update methods

use crate::ui::{ProgressEvent, ProgressEventType, TuiEvent};
use std::{collections::HashMap, time::Instant};
use tracing::debug;

use super::{
    state::Model,
    types::{CandidateProgress, CandidateStatus, TunerState, UiMode, WindowProgress},
};

impl Model {
    /// Update the model based on a TUI event (progress or discovery)
    pub fn update_tui_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::Progress(progress_event) => self.update(progress_event),
            TuiEvent::TunerAdded(tuner) => self.add_device(tuner),
            TuiEvent::TunerRemoved(tuner_id) => self.remove_device(&tuner_id),
            TuiEvent::Paused { tuner_id } => {
                debug!(tuner_id = ?tuner_id, "Scanning paused, tuner now available");
                self.tuner_states.insert(tuner_id, TunerState::Available);
            }
            TuiEvent::ActiveTunersUpdated { status } => {
                debug!(
                    total_tuners = status.tuners.len(),
                    available_count = status
                        .tuners
                        .iter()
                        .filter(|t| t.state == crate::hardware::pool::TunerState::Available)
                        .count(),
                    allocated_count = status
                        .tuners
                        .iter()
                        .filter(|t| t.state == crate::hardware::pool::TunerState::Allocated)
                        .count(),
                    scanning_count = status
                        .tuners
                        .iter()
                        .filter(
                            |t| t.activity == Some(crate::hardware::pool::TunerActivity::Scanning)
                        )
                        .count(),
                    listening_count = status
                        .tuners
                        .iter()
                        .filter(
                            |t| t.activity == Some(crate::hardware::pool::TunerActivity::Listening)
                        )
                        .count(),
                    "Pool status updated"
                );

                // Debug each tuner's state
                for tuner in &status.tuners {
                    debug!(
                        device_id = ?tuner.id.device_id,
                        state = ?tuner.state,
                        activity = ?tuner.activity,
                        "Tuner status"
                    );
                }

                // Sync tuner list with pool status to ensure device IDs match
                // The pool may have different device IDs than discovery service
                for pool_tuner in &status.tuners {
                    let device_id = pool_tuner.id.device_id.clone();

                    // Add tuner if not already in list
                    if !self.tuners.iter().any(|t| t.id == device_id) {
                        debug!(
                            device_id = ?device_id,
                            model = %pool_tuner.model,
                            backend = %pool_tuner.backend,
                            "Adding tuner from pool status"
                        );
                        self.tuners.push(crate::hardware::DeviceInfo {
                            id: device_id.clone(),
                            label: format!("{} ({})", pool_tuner.model, pool_tuner.backend),
                        });
                    }
                }

                self.pool_status = Some(status);
            }
        }
    }

    /// Update the model based on a progress event
    pub fn update(&mut self, event: ProgressEvent) {
        if !self.should_process_event(&event) {
            return;
        }

        self.update_current_window(&event);

        if let Some(candidate_id) = event.candidate_id.clone() {
            self.update_candidate(event, &candidate_id);
        }

        self.complete_window_if_done();
    }

    fn should_process_event(&self, event: &ProgressEvent) -> bool {
        if self.is_interactive()
            && event.event_type != ProgressEventType::AudioPlaybackStarted
            && event.event_type != ProgressEventType::AudioPlaybackCompleted
        {
            return false;
        }

        !matches!(event.event_type, ProgressEventType::PeakDetected)
    }

    fn update_current_window(&mut self, event: &ProgressEvent) {
        if event.metadata.window_id > self.current_window {
            self.current_window = event.metadata.window_id;
            for (window_id, window) in self.windows.iter_mut() {
                if *window_id < self.current_window {
                    window.is_complete = true;
                }
            }
        }
    }

    fn update_candidate(&mut self, event: ProgressEvent, candidate_id: &str) {
        debug!(
            event_type = ?event.event_type,
            candidate_id = ?candidate_id,
            window_id = event.metadata.window_id,
            current_window = self.current_window,
            ui_mode = ?self.ui_mode,
            "Processing event with candidate_id"
        );

        if event.metadata.window_id < self.current_window
            && !(self.is_interactive()
                && (event.event_type == ProgressEventType::AudioPlaybackStarted
                    || event.event_type == ProgressEventType::AudioPlaybackCompleted))
        {
            debug!("Ignoring event for old window");
            return;
        }

        if event.event_type == ProgressEventType::AudioPlaybackStarted {
            self.clear_playing_candidates(candidate_id);
        }

        let window_id = event.metadata.window_id;
        let window = self.or_create_window(window_id);

        let candidate_index = if let Some(&index) = window.candidate_lookup.get(candidate_id) {
            debug!(index = index, "Found existing candidate");
            index
        } else {
            debug!("Creating new candidate");
            let new_candidate = CandidateProgress {
                candidate_id: candidate_id.to_string(),
                frequency_hz: event.frequency_hz,
                metadata: event.metadata,
                completion: 0.0,
                status: CandidateStatus::Detected,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
            };
            let index = window.candidates.len();
            window.candidates.push(new_candidate);
            window
                .candidate_lookup
                .insert(candidate_id.to_string(), index);
            index
        };

        {
            let candidate = &mut window.candidates[candidate_index];

            match event.event_type {
                ProgressEventType::CandidateCreated => {
                    candidate.status = CandidateStatus::Detected;
                    candidate.completion = 0.3;
                }
                ProgressEventType::AudioAnalysisStarted => {
                    candidate.status = CandidateStatus::Analyzing;
                    candidate.completion = 0.5;
                }
                ProgressEventType::AudioAnalysisCompleted => {
                    if candidate.status == CandidateStatus::Signal {
                    } else if candidate.status != CandidateStatus::Rejected {
                        candidate.status = CandidateStatus::Signal;
                        candidate.completion = 0.6;
                    } else {
                        candidate.completion = 1.0;
                    }
                }
                ProgressEventType::CandidateRejected => {
                    candidate.status = CandidateStatus::Rejected;
                    candidate.completion = 1.0;
                }
                ProgressEventType::SignalGenerated => {
                    candidate.status = CandidateStatus::Signal;
                    candidate.completion = 0.6;
                    if let Some(quality) = event.audio_quality {
                        candidate.audio_quality = Some(quality);
                    }
                    if let Some(strength) = event.signal_strength {
                        candidate.signal_strength = Some(strength);
                    }
                }
                ProgressEventType::AudioPlaybackStarted => {
                    debug!(
                        frequency_mhz = event.frequency_hz / 1e6,
                        candidate_id = ?candidate_id,
                        "Setting candidate to Playing status"
                    );
                    candidate.status = CandidateStatus::Playing;
                    candidate.completion = 0.8;
                }
                ProgressEventType::AudioPlaybackCompleted => {
                    candidate.status = CandidateStatus::Completed;
                    candidate.completion = 1.0;
                }
                ProgressEventType::ThreadCompleted | ProgressEventType::PeakDetected => {}
            }

            if let Some(quality) = event.audio_quality {
                candidate.audio_quality = Some(quality);
            }
            candidate.last_update = Instant::now();
        }

        if event.event_type == ProgressEventType::AudioPlaybackStarted {
            match &self.ui_mode {
                UiMode::AwaitingTune {
                    navigation_index,
                    tuning_index,
                }
                | UiMode::Listening {
                    navigation_index,
                    playing_index: tuning_index,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: *navigation_index,
                        playing_index: *tuning_index,
                        playing_candidate_id: candidate_id.to_string(),
                    };
                }
                _ => {}
            }
        }
    }

    fn clear_playing_candidates(&mut self, new_playing_id: &str) {
        debug!(
            new_playing_candidate = ?new_playing_id,
            "Clearing all other Playing candidates before setting new one"
        );
        for window in self.windows.values_mut() {
            for candidate in &mut window.candidates {
                if candidate.status == CandidateStatus::Playing {
                    debug!(
                        cleared_candidate = ?candidate.candidate_id,
                        "Clearing Playing status from candidate"
                    );
                    candidate.status = CandidateStatus::Completed;
                    candidate.completion = 1.0;
                }
            }
        }
    }

    fn or_create_window(&mut self, window_id: usize) -> &mut WindowProgress {
        self.windows
            .entry(window_id)
            .or_insert_with(|| WindowProgress {
                window_id,
                candidates: Vec::new(),
                is_complete: false,
                candidate_lookup: HashMap::new(),
            })
    }

    fn complete_window_if_done(&mut self) {
        if self.all_complete()
            && let Some(window) = self.windows.get_mut(&self.current_window)
        {
            window.is_complete = true;
        }
    }
}
