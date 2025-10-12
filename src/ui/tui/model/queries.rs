//! Read-only query methods for TUI model

use super::{state::Model, types::*};

impl Model {
    /// Check if all windows are empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty() || self.windows.values().all(|w| w.candidates.is_empty())
    }

    /// Check if all candidates in all windows are complete (not checking if scan itself is done)
    pub fn all_candidates_complete(&self) -> bool {
        !self.windows.is_empty()
            && self.windows.values().all(|window| {
                window.candidates.iter().all(|candidate| {
                    candidate.completion >= 1.0
                        && (candidate.status == CandidateStatus::Completed
                            || candidate.status == CandidateStatus::Rejected)
                })
            })
    }

    /// Check if scan is complete (all windows scanned AND all candidates complete)
    pub fn all_complete(&self) -> bool {
        if let Some(total) = self.total_windows {
            self.current_window >= total && self.all_candidates_complete()
        } else {
            false
        }
    }

    /// Get total candidate count across all windows
    pub fn candidate_count(&self) -> usize {
        self.windows.values().map(|w| w.candidates.len()).sum()
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.ui_mode, UiMode::Idle)
    }

    pub fn is_navigating(&self) -> bool {
        matches!(self.ui_mode, UiMode::NavigatingScanner { .. })
    }

    pub fn is_awaiting_tune(&self) -> bool {
        matches!(self.ui_mode, UiMode::AwaitingTune { .. })
    }

    pub fn is_listening(&self) -> bool {
        matches!(self.ui_mode, UiMode::Listening { .. })
    }

    pub fn is_interactive(&self) -> bool {
        !matches!(self.ui_mode, UiMode::Idle)
    }

    /// Computed property: selection_mode derived from UiMode
    /// Returns true when user can navigate with arrow keys
    pub fn selection_mode(&self) -> bool {
        self.is_interactive()
    }

    /// Computed property: browsing_mode derived from UiMode
    /// Returns true when scan is paused for manual browsing/listening
    pub fn browsing_mode(&self) -> bool {
        matches!(
            self.ui_mode,
            UiMode::AwaitingTune { .. } | UiMode::Listening { .. }
        )
    }

    /// Computed property: selected_candidate_index derived from UiMode
    /// Returns the navigation index (where arrow keys are positioned)
    pub fn selected_candidate_index(&self) -> Option<usize> {
        match &self.ui_mode {
            UiMode::NavigatingScanner { selected_index } => Some(*selected_index),
            UiMode::AwaitingTune {
                navigation_index, ..
            } => Some(*navigation_index),
            UiMode::Listening {
                navigation_index, ..
            } => Some(*navigation_index),
            UiMode::Idle => None,
        }
    }

    /// Get ordered list of displayable windows (oldest to newest)
    pub fn displayable_windows(&self) -> Vec<(&usize, &WindowProgress)> {
        self.windows
            .iter()
            .filter(|(_, window)| window.should_display())
            .collect()
    }

    /// Get count of displayable windows
    pub fn displayable_window_count(&self) -> usize {
        self.windows
            .values()
            .filter(|window| window.should_display())
            .count()
    }

    /// Get flattened list of displayable candidates across all windows
    /// This includes rejected candidates for display purposes (during scanning)
    /// In selection mode, rejected candidates are filtered out
    pub fn displayable_candidates(&self) -> Vec<(usize, &CandidateProgress)> {
        let mut candidates = Vec::new();
        for (window_id, window) in self.displayable_windows() {
            let is_current = *window_id == self.current_window;
            for candidate in window.displayable_candidates(is_current, self.selection_mode()) {
                candidates.push((*window_id, candidate));
            }
        }
        candidates
    }

    /// Get flattened list of selectable candidates across all windows
    /// Filters out rejected candidates - users should not be able to select rejected stations
    pub fn selectable_candidates(&self) -> Vec<(usize, &CandidateProgress)> {
        let mut candidates = Vec::new();
        for (window_id, window) in self.displayable_windows() {
            let is_current = *window_id == self.current_window;
            for candidate in window.displayable_candidates(is_current, self.selection_mode()) {
                if candidate.status != CandidateStatus::Rejected {
                    candidates.push((*window_id, candidate));
                }
            }
        }
        candidates
    }

    /// Get count of displayable candidates (includes rejected for display)
    pub fn displayable_candidate_count(&self) -> usize {
        self.displayable_candidates().len()
    }

    /// Get count of selectable candidates (excludes rejected)
    pub fn selectable_candidate_count(&self) -> usize {
        self.selectable_candidates().len()
    }

    /// Get the window_id, center frequency, and candidate frequency for the currently selected candidate
    pub fn selected_candidate_info(&self) -> Option<SelectedCandidateInfo> {
        if !self.selection_mode() {
            return None;
        }

        let selected_idx = self.selected_candidate_index()?;
        let candidates = self.selectable_candidates();

        if selected_idx >= candidates.len() {
            return None;
        }

        let (_window_id, candidate) = candidates[selected_idx];

        Some(SelectedCandidateInfo {
            candidate_id: candidate.candidate_id.clone(),
            metadata: candidate.metadata,
            candidate_frequency: candidate.frequency_hz,
            signal_strength: candidate.signal_strength,
            audio_quality: candidate.audio_quality,
        })
    }

    /// Check if "Continue scan" option is currently selected
    pub fn is_continue_scan_selected(&self) -> bool {
        if !self.selection_mode() {
            return false;
        }

        let candidate_count = self.selectable_candidate_count();
        self.selected_candidate_index() == Some(candidate_count)
    }

    /// Get display states for all tuners
    /// This view model function makes the state-to-label mapping explicit and testable
    pub fn tuner_display_states(&self) -> Vec<super::TunerDisplayState> {
        self.tuners
            .iter()
            .map(|tuner| {
                let state = self.tuner_state(&tuner.id);
                super::TunerDisplayState {
                    tuner_id: tuner.id.clone(),
                    label: tuner.label.clone(),
                    status_label: state.display(),
                }
            })
            .collect()
    }

    /// Get flat list of tuners with their current states
    /// Tuners are naturally sorted by label via BTreeSet
    pub fn tuner_list(&self) -> Vec<super::TunerDisplayInfo> {
        self.tuners
            .iter()
            .map(|tuner| {
                let state = self.tuner_state(&tuner.id);
                super::TunerDisplayInfo {
                    id: tuner.id.clone(),
                    label: tuner.label.clone(),
                    state,
                }
            })
            .collect()
    }
}
