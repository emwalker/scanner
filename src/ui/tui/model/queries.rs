//! Read-only query methods for TUI model

use super::{state::Model, types::*};

impl Model {
    /// Check if all windows are empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty() || self.windows.values().all(|w| w.signals.is_empty())
    }

    /// Check if all signals in all windows are complete (not checking if scan itself is done)
    pub fn all_signals_complete(&self) -> bool {
        !self.windows.is_empty()
            && self.windows.values().all(|window| {
                window.signals.iter().all(|signal| {
                    signal.completion >= 1.0
                        && (signal.playback_state == PlaybackState::Completed
                            || signal.status == AnalysisStatus::Rejected)
                })
            })
    }

    /// Check if scan is complete (all windows scanned AND all signals complete)
    pub fn all_complete(&self) -> bool {
        if let Some(total) = self.total_windows {
            self.current_window >= total && self.all_signals_complete()
        } else {
            false
        }
    }

    /// Get total signal count across all windows
    pub fn signal_count(&self) -> usize {
        self.windows.values().map(|w| w.signals.len()).sum()
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

    /// Computed property: selected_signal_index derived from UiMode
    /// Returns the navigation index (where arrow keys are positioned)
    pub fn selected_signal_index(&self) -> Option<usize> {
        match &self.ui_mode {
            UiMode::NavigatingScanner { signal_index, .. } => Some(*signal_index),
            UiMode::AwaitingTune { signal_index, .. } => Some(*signal_index),
            UiMode::Listening { signal_index, .. } => Some(*signal_index),
            UiMode::Idle => None,
        }
    }

    /// Get selectable signal progress items (excludes rejected)
    /// Used for selection and navigation
    fn selectable_signal_progress(&self) -> Vec<&SignalProgress> {
        let rows = self.build_signal_rows();

        let all_indices: Vec<_> = rows.iter().enumerate().map(|(idx, _)| idx).collect();

        let mut result = Vec::new();
        let mut global_idx = 0;

        for window in self.windows.values() {
            for signal in &window.signals {
                if all_indices.contains(&global_idx) {
                    result.push(signal);
                }
                global_idx += 1;
            }
        }

        result
    }

    /// Get the window_id, center frequency, and signal frequency for the currently selected
    /// signal
    pub fn selected_signal_info(&self) -> Option<SelectedSignalInfo> {
        let (signal_index, _window_id) = match &self.ui_mode {
            UiMode::NavigatingScanner {
                signal_index,
                window_id,
            } => (*signal_index, *window_id),
            UiMode::AwaitingTune {
                signal_index,
                window_id,
                ..
            } => (*signal_index, *window_id),
            UiMode::Listening {
                signal_index,
                window_id,
                ..
            } => (*signal_index, *window_id),
            UiMode::Idle => return None,
        };

        let rows = self.build_signal_rows();
        let row = rows.get(signal_index)?;

        // Look up full signal data from windows
        let window = self.windows.get(&row.window_id)?;
        let signal = window
            .signals
            .iter()
            .find(|c| c.window_id == row.window_id && c.frequency_hz == row.frequency_hz)?;

        Some(SelectedSignalInfo {
            signal_id: signal.signal_id.clone(),
            window_id: signal.window_id,
            center_frequency_hz: signal.center_frequency_hz,
            signal_frequency: signal.frequency_hz,
            signal_strength: signal.signal_strength,
            audio_quality: signal.audio_quality,
        })
    }

    /// Get count of selectable signals (excludes rejected)
    pub fn selectable_signal_count(&self) -> usize {
        self.selectable_signal_progress().len()
    }

    /// Get count of displayable signals (for layout calculations)
    /// Uses the unified filter to count visible rows
    pub fn displayable_signal_count(&self) -> usize {
        use crate::ui::tui::renderers::table_styles::VisibilityContext;

        let context = VisibilityContext::new(None, Some(self.current_window));
        self.count_visible_signals(&context)
    }

    /// Check if "Continue scan" option is currently selected
    pub fn is_continue_scan_selected(&self) -> bool {
        if !self.selection_mode() {
            return false;
        }

        let signal_count = self.selectable_signal_count();
        self.selected_signal_index() == Some(signal_count)
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
        let active_tuner = &self.active_tuner_id;

        self.tuners
            .iter()
            .map(|tuner| {
                let state = self.tuner_state(&tuner.id);
                let is_playing = active_tuner.as_ref() == Some(&tuner.id);

                super::TunerDisplayInfo {
                    id: tuner.id.clone(),
                    label: tuner.label.clone(),
                    state,
                    is_playing,
                }
            })
            .collect()
    }

    /// Build flattened list of signal rows for filtering and rendering
    /// This is the canonical data source - all renderers and navigation should use this
    /// Rows are in reverse chronological order (newest first) so older signals roll off the top
    pub fn build_signal_rows(&self) -> Vec<SignalRow> {
        let mut rows = Vec::new();

        for window in self.windows.values() {
            for signal in &window.signals {
                rows.push(SignalRow {
                    window_id: signal.window_id,
                    frequency_hz: signal.frequency_hz,
                    status: signal.status.clone(),
                    playback_state: signal.playback_state.clone(),
                    audio_quality: signal.audio_quality,
                    is_window_complete: window.is_complete,
                    completion: signal.completion,
                    notes: signal.notes.clone(),
                });
            }
        }

        rows.reverse();
        rows
    }

    /// Build flattened list of confirmed signal rows (only those sent into playback)
    /// Used specifically for the "Signals" table which only shows confirmed signals
    pub fn build_confirmed_signal_rows(&self) -> Vec<SignalRow> {
        let mut rows = Vec::new();

        for window in self.windows.values() {
            for signal in &window.signals {
                // Only include confirmed signals (those that would go into playback)
                if signal.status == AnalysisStatus::Signal {
                    rows.push(SignalRow {
                        window_id: signal.window_id,
                        frequency_hz: signal.frequency_hz,
                        status: signal.status.clone(),
                        playback_state: signal.playback_state.clone(),
                        audio_quality: signal.audio_quality,
                        is_window_complete: window.is_complete,
                        completion: signal.completion,
                        notes: signal.notes.clone(),
                    });
                }
            }
        }

        // Add persistent signals from storage (loaded on startup)
        // But first, collect them in a way that allows deduplication
        let mut persistent_rows = Vec::new();
        for persistent_signal in &self.persistent_signals {
            persistent_rows.push(SignalRow {
                window_id: 0, // Persistent signals don't belong to a scan window
                frequency_hz: persistent_signal.frequency_hz,
                status: AnalysisStatus::Signal, // Persistent signals are confirmed
                playback_state: PlaybackState::NotPlaying,
                audio_quality: None,      // Could be enhanced later
                is_window_complete: true, // Persistent signals are "complete"
                completion: 1.0,
                notes: persistent_signal.notes.clone(),
            });
        }

        // Deduplicate by frequency: prefer scan signals but merge persistent notes
        rows = self.merge_duplicate_signals(rows, persistent_rows);

        rows.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());
        rows
    }

    /// Merge duplicate signals by frequency, preferring scan signal state but preserving persistent
    /// notes Following elm-design principle: pure function that combines data from multiple
    /// sources
    fn merge_duplicate_signals(
        &self,
        scan_rows: Vec<SignalRow>,
        persistent_rows: Vec<SignalRow>,
    ) -> Vec<SignalRow> {
        use std::collections::HashMap;

        let mut merged: HashMap<u64, SignalRow> = HashMap::new();

        // First, add all scan signals (these take priority for state)
        for scan_row in scan_rows {
            let freq_key = scan_row.frequency_hz as u64;
            merged.insert(freq_key, scan_row);
        }

        // Then, merge in persistent signals
        for persistent_row in persistent_rows {
            let freq_key = persistent_row.frequency_hz as u64;

            if let Some(existing_scan_row) = merged.get_mut(&freq_key) {
                // Merge: prefer scan signal state, but add persistent notes if scan has none
                if existing_scan_row.notes.is_none() && persistent_row.notes.is_some() {
                    existing_scan_row.notes = persistent_row.notes;
                }
            } else {
                // No scan signal at this frequency, add the persistent signal
                merged.insert(freq_key, persistent_row);
            }
        }

        merged.into_values().collect()
    }

    /// Get count of confirmed signals (for Signals table layout calculation)
    /// Only counts signals with AnalysisStatus::Signal
    pub fn confirmed_signal_count(&self) -> usize {
        self.build_confirmed_signal_rows().len()
    }

    /// Count all signals (shows all signals for scan progress)
    pub fn count_visible_signals(
        &self,
        _context: &crate::ui::tui::renderers::table_styles::VisibilityContext<usize>,
    ) -> usize {
        self.build_signal_rows().len()
    }

    /// Get window_id from UiMode if in interactive mode
    pub fn selected_window(&self) -> Option<usize> {
        match &self.ui_mode {
            UiMode::NavigatingScanner { window_id, .. } => Some(*window_id),
            UiMode::AwaitingTune { window_id, .. } => Some(*window_id),
            UiMode::Listening { window_id, .. } => Some(*window_id),
            UiMode::Idle => None,
        }
    }

    /// Map from flat index to visible index (for updating FocusState)
    /// Since we now show all signals, visible index equals flat index
    pub fn flat_to_visible_index(&self, flat_index: usize) -> Option<usize> {
        let rows = self.build_signal_rows();
        if flat_index < rows.len() {
            Some(flat_index)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod selected_window_tests {
    use super::*;
    use crate::{ecs::SignalId, ui::tui::model::types::UiMode};

    #[test]
    fn test_selected_window_returns_none_when_idle() {
        let model = Model::new();
        assert_eq!(model.selected_window(), None);
    }

    #[test]
    fn test_selected_window_returns_window_id_when_navigating() {
        let mut model = Model::new();
        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 5,
            window_id: 2,
        };
        assert_eq!(model.selected_window(), Some(2));
    }

    #[test]
    fn test_selected_window_returns_window_id_when_awaiting_tune() {
        let mut model = Model::new();
        model.ui_mode = UiMode::AwaitingTune {
            signal_index: 3,
            window_id: 1,
            tuning_signal_id: SignalId::from_string("test".to_string()),
        };
        assert_eq!(model.selected_window(), Some(1));
    }

    #[test]
    fn test_selected_window_returns_window_id_when_listening() {
        let mut model = Model::new();
        model.ui_mode = UiMode::Listening {
            signal_index: 7,
            window_id: 3,
            playing_signal_id: SignalId::from_string("test".to_string()),
        };
        assert_eq!(model.selected_window(), Some(3));
    }
}

#[cfg(test)]
mod flat_to_visible_tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        ecs::SignalId,
        ui::tui::model::types::{AnalysisStatus, PlaybackState, SignalProgress, WindowProgress},
    };

    fn create_test_signal(
        window_id: usize,
        frequency: f64,
        status: AnalysisStatus,
    ) -> SignalProgress {
        SignalProgress {
            signal_id: SignalId::from_string(format!("cand_{}", frequency)),
            frequency_hz: frequency,
            window_id,
            center_frequency_hz: frequency,
            completion: 1.0,
            status,
            playback_state: PlaybackState::NotPlaying,
            audio_quality: None,
            signal_strength: None,
            last_update: Instant::now(),
            notes: None,
        }
    }

    #[test]
    fn test_flat_to_visible_index_when_all_visible() {
        let mut model = Model::new();
        model.current_window = 0;

        let mut window = WindowProgress {
            window_id: 0,
            signals: vec![
                create_test_signal(0, 88.1e6, AnalysisStatus::Signal),
                create_test_signal(0, 88.5e6, AnalysisStatus::Signal),
                create_test_signal(0, 88.9e6, AnalysisStatus::Signal),
            ],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        for (idx, cand) in window.signals.iter().enumerate() {
            window.signal_lookup.insert(cand.signal_id.clone(), idx);
        }

        model.windows.insert(0, window);

        assert_eq!(model.flat_to_visible_index(0), Some(0));
        assert_eq!(model.flat_to_visible_index(1), Some(1));
        assert_eq!(model.flat_to_visible_index(2), Some(2));
    }

    #[test]
    fn test_flat_to_visible_index_with_all_signals_shown() {
        let mut model = Model::new();
        model.current_window = 1;

        let mut window0 = WindowProgress {
            window_id: 0,
            signals: vec![
                create_test_signal(0, 88.1e6, AnalysisStatus::Signal),
                create_test_signal(0, 88.5e6, AnalysisStatus::Rejected),
            ],
            is_complete: true,
            signal_lookup: std::collections::HashMap::new(),
        };

        for (idx, cand) in window0.signals.iter().enumerate() {
            window0.signal_lookup.insert(cand.signal_id.clone(), idx);
        }

        model.windows.insert(0, window0);

        // Flat index 0 (Signal) -> visible index 0
        // Flat index 1 (Rejected in complete window) -> visible index 1 (now showing all
        // signals)
        assert_eq!(model.flat_to_visible_index(0), Some(0));
        assert_eq!(model.flat_to_visible_index(1), Some(1));
    }

    #[test]
    fn test_flat_to_visible_index_returns_none_for_out_of_bounds() {
        let model = Model::new();
        assert_eq!(model.flat_to_visible_index(99), None);
    }
}

#[cfg(test)]
mod selected_signal_info_tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        ecs::SignalId,
        ui::tui::model::types::{
            AnalysisStatus, PlaybackState, SignalProgress, UiMode, WindowProgress,
        },
    };

    #[test]
    fn test_selected_signal_info_returns_none_when_idle() {
        let model = Model::new();
        assert!(model.selected_signal_info().is_none());
    }

    #[test]
    fn test_selected_signal_info_uses_flat_index() {
        let mut model = Model::new();
        model.current_window = 0;

        let mut window = WindowProgress {
            window_id: 0,
            signals: vec![
                SignalProgress {
                    signal_id: SignalId::from_string("c1".to_string()),
                    frequency_hz: 88.1e6,
                    window_id: 0,
                    center_frequency_hz: 88.1e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: None,
                    signal_strength: Some(0.8),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: SignalId::from_string("c2".to_string()),
                    frequency_hz: 88.5e6,
                    window_id: 0,
                    center_frequency_hz: 88.5e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: None,
                    signal_strength: Some(0.9),
                    last_update: Instant::now(),
                    notes: None,
                },
            ],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        for (idx, cand) in window.signals.iter().enumerate() {
            window.signal_lookup.insert(cand.signal_id.clone(), idx);
        }

        model.windows.insert(0, window);
        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 1,
            window_id: 0,
        };

        let info = model.selected_signal_info().unwrap();
        // After reverse: [c2, c1]
        // Index 1 -> c1
        assert_eq!(info.signal_id, SignalId::from_string("c1".to_string()));
        assert_eq!(info.signal_frequency, 88.1e6);
        assert_eq!(info.signal_strength, Some(0.8));
    }
}
