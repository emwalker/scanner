//! Navigation and state-modifying methods for TUI model

use tracing::debug;

use super::{state::Model, types::*};

impl Model {
    /// Request to quit the application
    pub fn quit(&mut self) {
        self.should_quit = true;
        self.mark_dirty();
    }

    pub fn toggle_theme_selector(&mut self) {
        self.theme_selector_open = !self.theme_selector_open;
        self.mark_dirty();
    }

    pub fn close_theme_selector(&mut self) {
        self.theme_selector_open = false;
        self.mark_dirty();
    }

    pub fn theme_selector_next(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + 1) % theme_count;
            self.mark_dirty();
        }
    }

    pub fn theme_selector_prev(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + theme_count - 1) % theme_count;
            self.mark_dirty();
        }
    }

    /// Enter selection mode - pauses scanning and allows browsing signals
    pub fn enter_selection_mode(&mut self) {
        let rows = self.build_signal_rows();
        if !rows.is_empty() {
            let last_index = rows.len() - 1;
            let window_id = rows[last_index].window_id;
            self.ui_mode = UiMode::NavigatingScanner {
                signal_index: last_index,
                window_id,
            };
            self.mark_dirty();
        }
    }

    /// Exit selection mode - returns to normal scanning
    pub fn exit_selection_mode(&mut self) {
        self.ui_mode = UiMode::Idle;
        self.mark_dirty();
    }

    /// Select next signal (moving forward in time)
    pub fn select_next_signal(&mut self) {
        self.select_next_signal_with_viewport(20);
    }

    /// Select next signal with viewport height for scroll adjustment
    pub fn select_next_signal_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let rows = self.build_signal_rows();
        if rows.is_empty() {
            return;
        }

        let current = match &self.ui_mode {
            UiMode::NavigatingScanner { signal_index, .. } => *signal_index,
            UiMode::AwaitingTune { signal_index, .. } => *signal_index,
            UiMode::Listening { signal_index, .. } => *signal_index,
            UiMode::Idle => return,
        };

        let next = (current + 1).min(rows.len() - 1);

        if next != current {
            let new_window_id = rows[next].window_id;

            self.ui_mode = match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => UiMode::NavigatingScanner {
                    signal_index: next,
                    window_id: new_window_id,
                },
                UiMode::AwaitingTune {
                    tuning_signal_id, ..
                } => UiMode::AwaitingTune {
                    signal_index: next,
                    window_id: new_window_id,
                    tuning_signal_id: tuning_signal_id.clone(),
                },
                UiMode::Listening {
                    playing_signal_id, ..
                } => UiMode::Listening {
                    signal_index: next,
                    window_id: new_window_id,
                    playing_signal_id: playing_signal_id.clone(),
                },
                UiMode::Idle => return,
            };

            // Update FocusState to match the visible row
            if let Some(visible_idx) = self.flat_to_visible_index(next) {
                self.focus_state = FocusState::ScanProgress(visible_idx);
            }

            self.adjust_scroll_to_selection(viewport_height);
            self.mark_dirty();
        }
    }

    /// Select previous signal (moving backward in time)
    pub fn select_previous_signal(&mut self) {
        self.select_previous_signal_with_viewport(20);
    }

    /// Select previous signal with viewport height for scroll adjustment
    pub fn select_previous_signal_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let rows = self.build_signal_rows();

        let current = match &self.ui_mode {
            UiMode::NavigatingScanner { signal_index, .. } => *signal_index,
            UiMode::AwaitingTune { signal_index, .. } => *signal_index,
            UiMode::Listening { signal_index, .. } => *signal_index,
            UiMode::Idle => return,
        };

        if current > 0 {
            let prev = current - 1;
            let new_window_id = rows[prev].window_id;

            self.ui_mode = match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => UiMode::NavigatingScanner {
                    signal_index: prev,
                    window_id: new_window_id,
                },
                UiMode::AwaitingTune {
                    tuning_signal_id, ..
                } => UiMode::AwaitingTune {
                    signal_index: prev,
                    window_id: new_window_id,
                    tuning_signal_id: tuning_signal_id.clone(),
                },
                UiMode::Listening {
                    playing_signal_id, ..
                } => UiMode::Listening {
                    signal_index: prev,
                    window_id: new_window_id,
                    playing_signal_id: playing_signal_id.clone(),
                },
                UiMode::Idle => return,
            };

            // Update FocusState to match the visible row
            if let Some(visible_idx) = self.flat_to_visible_index(prev) {
                self.focus_state = FocusState::ScanProgress(visible_idx);
            }

            self.adjust_scroll_to_selection(viewport_height);
            self.mark_dirty();
        }
    }

    /// Adjust scroll offset to ensure the selected signal is visible
    pub fn adjust_scroll_to_selection(&mut self, viewport_height: usize) {
        if let Some(selected_idx) = self.selected_signal_index() {
            if selected_idx < self.scroll_offset {
                self.scroll_offset = selected_idx;
                self.mark_dirty();
            } else if selected_idx >= self.scroll_offset + viewport_height {
                self.scroll_offset = selected_idx.saturating_sub(viewport_height.saturating_sub(1));
                self.mark_dirty();
            }
        }
    }

    /// Navigate to next table (Tab key)
    pub fn navigate_next_table(&mut self, tuner_count: usize) {
        let entering_scan_progress = match self.focus_state {
            FocusState::ScanProgress(_) => false,
            FocusState::Activities(_)
            | FocusState::TunersTable(_)
            | FocusState::SignalsTable(_) => true,
            FocusState::SignalDetailModal => false,
        };

        self.focus_state = match self.focus_state {
            FocusState::Activities(_) => {
                if tuner_count > 0 {
                    FocusState::TunersTable(0)
                } else {
                    FocusState::SignalsTable(0)
                }
            }
            FocusState::TunersTable(_) => FocusState::SignalsTable(0),
            FocusState::SignalsTable(_) => FocusState::ScanProgress(0),
            FocusState::ScanProgress(_) => FocusState::Activities(0),
            FocusState::SignalDetailModal => FocusState::SignalsTable(0),
        };

        // When entering ScanProgress table, transition to NavigatingScanner mode
        if entering_scan_progress && matches!(self.focus_state, FocusState::ScanProgress(_)) {
            let rows = self.build_signal_rows();
            if !rows.is_empty() {
                let window_id = rows[0].window_id;
                debug!("Tab entering ScanProgress - setting NavigatingScanner mode");
                self.ui_mode = UiMode::NavigatingScanner {
                    signal_index: 0,
                    window_id,
                };
            } else {
                debug!("Tab entering ScanProgress but no rows available");
            }
        }

        self.mark_dirty();
    }

    /// Navigate to previous table (Shift-Tab key)
    pub fn navigate_previous_table(&mut self, tuner_count: usize) {
        let entering_scan_progress = match self.focus_state {
            FocusState::Activities(_) => true,
            FocusState::TunersTable(_) => false,
            FocusState::SignalsTable(_) => false,
            FocusState::ScanProgress(_) => false,
            FocusState::SignalDetailModal => false,
        };

        self.focus_state = match self.focus_state {
            FocusState::Activities(_) => FocusState::ScanProgress(0),
            FocusState::TunersTable(_) => FocusState::Activities(0),
            FocusState::SignalsTable(_) => {
                if tuner_count > 0 {
                    FocusState::TunersTable(0)
                } else {
                    FocusState::Activities(0)
                }
            }
            FocusState::ScanProgress(_) => FocusState::SignalsTable(0),
            FocusState::SignalDetailModal => FocusState::SignalsTable(0),
        };

        // When entering ScanProgress table, transition to NavigatingScanner mode
        if entering_scan_progress && matches!(self.focus_state, FocusState::ScanProgress(_)) {
            let rows = self.build_signal_rows();
            if !rows.is_empty() {
                let window_id = rows[0].window_id;
                debug!("Shift-Tab entering ScanProgress - setting NavigatingScanner mode");
                self.ui_mode = UiMode::NavigatingScanner {
                    signal_index: 0,
                    window_id,
                };
            } else {
                debug!("Shift-Tab entering ScanProgress but no rows available");
            }
        }

        self.mark_dirty();
    }

    /// Handle up arrow - move up within table with wrap-around
    pub fn navigate_up(&mut self) {
        match self.focus_state {
            FocusState::Activities(idx) => {
                let row_count = self.activities_row_count();
                if row_count > 0 {
                    let new_idx = if idx == 0 { row_count - 1 } else { idx - 1 };
                    self.focus_state = FocusState::Activities(new_idx);

                    // Update displayed_task_id when selection changes
                    if let Some(task) = self.tasks.get(new_idx) {
                        self.displayed_task_id = Some(task.task_id.clone());
                    }

                    self.mark_dirty();
                }
            }
            FocusState::TunersTable(idx) => {
                let row_count = self.tuners_row_count();
                if row_count > 0 {
                    self.focus_state =
                        FocusState::TunersTable(if idx == 0 { row_count - 1 } else { idx - 1 });
                    self.mark_dirty();
                }
            }
            FocusState::SignalsTable(idx) => {
                let row_count = self.signals_table_row_count();
                if row_count > 0 {
                    self.focus_state =
                        FocusState::SignalsTable(if idx == 0 { row_count - 1 } else { idx - 1 });
                    self.mark_dirty();
                }
            }
            FocusState::ScanProgress(idx) => {
                let row_count = self.scan_signals_row_count();
                if row_count > 0 {
                    let new_idx = if idx == 0 { row_count - 1 } else { idx - 1 };
                    self.focus_state = FocusState::ScanProgress(new_idx);

                    // Keep ui_mode in sync when navigating signals
                    if matches!(self.ui_mode, UiMode::NavigatingScanner { .. }) {
                        let rows = self.build_signal_rows();
                        if let Some(row) = rows.get(new_idx) {
                            self.ui_mode = UiMode::NavigatingScanner {
                                signal_index: new_idx,
                                window_id: row.window_id,
                            };
                        }
                    }

                    self.mark_dirty();
                }
            }
            FocusState::SignalDetailModal => {
                // Modal doesn't support navigation
            }
        }
    }

    /// Handle down arrow - move down within table with wrap-around
    pub fn navigate_down(&mut self) {
        match self.focus_state {
            FocusState::Activities(idx) => {
                let row_count = self.activities_row_count();
                if row_count > 0 {
                    let new_idx = (idx + 1) % row_count;
                    self.focus_state = FocusState::Activities(new_idx);

                    // Update displayed_task_id when selection changes
                    if let Some(task) = self.tasks.get(new_idx) {
                        self.displayed_task_id = Some(task.task_id.clone());
                    }

                    self.mark_dirty();
                }
            }
            FocusState::TunersTable(idx) => {
                let row_count = self.tuners_row_count();
                if row_count > 0 {
                    self.focus_state = FocusState::TunersTable((idx + 1) % row_count);
                    self.mark_dirty();
                }
            }
            FocusState::SignalsTable(idx) => {
                let row_count = self.signals_table_row_count();
                if row_count > 0 {
                    self.focus_state = FocusState::SignalsTable((idx + 1) % row_count);
                    self.mark_dirty();
                }
            }
            FocusState::ScanProgress(idx) => {
                let row_count = self.scan_signals_row_count();
                if row_count > 0 {
                    let new_idx = (idx + 1) % row_count;
                    self.focus_state = FocusState::ScanProgress(new_idx);

                    // Keep ui_mode in sync when navigating signals
                    if matches!(self.ui_mode, UiMode::NavigatingScanner { .. }) {
                        let rows = self.build_signal_rows();
                        if let Some(row) = rows.get(new_idx) {
                            self.ui_mode = UiMode::NavigatingScanner {
                                signal_index: new_idx,
                                window_id: row.window_id,
                            };
                        }
                    }

                    self.mark_dirty();
                }
            }
            FocusState::SignalDetailModal => {
                // Modal doesn't support navigation
            }
        }
    }

    /// Handle left arrow - move to previous table
    pub fn navigate_left(&mut self, tuner_count: usize) {
        match self.focus_state {
            FocusState::Activities(_) => {}
            FocusState::TunersTable(_) => {
                self.focus_state = FocusState::Activities(0);
                self.mark_dirty();
            }
            FocusState::SignalsTable(_) => {
                if tuner_count > 0 {
                    self.focus_state = FocusState::TunersTable(0);
                } else {
                    self.focus_state = FocusState::Activities(0);
                }
                self.mark_dirty();
            }
            FocusState::ScanProgress(_) => {
                // Exiting ScanProgress, but keep NavigatingScanner mode to remember selection
                self.focus_state = FocusState::SignalsTable(0);
                self.mark_dirty();
            }
            FocusState::SignalDetailModal => {
                // Left arrow in modal doesn't change tables
            }
        }
    }

    /// Handle right arrow - move to next table
    pub fn navigate_right(&mut self, tuner_count: usize) {
        let entering_scan_progress = match self.focus_state {
            FocusState::ScanProgress(_) => false,
            FocusState::Activities(_)
            | FocusState::TunersTable(_)
            | FocusState::SignalsTable(_) => true,
            FocusState::SignalDetailModal => false,
        };

        match self.focus_state {
            FocusState::Activities(_) => {
                if tuner_count > 0 {
                    self.focus_state = FocusState::TunersTable(0);
                } else {
                    self.focus_state = FocusState::SignalsTable(0);
                }
                self.mark_dirty();
            }
            FocusState::TunersTable(_) => {
                self.focus_state = FocusState::SignalsTable(0);
                self.mark_dirty();
            }
            FocusState::SignalsTable(_) => {
                self.focus_state = FocusState::ScanProgress(0);
                self.mark_dirty();
            }
            FocusState::ScanProgress(_) => {}
            FocusState::SignalDetailModal => {
                // Right arrow in modal doesn't change tables
            }
        }

        // When entering ScanProgress table from arrow key, transition to NavigatingScanner mode
        // (if not already in it), and restore to first signal
        if entering_scan_progress
            && matches!(self.focus_state, FocusState::ScanProgress(_))
            && !matches!(self.ui_mode, UiMode::NavigatingScanner { .. })
        {
            let rows = self.build_signal_rows();
            if !rows.is_empty() {
                let window_id = rows[0].window_id;
                self.ui_mode = UiMode::NavigatingScanner {
                    signal_index: 0,
                    window_id,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_cycle_includes_signals_table() {
        let mut model = Model::default();
        model.focus_state = FocusState::Activities(0);

        model.navigate_next_table(1); // 1 tuner available
        assert_eq!(model.focus_state, FocusState::TunersTable(0));

        model.navigate_next_table(1);
        assert_eq!(model.focus_state, FocusState::SignalsTable(0));

        model.navigate_next_table(1);
        assert_eq!(model.focus_state, FocusState::ScanProgress(0));
    }
}

#[cfg(test)]
mod navigate_stable_tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        ecs::SignalId,
        ui::tui::model::types::{AnalysisStatus, PlaybackState, SignalProgress, WindowProgress},
    };

    fn create_test_model_with_signals() -> Model {
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
                    signal_strength: None,
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
                    signal_strength: None,
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: SignalId::from_string("c3".to_string()),
                    frequency_hz: 88.9e6,
                    window_id: 0,
                    center_frequency_hz: 88.9e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: None,
                    signal_strength: None,
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
        model.focus_state = FocusState::ScanProgress(2);

        model
    }

    #[test]
    fn test_navigate_up_in_scan_progress_updates_focus_state() {
        let mut model = create_test_model_with_signals();
        model.focus_state = FocusState::ScanProgress(2);

        model.navigate_up();

        assert_eq!(model.focus_state, FocusState::ScanProgress(1));
    }

    #[test]
    fn test_navigate_up_wraps_around_at_top() {
        let mut model = create_test_model_with_signals();
        model.focus_state = FocusState::ScanProgress(0);

        model.navigate_up();

        assert_eq!(model.focus_state, FocusState::ScanProgress(2));
    }

    #[test]
    fn test_navigate_down_in_scan_progress_updates_focus_state() {
        let mut model = create_test_model_with_signals();
        model.focus_state = FocusState::ScanProgress(0);

        model.navigate_down();

        assert_eq!(model.focus_state, FocusState::ScanProgress(1));
    }

    #[test]
    fn test_navigate_down_wraps_around_at_bottom() {
        let mut model = create_test_model_with_signals();
        model.focus_state = FocusState::ScanProgress(2);

        model.navigate_down();

        assert_eq!(model.focus_state, FocusState::ScanProgress(0));
    }

    #[test]
    fn test_enter_selection_mode_sets_window_id() {
        let mut model = create_test_model_with_signals();

        model.enter_selection_mode();

        match model.ui_mode {
            UiMode::NavigatingScanner {
                signal_index,
                window_id,
            } => {
                assert_eq!(signal_index, 2); // Last signal (count - 1)
                assert_eq!(window_id, 0); // Window from last signal
            }
            _ => panic!("Expected NavigatingScanner mode"),
        }
    }

    #[test]
    fn test_select_next_signal_updates_window_id() {
        let mut model = Model::new();
        model.current_window = 1;

        // Window 0 with 1 signal
        let mut window0 = WindowProgress {
            window_id: 0,
            signals: vec![SignalProgress {
                signal_id: SignalId::from_string("c1".to_string()),
                frequency_hz: 88.1e6,
                window_id: 0,
                center_frequency_hz: 88.1e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::NotPlaying,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
                notes: None,
            }],
            is_complete: true,
            signal_lookup: std::collections::HashMap::new(),
        };
        window0
            .signal_lookup
            .insert(SignalId::from_string("c1".to_string()), 0);

        // Window 1 with 1 signal
        let mut window1 = WindowProgress {
            window_id: 1,
            signals: vec![SignalProgress {
                signal_id: SignalId::from_string("c2".to_string()),
                frequency_hz: 95.5e6,
                window_id: 1,
                center_frequency_hz: 95.5e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::NotPlaying,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
                notes: None,
            }],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };
        window1
            .signal_lookup
            .insert(SignalId::from_string("c2".to_string()), 0);

        model.windows.insert(0, window0);
        model.windows.insert(1, window1);

        // Start at first signal (window 0)
        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 0,
            window_id: 0,
        };

        model.select_next_signal();

        // Should move to second signal (window 0)
        // After reverse: [c2@window1, c1@window0]
        // Index 0 -> Index 1 means c2@window1 -> c1@window0
        match model.ui_mode {
            UiMode::NavigatingScanner {
                signal_index,
                window_id,
            } => {
                assert_eq!(signal_index, 1);
                assert_eq!(window_id, 0);
            }
            _ => panic!("Expected NavigatingScanner mode"),
        }
    }

    #[test]
    fn test_select_previous_signal_updates_window_id() {
        let mut model = Model::new();
        model.current_window = 1;

        let mut window0 = WindowProgress {
            window_id: 0,
            signals: vec![SignalProgress {
                signal_id: SignalId::from_string("c1".to_string()),
                frequency_hz: 88.1e6,
                window_id: 0,
                center_frequency_hz: 88.1e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::NotPlaying,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
                notes: None,
            }],
            is_complete: true,
            signal_lookup: std::collections::HashMap::new(),
        };
        window0
            .signal_lookup
            .insert(SignalId::from_string("c1".to_string()), 0);

        let mut window1 = WindowProgress {
            window_id: 1,
            signals: vec![SignalProgress {
                signal_id: SignalId::from_string("c2".to_string()),
                frequency_hz: 95.5e6,
                window_id: 1,
                center_frequency_hz: 95.5e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::NotPlaying,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
                notes: None,
            }],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };
        window1
            .signal_lookup
            .insert(SignalId::from_string("c2".to_string()), 0);

        model.windows.insert(0, window0);
        model.windows.insert(1, window1);

        // Start at second signal (window 1)
        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 1,
            window_id: 1,
        };

        model.select_previous_signal();

        // Should move to first signal (window 1)
        // After reverse: [c2@window1, c1@window0]
        // Index 1 -> Index 0 means c1@window0 -> c2@window1
        match model.ui_mode {
            UiMode::NavigatingScanner {
                signal_index,
                window_id,
            } => {
                assert_eq!(signal_index, 0);
                assert_eq!(window_id, 1);
            }
            _ => panic!("Expected NavigatingScanner mode"),
        }
    }

    #[test]
    fn test_adjust_scroll_to_selection_zero_viewport_height() {
        let mut model = create_test_model_with_signals();

        // Set up navigation mode
        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 0,
            window_id: 0,
        };

        // This used to panic with "attempt to subtract with overflow"
        // when viewport_height was 0 and selected_index was beyond scroll_offset
        model.adjust_scroll_to_selection(0);

        // Should handle gracefully without panicking
        // When viewport_height is 0, scroll_offset should stay at selection or adjust to it
        assert_eq!(model.scroll_offset, 0);
    }

    #[test]
    fn test_adjust_scroll_to_selection_one_viewport_height() {
        let mut model = create_test_model_with_signals();

        model.ui_mode = UiMode::NavigatingScanner {
            signal_index: 2,
            window_id: 0,
        };
        model.scroll_offset = 0;

        // With viewport_height of 1, selecting at index 2
        // should position scroll_offset at 2 - (1 - 1) = 2
        model.adjust_scroll_to_selection(1);

        assert_eq!(model.scroll_offset, 2);
    }

    #[test]
    fn test_signals_table_navigation_should_work_with_confirmed_signals() {
        // This test will fail because signals_table_row_count() returns 0
        // even when there are confirmed signals in the model
        let mut model = create_test_model_with_signals();

        // Focus on signals table
        model.focus_state = FocusState::SignalsTable(0);

        // Verify we have confirmed signals to navigate through
        let confirmed_signals = model.build_confirmed_signal_rows();
        assert!(
            !confirmed_signals.is_empty(),
            "Model should have confirmed signals for navigation testing"
        );

        // The bug: signals_table_row_count() returns 0 instead of confirmed_signals.len()
        let row_count = model.signals_table_row_count();
        assert_eq!(
            row_count,
            confirmed_signals.len(),
            "signals_table_row_count() should return count of confirmed signals but returned {} \
             instead of {}",
            row_count,
            confirmed_signals.len()
        );

        // Navigation down should work when there are signals
        let initial_state = model.focus_state;
        model.navigate_down();

        // Should have moved to index 1 if there are 2+ signals, or stay at 0 if only 1
        match (initial_state, model.focus_state) {
            (FocusState::SignalsTable(0), FocusState::SignalsTable(new_idx)) => {
                if confirmed_signals.len() > 1 {
                    assert_eq!(
                        new_idx, 1,
                        "Should navigate to index 1 when there are multiple signals"
                    );
                } else {
                    assert_eq!(
                        new_idx, 0,
                        "Should stay at index 0 when there's only one signal"
                    );
                }
            }
            _ => panic!("Navigation should maintain SignalsTable focus state"),
        }

        // Navigation up should also work
        model.focus_state = FocusState::SignalsTable(1);
        model.navigate_up();

        match model.focus_state {
            FocusState::SignalsTable(idx) => {
                assert_eq!(idx, 0, "Should navigate up to index 0");
            }
            _ => panic!("Navigation should maintain SignalsTable focus state"),
        }
    }
}
