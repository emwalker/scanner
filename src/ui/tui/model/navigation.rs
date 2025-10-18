//! Navigation and state-modifying methods for TUI model

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

    /// Enter selection mode - pauses scanning and allows browsing candidates
    pub fn enter_selection_mode(&mut self) {
        let candidate_count = self.selectable_candidate_count();
        if candidate_count > 0 {
            let selected_index = candidate_count - 1;
            self.ui_mode = UiMode::NavigatingScanner { selected_index };
            self.mark_dirty();
        }
    }

    /// Exit selection mode - returns to normal scanning
    pub fn exit_selection_mode(&mut self) {
        self.ui_mode = UiMode::Idle;
        self.mark_dirty();
    }

    /// Exit browsing mode and return to normal scanning (clears both modes)
    pub fn exit_browsing_mode(&mut self) {
        self.ui_mode = UiMode::Idle;
        self.mark_dirty();
    }

    /// Select next candidate (moving forward in time)
    pub fn select_next_candidate(&mut self) {
        self.select_next_candidate_with_viewport(20);
    }

    /// Select next candidate with viewport height for scroll adjustment
    pub fn select_next_candidate_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let candidate_count = self.selectable_candidate_count();
        if candidate_count == 0 {
            return;
        }

        let current = self.selected_candidate_index().unwrap_or(0);
        let next = (current + 1).min(candidate_count);

        if next != current {
            match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => {
                    self.ui_mode = UiMode::NavigatingScanner {
                        selected_index: next,
                    };
                }
                UiMode::AwaitingTune { tuning_index, .. } => {
                    self.ui_mode = UiMode::AwaitingTune {
                        navigation_index: next,
                        tuning_index: *tuning_index,
                    };
                }
                UiMode::Listening {
                    playing_index,
                    playing_candidate_id,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: next,
                        playing_index: *playing_index,
                        playing_candidate_id: playing_candidate_id.clone(),
                    };
                }
                UiMode::Idle => {}
            }
            self.adjust_scroll_to_selection(viewport_height);
            self.mark_dirty();
        }
    }

    /// Select previous candidate (moving backward in time)
    pub fn select_previous_candidate(&mut self) {
        self.select_previous_candidate_with_viewport(20);
    }

    /// Select previous candidate with viewport height for scroll adjustment
    pub fn select_previous_candidate_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let current = self.selected_candidate_index().unwrap_or(0);
        if current > 0 {
            let prev = current - 1;
            match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => {
                    self.ui_mode = UiMode::NavigatingScanner {
                        selected_index: prev,
                    };
                }
                UiMode::AwaitingTune { tuning_index, .. } => {
                    self.ui_mode = UiMode::AwaitingTune {
                        navigation_index: prev,
                        tuning_index: *tuning_index,
                    };
                }
                UiMode::Listening {
                    playing_index,
                    playing_candidate_id,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: prev,
                        playing_index: *playing_index,
                        playing_candidate_id: playing_candidate_id.clone(),
                    };
                }
                UiMode::Idle => {}
            }
            self.adjust_scroll_to_selection(viewport_height);
            self.mark_dirty();
        }
    }

    /// Adjust scroll offset to ensure the selected candidate is visible
    pub fn adjust_scroll_to_selection(&mut self, viewport_height: usize) {
        if let Some(selected_idx) = self.selected_candidate_index() {
            if selected_idx < self.scroll_offset {
                self.scroll_offset = selected_idx;
                self.mark_dirty();
            } else if selected_idx >= self.scroll_offset + viewport_height {
                self.scroll_offset = selected_idx.saturating_sub(viewport_height - 1);
                self.mark_dirty();
            }
        }
    }

    /// Scroll up by one line
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
            self.mark_dirty();
        }
    }

    /// Scroll down by one line
    pub fn scroll_down(&mut self, total_candidates: usize, viewport_height: usize) {
        if self.scroll_offset + viewport_height < total_candidates {
            self.scroll_offset += 1;
            self.mark_dirty();
        }
    }

    /// Handle arrow down navigation based on current focus state
    pub fn navigate_down(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {
                self.focus_state = FocusState::Scan;
                self.mark_dirty();
            }
            FocusState::Scan => {
                if self.selection_mode() {
                    self.select_next_candidate();
                }
            }
            FocusState::Tuner(_) => {}
        }
    }

    /// Handle arrow up navigation based on current focus state
    pub fn navigate_up(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {
                if !self.selection_mode() {
                    let candidate_count = self.selectable_candidate_count();
                    if candidate_count > 0 {
                        let selected_index = candidate_count - 1;
                        self.ui_mode = UiMode::NavigatingScanner { selected_index };
                    }
                    self.focus_state = FocusState::Scan;
                    self.mark_dirty();
                } else {
                    let prev_idx = self.selected_candidate_index();
                    self.select_previous_candidate();

                    if self.selected_candidate_index().is_none() && prev_idx.is_some() {
                        self.focus_state = FocusState::Spectrum;
                        self.exit_selection_mode();
                    }
                }
            }
            FocusState::Tuner(_) => {}
        }
    }

    /// Handle arrow right navigation based on current focus state
    pub fn navigate_right(&mut self, tuner_count: usize) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {
                if self.selection_mode() && tuner_count > 0 {
                    self.focus_state = FocusState::Tuner(0);
                    self.mark_dirty();
                }
            }
            FocusState::Tuner(idx) => {
                if idx + 1 < tuner_count {
                    self.focus_state = FocusState::Tuner(idx + 1);
                    self.mark_dirty();
                }
            }
        }
    }

    /// Handle arrow left navigation based on current focus state
    pub fn navigate_left(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {}
            FocusState::Tuner(idx) => {
                if idx == 0 {
                    self.focus_state = FocusState::Scan;
                    self.mark_dirty();
                } else {
                    self.focus_state = FocusState::Tuner(idx - 1);
                    self.mark_dirty();
                }
            }
        }
    }
}
