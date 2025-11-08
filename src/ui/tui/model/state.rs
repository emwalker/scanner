//! Core state management for TUI model

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::types::{
    FocusState, SignalDetailModal, SignalProgress, SpectrumStation, UiMode, WindowProgress,
};
use crate::{
    ecs::entities::{TaskId, TaskWindowCell},
    hardware::pool::TunerId,
    persistence::{location::Location, storage::SignalStorage, types::PersistedSignal},
    ui::tui::renderers::table_styles::{RowGroup, ScrollState, TableRow},
};

/// Summary of task data for display in Activities table
#[derive(Debug, Clone)]
pub struct TaskSummary {
    pub task_id: TaskId,
    pub label: String,
    pub summary: String,
    pub activity: String,
    pub assigned_tuner: Option<String>,
    pub assigned_tuner_id: Option<crate::hardware::pool::TunerId>,
    pub window_cell_data: TaskWindowCell,
}

/// Information about an individual tuner (channel) for UI display
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TunerInfo {
    pub id: TunerId,
    pub label: String,
}

impl Ord for TunerInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.label, &self.id).cmp(&(&other.label, &other.id))
    }
}

impl PartialOrd for TunerInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl TableRow for TaskSummary {
    fn build_cells(
        &self,
        _theme: &dyn crate::ui::tui::themes::Theme,
    ) -> Vec<ratatui::widgets::Cell<'static>> {
        use ratatui::widgets::Cell;

        let window_cell = match &self.window_cell_data {
            TaskWindowCell::SpectrumBar {
                full_range_hz,
                current_window_hz,
            } => crate::ui::tui::renderers::activities::render_window_cell_content(
                full_range_hz,
                current_window_hz,
                24,
            ),
        };

        vec![
            Cell::from(self.label.clone()),
            Cell::from(self.summary.clone()),
            Cell::from(window_cell),
            Cell::from(self.assigned_tuner.clone().unwrap_or_default()),
            Cell::from(self.activity.clone()),
        ]
    }
}

impl RowGroup for TaskSummary {
    type GroupId = ();

    fn group_id(&self) -> Self::GroupId {}
}

/// Main application model following The Elm Architecture
#[derive(Debug)]
pub struct Model {
    pub windows: BTreeMap<usize, WindowProgress>,
    pub current_window: usize,
    pub total_windows: Option<usize>,
    pub should_quit: bool,
    pub theme_selector_open: bool,
    pub theme_selector_index: usize,
    pub ui_mode: UiMode,
    pub scroll_offset: usize,
    pub playback_active: bool,
    pub focus_state: FocusState,
    pub displayed_task_id: Option<TaskId>,
    pub activities_selected_index: usize,
    pub tuners_selected_index: usize,
    pub signals_table_scroll: ScrollState,
    pub signals_table_selection: usize,
    pub tasks: Vec<TaskSummary>,
    pub tuners: BTreeSet<TunerInfo>,
    pub pool_info: HashMap<TunerId, crate::hardware::pool::TunerStatus>,
    pub pool_status: Option<crate::hardware::pool::PoolStatus>,
    pub devices: HashMap<crate::hardware::DeviceId, crate::hardware::DeviceInfo>,
    pub spectrum_stations: Vec<SpectrumStation>,
    pub active_audio_frequency: Option<f64>,
    pub active_tuner_id: Option<TunerId>,
    pub global_pause_resource: Option<crate::ecs::GlobalPauseResource>,
    pub activities_scroll: ScrollState,
    pub tuners_scroll: ScrollState,
    pub scan_progress_scroll: ScrollState,
    pub notes_input: crate::ui::tui::widgets::NotesInput,
    pub editing_signal_id: Option<crate::ecs::components::SignalId>,
    pub signal_detail_modal: Option<SignalDetailModal>,
    pub previous_signals_table_selection: Option<usize>,
    pub persistent_signals: Vec<PersistedSignal>,
    pub persistent_signal_ids: std::collections::HashMap<u64, crate::ecs::components::SignalId>, /* frequency -> SignalId mapping */
    dirty: bool,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self {
            windows: BTreeMap::new(),
            current_window: 0,
            total_windows: None,
            should_quit: false,
            theme_selector_open: false,
            theme_selector_index: 0,
            ui_mode: UiMode::Idle,
            scroll_offset: 0,
            playback_active: false,
            focus_state: FocusState::Activities(0),
            displayed_task_id: None,
            activities_selected_index: 0,
            tuners_selected_index: 0,
            signals_table_scroll: ScrollState::default(),
            signals_table_selection: 0,
            tasks: Vec::new(),
            tuners: BTreeSet::new(),
            pool_info: HashMap::new(),
            pool_status: None,
            devices: HashMap::new(),
            spectrum_stations: Vec::new(),
            active_audio_frequency: None,
            active_tuner_id: None,
            global_pause_resource: None,
            activities_scroll: ScrollState::default(),
            tuners_scroll: ScrollState::default(),
            scan_progress_scroll: ScrollState::default(),
            notes_input: crate::ui::tui::widgets::NotesInput::new(),
            editing_signal_id: None,
            signal_detail_modal: None,
            previous_signals_table_selection: None,
            persistent_signals: Vec::new(),
            persistent_signal_ids: HashMap::new(),
            dirty: true,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Load persistent signals from storage and add to model state
    /// Following Elm Architecture - this updates Model state with persistent data
    pub fn load_persistent_signals_from_storage(
        &mut self,
        storage: &SignalStorage,
        location: Location,
    ) -> Result<(), Box<dyn std::error::Error>> {
        tracing::debug!(
            lat = location.lat,
            lon = location.lon,
            "Loading persistent signals from storage"
        );
        let loaded_signals = storage.load_signals_for_location(location)?;
        tracing::debug!(
            signal_count = loaded_signals.len(),
            "Loaded persistent signals"
        );

        // Create SignalIds for persistent signals to unify with scan signals
        self.persistent_signal_ids.clear();

        for signal in &loaded_signals {
            tracing::debug!(
                frequency_hz = signal.frequency_hz,
                notes = signal.notes,
                "Loaded persistent signal"
            );

            // Create a consistent SignalId for this persistent signal
            let signal_id = crate::ecs::components::SignalId::new(
                signal.frequency_hz,
                signal.modulation.clone(),
            );
            let freq_key = signal.frequency_hz as u64;
            self.persistent_signal_ids.insert(freq_key, signal_id);
        }

        self.persistent_signals = loaded_signals;
        self.mark_dirty();
        Ok(())
    }

    /// Handle ENTER key when focused on signals table
    pub fn handle_signal_table_enter_key(&mut self, _key: &crossterm::event::KeyEvent) {
        if let FocusState::SignalsTable(selected_index) = self.focus_state {
            let confirmed_signals = self.build_confirmed_signal_rows();
            if let Some(signal_row) = confirmed_signals.get(selected_index) {
                // Find signal by frequency - this now works for both scan and persistent signals
                if let Some(signal_id) = self.find_signal_id_by_frequency(signal_row.frequency_hz) {
                    // Save the current selection before opening modal
                    self.previous_signals_table_selection = Some(selected_index);
                    self.open_signal_detail_modal_with_frequency(
                        signal_id,
                        signal_row.frequency_hz,
                    );
                }
            }
        }
    }

    /// Open signal detail modal for the given signal ID with explicit frequency
    /// This ensures stable frequency-based lookup for signal editing operations
    pub fn open_signal_detail_modal_with_frequency(
        &mut self,
        signal_id: crate::ecs::components::SignalId,
        frequency_hz: f64,
    ) {
        // Find the signal to get existing notes
        let existing_notes = self.find_signal_notes(&signal_id);
        self.signal_detail_modal = Some(SignalDetailModal::new(
            signal_id,
            frequency_hz,
            existing_notes,
        ));
        self.focus_state = FocusState::SignalDetailModal;
    }

    /// Open signal detail modal for the given signal ID (legacy method)
    pub fn open_signal_detail_modal(&mut self, signal_id: crate::ecs::components::SignalId) {
        // For backward compatibility, try to extract frequency from SignalId string
        // This is fragile but maintains existing API compatibility
        let frequency_hz = self
            .extract_frequency_from_signal_id(&signal_id)
            .unwrap_or(0.0);
        self.open_signal_detail_modal_with_frequency(signal_id, frequency_hz);
    }

    /// Extract frequency from SignalId string (fragile but necessary for compatibility)
    fn extract_frequency_from_signal_id(
        &self,
        signal_id: &crate::ecs::components::SignalId,
    ) -> Option<f64> {
        // SignalId format: "{frequency_mhz}-{task_id}-{window_index}"
        let signal_str = signal_id.as_str();
        if let Some(first_dash) = signal_str.find('-')
            && let Ok(frequency_mhz) = signal_str[..first_dash].parse::<f64>()
        {
            return Some(frequency_mhz * 1e6); // Convert MHz to Hz
        }
        None
    }

    /// Close signal detail modal
    pub fn close_signal_detail_modal(&mut self) {
        self.signal_detail_modal = None;
        // Restore the previous selection if it exists, otherwise default to 0
        let selection_index = self.previous_signals_table_selection.unwrap_or(0);
        self.focus_state = FocusState::SignalsTable(selection_index);
        self.previous_signals_table_selection = None; // Clear saved selection
    }

    /// Handle ESC key when modal is open
    pub fn handle_modal_escape_key(&mut self, _key: &crossterm::event::KeyEvent) {
        self.close_signal_detail_modal();
    }

    /// Find signal by frequency to get the original SignalProgress
    pub fn find_signal_by_frequency(&self, frequency_hz: f64) -> Option<&SignalProgress> {
        for window in self.windows.values() {
            for signal in &window.signals {
                if (signal.frequency_hz - frequency_hz).abs() < 1000.0 {
                    // 1kHz tolerance
                    return Some(signal);
                }
            }
        }
        None
    }

    /// Find signal notes by signal ID - works for both scan and persistent signals
    fn find_signal_notes(&self, signal_id: &crate::ecs::components::SignalId) -> Option<String> {
        // First try scan signals
        for window in self.windows.values() {
            if let Some(index) = window.signal_lookup.get(signal_id)
                && let Some(signal) = window.signals.get(*index)
            {
                return signal.notes.clone();
            }
        }

        // Then try persistent signals - find by matching SignalId to frequency
        for (stored_freq_key, stored_signal_id) in &self.persistent_signal_ids {
            if stored_signal_id == signal_id {
                let frequency_hz = *stored_freq_key as f64;
                // Find the persistent signal with this frequency and return its notes
                return self
                    .persistent_signals
                    .iter()
                    .find(|s| (s.frequency_hz - frequency_hz).abs() < 1000.0)
                    .and_then(|s| s.notes.clone());
            }
        }

        None
    }

    /// Find SignalId by frequency - works for both scan and persistent signals
    /// This is the unified method that handles all signal types
    pub fn find_signal_id_by_frequency(
        &self,
        frequency_hz: f64,
    ) -> Option<crate::ecs::components::SignalId> {
        // First try scan signals (if any exist)
        if let Some(signal_progress) = self.find_signal_by_frequency(frequency_hz) {
            return Some(signal_progress.signal_id.clone());
        }

        // Then try persistent signals using pre-created SignalIds
        let freq_key = frequency_hz as u64;
        if let Some(signal_id) = self.persistent_signal_ids.get(&freq_key) {
            return Some(signal_id.clone());
        }

        // Fallback: search persistent signals with tolerance for frequency matching
        for (stored_freq_key, signal_id) in &self.persistent_signal_ids {
            let stored_freq = *stored_freq_key as f64;
            if (stored_freq - frequency_hz).abs() < 1000.0 {
                // 1kHz tolerance
                return Some(signal_id.clone());
            }
        }

        None
    }

    pub fn set_global_pause_resource(&mut self, resource: crate::ecs::GlobalPauseResource) {
        self.global_pause_resource = Some(resource);
    }

    pub fn is_globally_paused(&self) -> bool {
        if let Some(ref resource) = self.global_pause_resource
            && let Ok(state) = resource.lock()
        {
            return matches!(*state, crate::ecs::GlobalPauseState::Paused { .. });
        }
        false
    }

    /// Get row count for Activities table
    pub fn activities_row_count(&self) -> usize {
        self.tasks.len()
    }

    /// Get row count for Tuners table
    pub fn tuners_row_count(&self) -> usize {
        self.tuners.len()
    }

    /// Get row count for Signals table
    pub fn signals_table_row_count(&self) -> usize {
        self.build_confirmed_signal_rows().len()
    }

    /// Get row count for Scan Progress table (for displayed scan)
    pub fn scan_signals_row_count(&self) -> usize {
        use crate::ui::tui::renderers::table_styles::VisibilityContext;

        let context = VisibilityContext::new(None, Some(self.current_window));
        self.count_visible_signals(&context)
    }

    /// Start editing notes for the currently selected signal
    pub fn start_editing_notes(&mut self) {
        if let Some(signal_info) = self.selected_signal_info() {
            // Find the signal to get its current notes
            if let Some(window) = self.windows.get(&signal_info.window_id)
                && let Some(signal) = window
                    .signals
                    .iter()
                    .find(|s| s.signal_id == signal_info.signal_id)
            {
                let current_notes = signal.notes.clone().unwrap_or_else(String::new);
                self.notes_input =
                    crate::ui::tui::widgets::NotesInput::with_content(&current_notes);
                self.notes_input.activate();
                self.editing_signal_id = Some(signal_info.signal_id.clone());
                self.mark_dirty();
            }
        }
    }

    /// Cancel notes editing
    pub fn cancel_editing_notes(&mut self) {
        self.notes_input.deactivate();
        self.editing_signal_id = None;
        self.mark_dirty();
    }

    /// Save notes for the currently editing signal
    pub fn save_editing_notes(&mut self) -> Option<(crate::ecs::components::SignalId, String)> {
        if let Some(signal_id) = &self.editing_signal_id {
            let notes = self.notes_input.content().to_string();
            let result = Some((signal_id.clone(), notes.clone()));

            // Update the signal's notes in the local model
            if let Some(signal_info) = self.selected_signal_info()
                && let Some(window) = self.windows.get_mut(&signal_info.window_id)
                && let Some(signal) = window
                    .signals
                    .iter_mut()
                    .find(|s| s.signal_id == signal_info.signal_id)
            {
                signal.notes = if notes.is_empty() { None } else { Some(notes) };
            }

            self.cancel_editing_notes();
            result
        } else {
            None
        }
    }

    /// Check if currently editing notes
    pub fn is_editing_notes(&self) -> bool {
        self.editing_signal_id.is_some() && self.notes_input.is_active()
    }

    /// Get the previous signals table selection (used when modal is open)
    pub fn get_previous_signals_table_selection(&self) -> Option<usize> {
        self.previous_signals_table_selection
    }

    /// Check if modal should be rendered
    pub fn should_render_modal(&self) -> bool {
        self.signal_detail_modal.is_some()
    }

    /// Check if modal should handle keyboard input
    pub fn should_handle_modal_input(&self, _key: &crossterm::event::KeyEvent) -> bool {
        // Modal handles input when it's open and has focus
        self.signal_detail_modal.is_some()
            && matches!(
                self.focus_state,
                super::types::FocusState::SignalDetailModal
            )
    }

    /// Handle text input in modal for editing notes
    pub fn handle_modal_text_input(&mut self, key: &crossterm::event::KeyEvent) {
        if let Some(modal) = &mut self.signal_detail_modal
            && let crossterm::event::KeyCode::Char(ch) = key.code
        {
            modal.notes_input.push(ch);
            modal.is_notes_dirty = true;
            self.mark_dirty();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::{renderers::table_styles::TableRow, themes::basic::BasicDarkTheme};

    #[test]
    fn test_task_summary_builds_cells() {
        use crate::ecs::entities::TaskWindowCell;

        let task = TaskSummary {
            task_id: crate::ecs::entities::TaskId::new("test-task".to_string()),
            label: "Test".to_string(),
            summary: "Summary".to_string(),
            activity: "Active".to_string(),
            assigned_tuner: Some("Tuner 1".to_string()),
            assigned_tuner_id: None,
            window_cell_data: TaskWindowCell::SpectrumBar {
                full_range_hz: (88.0e6, 108.0e6),
                current_window_hz: None,
            },
        };

        let theme = BasicDarkTheme;
        let cells = task.build_cells(&theme);
        assert_eq!(cells.len(), 5);
    }

    #[test]
    fn test_model_has_signals_table_state() {
        let model = Model::default();
        assert_eq!(model.signals_table_scroll.offset, 0);
        assert_eq!(model.signals_table_selection, 0);
    }
}
