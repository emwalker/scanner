//! Core state management for TUI model

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::types::{FocusState, SpectrumStation, UiMode, WindowProgress};
use crate::{
    ecs::entities::{TaskId, TaskWindowCell},
    hardware::pool::TunerId,
    ui::tui::renderers::table_styles::{RowGroup, TableRow},
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
    pub tasks: Vec<TaskSummary>,
    pub tuners: BTreeSet<TunerInfo>,
    pub pool_info: HashMap<TunerId, crate::hardware::pool::TunerStatus>,
    pub pool_status: Option<crate::hardware::pool::PoolStatus>,
    pub devices: HashMap<crate::hardware::DeviceId, crate::hardware::DeviceInfo>,
    pub spectrum_stations: Vec<SpectrumStation>,
    pub active_audio_frequency: Option<f64>,
    pub active_tuner_id: Option<TunerId>,
    pub global_pause_resource: Option<crate::ecs::GlobalPauseResource>,
    pub activities_scroll: crate::ui::tui::renderers::table_styles::ScrollState,
    pub tuners_scroll: crate::ui::tui::renderers::table_styles::ScrollState,
    pub scan_progress_scroll: crate::ui::tui::renderers::table_styles::ScrollState,
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
            tasks: Vec::new(),
            tuners: BTreeSet::new(),
            pool_info: HashMap::new(),
            pool_status: None,
            devices: HashMap::new(),
            spectrum_stations: Vec::new(),
            active_audio_frequency: None,
            active_tuner_id: None,
            global_pause_resource: None,
            activities_scroll: crate::ui::tui::renderers::table_styles::ScrollState::default(),
            tuners_scroll: crate::ui::tui::renderers::table_styles::ScrollState::default(),
            scan_progress_scroll: crate::ui::tui::renderers::table_styles::ScrollState::default(),
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

    /// Get row count for Scan Progress table (for displayed scan)
    pub fn scan_signals_row_count(&self) -> usize {
        use crate::ui::tui::renderers::table_styles::VisibilityContext;

        let context = VisibilityContext::new(None, Some(self.current_window));
        self.count_visible_signals(&context)
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
}
