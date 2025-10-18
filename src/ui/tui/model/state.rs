//! Core state management for TUI model

use crate::hardware::pool::TunerId;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::types::{FocusState, SpectrumStation, UiMode, WindowProgress};

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
    pub tuners: BTreeSet<TunerInfo>,
    pub pool_info: HashMap<TunerId, crate::hardware::pool::TunerStatus>,
    pub pool_status: Option<crate::hardware::pool::PoolStatus>,
    pub devices: HashMap<crate::hardware::DeviceId, crate::hardware::DeviceInfo>,
    pub spectrum_stations: Vec<SpectrumStation>,
    pub active_audio_frequency: Option<f64>,
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
            focus_state: FocusState::Spectrum,
            tuners: BTreeSet::new(),
            pool_info: HashMap::new(),
            pool_status: None,
            devices: HashMap::new(),
            spectrum_stations: Vec::new(),
            active_audio_frequency: None,
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
}
