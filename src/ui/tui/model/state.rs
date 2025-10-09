//! Core state management for TUI model

use crate::hardware::DeviceInfo;
use std::collections::{BTreeMap, HashMap};

use super::types::{FocusState, TunerState, UiMode, WindowProgress};

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
    pub tuners: Vec<DeviceInfo>,
    pub tuner_states: HashMap<crate::hardware::DeviceId, TunerState>,
    pub pool_status: Option<crate::hardware::pool::PoolStatus>,
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
            tuners: Vec::new(),
            tuner_states: HashMap::new(),
            pool_status: None,
        }
    }
}
