//! Core state management for TUI model

use crate::hardware::pool::TunerId;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::types::{FocusState, UiMode, WindowProgress};

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
    pub cached_devices: HashMap<crate::hardware::DeviceId, crate::hardware::DeviceInfo>,
    pub devices: HashMap<crate::hardware::DeviceId, crate::hardware::DeviceInfo>,
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
            cached_devices: HashMap::new(),
            devices: HashMap::new(),
        }
    }

    pub fn with_cached_devices(mut self, devices: Vec<crate::hardware::DeviceInfo>) -> Self {
        use tracing::debug;
        debug!(
            device_count = devices.len(),
            "Initializing model with cached devices"
        );
        for device in &devices {
            debug!(
                device_id = ?device.id,
                label = %device.label,
                tuner_count = device.tuners.len(),
                "Caching device"
            );

            for tuner in &device.tuners {
                let tuner_info = super::TunerInfo {
                    id: tuner.id.clone(),
                    label: tuner.label.clone(),
                };
                self.tuners.insert(tuner_info);
                debug!(
                    tuner_id = ?tuner.id,
                    label = %tuner.label,
                    "Populating tuner from cached device"
                );
            }
        }
        self.cached_devices = devices.into_iter().map(|d| (d.id.clone(), d)).collect();
        self
    }
}
