//! Device management methods

use crate::hardware::DeviceInfo;
use tracing::debug;

use super::{state::Model, types::TunerState};

impl Model {
    /// Add a newly discovered device
    pub fn add_device(&mut self, device: DeviceInfo) {
        let device_id = device.id.clone();

        // If this is a cached device (pre-enumerated), do nothing - it's already there
        if self.cached_devices.contains_key(&device_id) {
            return;
        }

        // Populate tuners immediately from dynamically discovered device
        for tuner in &device.tuners {
            let tuner_info = crate::ui::tui::model::TunerInfo {
                id: tuner.id.clone(),
                label: tuner.label.clone(),
            };
            if self.tuners.insert(tuner_info.clone()) {
                debug!(
                    tuner_id = ?tuner_info.id,
                    label = %tuner_info.label,
                    "Populating tuner from dynamically discovered device"
                );
            }
        }

        self.devices.insert(device_id, device);
    }

    /// Remove a device and all its tuners
    pub fn remove_device(&mut self, device_id: &crate::hardware::DeviceId) {
        // Only remove from devices (cached devices are never removed)
        if self.devices.remove(device_id).is_some() {
            self.tuners.retain(|t| &t.id.device_id != device_id);
            debug!(device_id = ?device_id, "Device and all its tuners removed from TUI model");
        }
    }

    /// Get the state of a specific tuner based on pool status
    pub fn tuner_state(&self, tuner_id: &crate::hardware::pool::TunerId) -> TunerState {
        if let Some(tuner) = self.pool_info.get(tuner_id) {
            match (&tuner.state, &tuner.activity) {
                (crate::hardware::pool::TunerState::Available, _) => TunerState::Available,
                (crate::hardware::pool::TunerState::Allocated, Some(activity)) => match activity {
                    crate::hardware::pool::TunerActivity::Scanning => TunerState::Scanning,
                    crate::hardware::pool::TunerActivity::Listening => TunerState::Listening,
                    crate::hardware::pool::TunerActivity::Other => TunerState::Available,
                },
                (crate::hardware::pool::TunerState::Allocated, None) => TunerState::Available,
            }
        } else {
            TunerState::Available
        }
    }

    /// Get count of discovered tuners
    pub fn device_count(&self) -> usize {
        self.tuners.len()
    }
}
