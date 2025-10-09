//! Device management methods

use crate::hardware::DeviceInfo;
use tracing::debug;

use super::{state::Model, types::TunerState};

impl Model {
    /// Add a newly discovered tuner
    pub fn add_device(&mut self, tuner: DeviceInfo) {
        if !self.tuners.iter().any(|d| d.id == tuner.id) {
            debug!(tuner_id = ?tuner.id, label = %tuner.label, "Tuner added to TUI model");

            self.tuner_states
                .insert(tuner.id.clone(), TunerState::Available);
            self.tuners.push(tuner);
        }
    }

    /// Remove a tuner that was unplugged
    pub fn remove_device(&mut self, tuner_id: &crate::hardware::DeviceId) {
        if let Some(pos) = self.tuners.iter().position(|d| &d.id == tuner_id) {
            let tuner = self.tuners.remove(pos);
            self.tuner_states.remove(&tuner.id);
            debug!(tuner_id = ?tuner.id, label = %tuner.label, "Tuner removed from TUI model");
        }
    }

    /// Get the state of a specific tuner based on pool status
    pub fn tuner_state(&self, tuner_id: &crate::hardware::DeviceId) -> TunerState {
        if let Some(ref status) = self.pool_status {
            for tuner in &status.tuners {
                if &tuner.id.device_id == tuner_id {
                    return match (&tuner.state, &tuner.activity) {
                        (crate::hardware::pool::TunerState::Available, _) => TunerState::Available,
                        (crate::hardware::pool::TunerState::Allocated, Some(activity)) => {
                            match activity {
                                crate::hardware::pool::TunerActivity::Scanning => {
                                    TunerState::Scanning
                                }
                                crate::hardware::pool::TunerActivity::Listening => {
                                    TunerState::Listening
                                }
                                crate::hardware::pool::TunerActivity::Other => {
                                    TunerState::Available
                                }
                            }
                        }
                        (crate::hardware::pool::TunerState::Allocated, None) => {
                            TunerState::Available
                        }
                    };
                }
            }
        }
        self.tuner_states
            .get(tuner_id)
            .cloned()
            .unwrap_or(TunerState::Available)
    }

    /// Get count of discovered tuners
    pub fn device_count(&self) -> usize {
        self.tuners.len()
    }
}
