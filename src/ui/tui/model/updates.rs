//! Event processing and state update methods

use tracing::debug;

use super::state::Model;
use crate::ui::TuiEvent;

impl Model {
    /// Update the model based on a TUI event
    pub fn update_tui_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::TunerAdded(tuner) => {
                debug!(device_id = ?tuner.id, "TUI Model received TunerAdded event");
                self.add_device(tuner);
            }
            TuiEvent::TunerRemoved(tuner_id) => {
                debug!(device_id = ?tuner_id, "TUI Model received TunerRemoved event");
                self.remove_device(&tuner_id);
            }
            TuiEvent::Paused { tuner_id } => {
                debug!(tuner_id = ?tuner_id, ui_mode = ?self.ui_mode, "Scanning paused, tuner now available");
            }
            TuiEvent::ActiveTunersUpdated { status } => {
                debug!(
                    event_tuner_count = status.tuners.len(),
                    "TUI Model: ActiveTunersUpdated event RECEIVED"
                );

                for tuner in &status.tuners {
                    debug!(
                        tuner_id = ?tuner.id,
                        state = ?tuner.state,
                        activity = ?tuner.activity,
                        "TUI Model: Event contains tuner"
                    );
                }

                let status_changed = if let Some(prev_status) = &self.pool_status {
                    prev_status.tuners.len() != status.tuners.len()
                        || prev_status.tuners.iter().zip(status.tuners.iter()).any(
                            |(prev, curr)| {
                                prev.state != curr.state || prev.activity != curr.activity
                            },
                        )
                } else {
                    true
                };

                if !status_changed {
                    debug!("TUI Model: Status unchanged, skipping pool_info update");
                    return;
                }

                debug!(
                    total_tuners = status.tuners.len(),
                    available_count = status
                        .tuners
                        .iter()
                        .filter(|t| t.state == crate::hardware::pool::TunerState::Available)
                        .count(),
                    allocated_count = status
                        .tuners
                        .iter()
                        .filter(|t| t.state == crate::hardware::pool::TunerState::Allocated)
                        .count(),
                    scanning_count = status
                        .tuners
                        .iter()
                        .filter(
                            |t| t.activity == Some(crate::hardware::pool::TunerActivity::Scanning)
                        )
                        .count(),
                    listening_count = status
                        .tuners
                        .iter()
                        .filter(
                            |t| t.activity == Some(crate::hardware::pool::TunerActivity::Listening)
                        )
                        .count(),
                    "Pool status updated"
                );

                // Debug each tuner's state
                for tuner in &status.tuners {
                    debug!(
                        device_id = ?tuner.id.device_id,
                        state = ?tuner.state,
                        activity = ?tuner.activity,
                        "Tuner status"
                    );
                }

                // Build pool_info HashMap for O(1) lookups
                self.pool_info = status
                    .tuners
                    .iter()
                    .map(|t| (t.id.clone(), t.clone()))
                    .collect();

                // Populate tuners from all devices
                for device in self.devices.values() {
                    for tuner in &device.tuners {
                        let tuner_info = crate::ui::tui::model::TunerInfo {
                            id: tuner.id.clone(),
                            label: tuner.label.clone(),
                        };

                        if self.tuners.insert(tuner_info.clone()) {
                            debug!(
                                tuner_id = ?tuner_info.id,
                                label = %tuner_info.label,
                                "Adding tuner to UI"
                            );
                        }
                    }
                }

                self.pool_status = Some(status);
                self.mark_dirty();
            }
        }
    }
}
