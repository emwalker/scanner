//! Device enumeration task - discovers available SDR devices for a backend

use crate::core::types::{Result, ScannerError};
use crate::discovery::tracker::DeviceTracker;
use crate::hardware::Capabilities;
use crate::hardware::backend::Backend as BackendTrait;
use crate::hardware::pool::{AddDeviceResult, Pool};
use crate::hardware::types::Backend;
use crate::task::TaskContinuation;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Device enumeration task - discovers available SDR devices for a backend
///
/// This task is per-backend and serialized through the backend queue.
/// When scheduled, it queries the backend API for available devices,
/// updates the pool, and emits discovery events for the TUI.
#[allow(dead_code)]
pub struct DeviceEnumerationTask {
    backend: Backend,
    pool: Arc<Pool>,
    discovery_tx: mpsc::Sender<crate::discovery::Event>,
    tracker: Option<Arc<Mutex<DeviceTracker>>>,
}

impl DeviceEnumerationTask {
    #[allow(dead_code)]
    pub fn new(
        backend: Backend,
        pool: Arc<Pool>,
        discovery_tx: mpsc::Sender<crate::discovery::Event>,
    ) -> Self {
        Self {
            backend,
            pool,
            discovery_tx,
            tracker: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_tracker(
        backend: Backend,
        pool: Arc<Pool>,
        discovery_tx: mpsc::Sender<crate::discovery::Event>,
        tracker: Arc<Mutex<DeviceTracker>>,
    ) -> Self {
        Self {
            backend,
            pool,
            discovery_tx,
            tracker: Some(tracker),
        }
    }

    /// Device enumeration doesn't need a tuner - it discovers tuners
    #[allow(dead_code)]
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    #[allow(dead_code)]
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<TaskContinuation> {
        debug!(backend = ?self.backend, "Starting device enumeration task");

        if shutdown.is_cancelled() {
            return Ok(TaskContinuation::Complete);
        }

        let discovered_devices = match &self.backend {
            Backend::Soapy | Backend::Usb => {
                use crate::discovery::{DeviceEnumerator, SubprocessEnumerator};

                let parent_log_file = self.pool.parent_log_file();
                let enumerator =
                    SubprocessEnumerator::new(self.backend.as_str().to_string(), parent_log_file);

                enumerator.enumerate().map_err(|e| {
                    ScannerError::ConfigurationError(format!(
                        "Enumeration failed for {:?}: {}",
                        self.backend, e
                    ))
                })?
            }
            Backend::Mock => {
                let backend = crate::hardware::Mock;
                backend.enumerate_devices()?
            }
            Backend::Unknown(name) => {
                return Err(ScannerError::ConfigurationError(format!(
                    "Unknown backend: {}",
                    name
                )));
            }
        };

        debug!(
            backend = ?self.backend,
            device_count = discovered_devices.len(),
            "Discovered devices"
        );

        let (added_devices, removed_device_ids) = if let Some(tracker) = &self.tracker {
            let mut tracker_guard = tracker.lock().unwrap();
            tracker_guard.update(discovered_devices)
        } else {
            (discovered_devices, Vec::new())
        };

        for device_id in removed_device_ids {
            debug!(device_id = ?device_id, "Sending device removal event");
            match self
                .discovery_tx
                .send(crate::discovery::Event::Removed(device_id))
            {
                Ok(_) => debug!("Device removal event sent successfully"),
                Err(e) => {
                    debug!(error = ?e, "Failed to send device removal event (channel closed?)")
                }
            }
        }

        for device_info in added_devices {
            let capabilities = Capabilities::for_device(&device_info.id);

            let result = self.pool.add_device_metadata(
                device_info.id.clone(),
                capabilities,
                self.backend.clone(),
            );

            match result {
                AddDeviceResult::Added {
                    device_id,
                    tuner_count,
                } => {
                    debug!(
                        device_id = ?device_id,
                        tuner_count = tuner_count,
                        "Added device to pool"
                    );

                    let _ = self
                        .discovery_tx
                        .send(crate::discovery::Event::Added(device_info));
                }
                AddDeviceResult::FilteredOut { device_id, reason } => {
                    debug!(
                        device_id = ?device_id,
                        reason = reason,
                        "Device filtered out by pool but still showing in TUI"
                    );

                    let _ = self
                        .discovery_tx
                        .send(crate::discovery::Event::Added(device_info));
                }
                AddDeviceResult::ShutdownMode => {
                    debug!("Pool in shutdown mode, stopping enumeration");
                    break;
                }
                AddDeviceResult::PoolBusy => {
                    debug!("Pool busy, skipping device");
                }
            }
        }

        debug!(backend = ?self.backend, "Device enumeration task completed");
        Ok(TaskContinuation::Complete)
    }

    #[allow(dead_code)]
    pub fn description(&self) -> String {
        format!("Device Enumeration: {:?}", self.backend)
    }

    #[allow(dead_code)]
    pub fn on_start(&mut self) {
        // TODO: Notify progress_reporter
    }

    #[allow(dead_code)]
    pub fn on_complete(&mut self) {
        // TODO: Notify progress_reporter
    }

    #[allow(dead_code)]
    pub fn on_error(&mut self, _error: &ScannerError) {
        // TODO: Report error
    }
}
