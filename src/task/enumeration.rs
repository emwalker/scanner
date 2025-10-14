//! Device enumeration task - discovers available SDR devices for a backend

use crate::core::types::{Result, ScannerError};
use crate::hardware::Capabilities;
use crate::hardware::backend::Backend as BackendTrait;
use crate::hardware::pool::{AddDeviceResult, Pool};
use crate::hardware::types::Backend;
use std::sync::Arc;
use std::sync::mpsc;
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
        }
    }

    /// Device enumeration doesn't need a tuner - it discovers tuners
    #[allow(dead_code)]
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    #[allow(dead_code)]
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<()> {
        debug!(backend = ?self.backend, "Starting device enumeration task");

        if shutdown.is_cancelled() {
            return Ok(());
        }

        let discovered_devices = match &self.backend {
            Backend::Soapy => {
                let backend = crate::hardware::Soapy;
                backend.enumerate_devices()?
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
            Backend::Usb => {
                return Err(ScannerError::ConfigurationError(
                    "USB is not a device backend".to_string(),
                ));
            }
        };

        debug!(
            backend = ?self.backend,
            device_count = discovered_devices.len(),
            "Discovered devices"
        );

        for device_info in discovered_devices {
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
                        "Device filtered out"
                    );
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
        Ok(())
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
