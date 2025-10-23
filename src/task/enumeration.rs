//! Device enumeration task - discovers available SDR devices for a backend

use std::sync::{Arc, Mutex, mpsc};

use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{
    core::types::{Result, ScannerError},
    discovery::tracker::DeviceTracker,
    hardware::{Capabilities, backend::Backend as BackendTrait, pool::Pool, types::Backend},
    task::TaskContinuation,
};

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
    shared_tuner_entities: Option<Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>>,
    shared_device_entities: Option<Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>>,
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
            shared_tuner_entities: None,
            shared_device_entities: None,
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
            shared_tuner_entities: None,
            shared_device_entities: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_shared_entities(
        backend: Backend,
        pool: Arc<Pool>,
        discovery_tx: mpsc::Sender<crate::discovery::Event>,
        tracker: Option<Arc<Mutex<DeviceTracker>>>,
        shared_tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
        shared_device_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
    ) -> Self {
        Self {
            backend,
            pool,
            discovery_tx,
            tracker,
            shared_tuner_entities: Some(shared_tuner_entities),
            shared_device_entities: Some(shared_device_entities),
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

            // Write directly to shared EntityWorlds (Pool no longer creates entities)
            let (Some(shared_tuner_entities), Some(shared_device_entities)) =
                (&self.shared_tuner_entities, &self.shared_device_entities)
            else {
                debug!("Shared entities not available, skipping device addition");
                continue;
            };

            match (
                shared_tuner_entities.try_lock(),
                shared_device_entities.try_lock(),
            ) {
                (Ok(mut tuners), Ok(mut hardware)) => {
                    // Create hardware entity
                    let device_entity = crate::ecs::DeviceEntity::new_metadata_only(
                        device_info.id.clone(),
                        device_info.label.clone(),
                        capabilities.clone(),
                        self.backend.clone(),
                    );
                    hardware.insert(device_entity);

                    // Create tuner entities
                    for tuner_info in &device_info.tuners {
                        let tuner_entity = crate::ecs::TunerEntity::new(
                            device_info.id.clone(),
                            tuner_info.id.channel_index,
                            capabilities.clone(),
                            self.backend.clone(),
                            tuner_info.label.clone(),
                            tuner_info.antenna.clone(),
                            tuner_info.mode.clone(),
                        );
                        tuners.insert(tuner_entity);
                    }

                    debug!(
                        device_id = ?device_info.id,
                        num_tuners = device_info.tuners.len(),
                        "Created entities directly in shared EntityWorlds"
                    );

                    // Send discovery event
                    let _ = self
                        .discovery_tx
                        .send(crate::discovery::Event::Added(device_info));
                }
                (Err(_), _) | (_, Err(_)) => {
                    debug!(
                        device_id = ?device_info.id,
                        "Could not lock shared entities for device addition"
                    );
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
