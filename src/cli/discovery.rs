use crate::core::types::{Result, ScannerError};
use crate::discovery::{self, DiscoveryMode};
use crate::hardware::pool::{AddDeviceResult, Pool, PoolFilter, TuningMode};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::TuiEvent;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

pub struct DiscoverySetup {
    pub discovery_handle: thread::JoinHandle<()>,
    pub discovery_forwarder: thread::JoinHandle<()>,
}

pub fn initialize_pool_with_device(
    tuner_id: &crate::hardware::pool::TunerId,
    backend: crate::hardware::types::Backend,
    parent_log_file: Option<String>,
) -> Result<Arc<Pool>> {
    let filter = PoolFilter::new()
        .with_driver("sdrplay")
        .with_mode(TuningMode::SingleTuner);
    let pool = Pool::new(filter, parent_log_file);
    let pool = Arc::new(pool);

    let capabilities = crate::hardware::Capabilities::for_device(&tuner_id.device_id);
    let result = pool.add_device_metadata(tuner_id.device_id.clone(), capabilities, backend);

    match result {
        AddDeviceResult::Added {
            device_id,
            tuner_count,
        } => {
            tracing::debug!(
                device_id = ?device_id,
                tuner_count = tuner_count,
                "Initial device added to pool"
            );
        }
        AddDeviceResult::FilteredOut { device_id, reason } => {
            return Err(ScannerError::DeviceFilteredOut { device_id, reason });
        }
        AddDeviceResult::ShutdownMode => {
            return Err(ScannerError::PoolShutdown);
        }
        AddDeviceResult::PoolBusy => {
            return Err(ScannerError::PoolLockTimeout);
        }
    }

    Ok(pool)
}

/// Start discovery service with pre-enumerated devices
///
/// SDRplay driver limitation: Opening an SDRplay device prevents subsequent enumerations
/// from seeing any SDRplay devices, even in separate processes. To work around this:
/// 1. Enumerate all devices once at startup before opening any devices
/// 2. Send those cached devices to the TUI immediately
/// 3. Discovery service continues monitoring for USB hotplug events for other device types
///
/// See docs/research/2025-10-process-safety.md for details.
pub fn start_discovery_service(
    tui_event_sender: mpsc::Sender<TuiEvent>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    initial_devices: Vec<crate::hardware::DeviceInfo>,
    parent_log_file: Option<String>,
) -> Result<DiscoverySetup> {
    let backends = vec![];
    let mut discovery_service = discovery::create(backends, DiscoveryMode::Auto, parent_log_file);

    let (discovery_sender, discovery_receiver) = mpsc::channel();

    let discovery_forwarder = {
        let tui_sender = tui_event_sender.clone();
        let shutdown = shutdown_coordinator.clone();

        thread::spawn(move || {
            // Send pre-enumerated devices immediately (workaround for SDRplay limitation)
            for device in initial_devices {
                if shutdown.is_shutdown() {
                    return;
                }

                let tui_event = TuiEvent::TunerAdded(device);
                if tui_sender.send(tui_event).is_err() {
                    return;
                }
            }

            // Continue monitoring for hotplug events
            while let Ok(event) = discovery_receiver.recv() {
                if shutdown.is_shutdown() {
                    return;
                }

                match &event {
                    discovery::Event::Added(device_info) => {
                        let tui_event = TuiEvent::TunerAdded(device_info.clone());
                        if tui_sender.send(tui_event).is_err() {
                            return;
                        }
                    }
                    discovery::Event::Removed(device_id) => {
                        let tui_event = TuiEvent::TunerRemoved(device_id.clone());
                        if tui_sender.send(tui_event).is_err() {
                            return;
                        }
                    }
                }
            }
        })
    };

    let discovery_handle = thread::spawn(move || {
        discovery_service.run(discovery_sender, shutdown_coordinator.token());
    });

    Ok(DiscoverySetup {
        discovery_handle,
        discovery_forwarder,
    })
}
