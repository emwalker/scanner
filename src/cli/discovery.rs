use crate::core::types::{Result, ScannerError};
use crate::discovery::{self, DiscoveryMode};
use crate::hardware::pool::{AddDeviceResult, Pool, PoolFilter, TuningMode};
use crate::hardware::{Backend, DeviceId};
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
    device_id: &DeviceId,
    backend: &dyn Backend,
) -> Result<Arc<Pool>> {
    let filter = PoolFilter::new()
        .with_driver("sdrplay")
        .with_mode(TuningMode::SingleTuner);
    let pool = Arc::new(Pool::new(filter));

    let device_trait = backend.open_device(device_id)?;

    match pool.add_device(device_trait, backend.name().to_string()) {
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

pub fn start_discovery_service(
    tui_event_sender: mpsc::Sender<TuiEvent>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
) -> Result<DiscoverySetup> {
    let backends: Vec<Box<dyn Backend>> = vec![Box::new(crate::hardware::Soapy)];
    let mut discovery_service = discovery::create(backends, DiscoveryMode::Auto);

    let (discovery_sender, discovery_receiver) = mpsc::channel();

    let discovery_forwarder = {
        let tui_sender = tui_event_sender.clone();
        let shutdown = shutdown_coordinator.clone();

        thread::spawn(move || {
            while let Ok(event) = discovery_receiver.recv() {
                if shutdown.is_shutdown() {
                    return;
                }

                match &event {
                    discovery::Event::Added(device_info) => {
                        tracing::debug!(
                            device_id = ?device_info.id,
                            "Discovery event: device detected (not added to pool - hot-plug not yet implemented)"
                        );

                        let tui_event = TuiEvent::TunerAdded(device_info.clone());
                        if tui_sender.send(tui_event).is_err() {
                            return;
                        }
                    }
                    discovery::Event::Removed(device_id) => {
                        tracing::debug!(
                            device_id = ?device_id,
                            "Discovery event: device removed (hot-unplug not yet implemented)"
                        );

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
