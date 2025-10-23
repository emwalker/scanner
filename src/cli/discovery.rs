use std::{
    sync::{Arc, mpsc},
    thread,
};

use tracing::{debug, info};

use crate::{
    core::types::Result,
    discovery::{self, DiscoveryMode},
    hardware::pool::Pool,
    shutdown::ShutdownCoordinator,
    ui::TuiEvent,
};

pub enum OutputMode {
    Tui(mpsc::Sender<TuiEvent>),
    Headless,
}

impl OutputMode {
    pub fn send_tuner_added(&self, device_info: &crate::hardware::DeviceInfo) {
        match self {
            OutputMode::Tui(sender) => {
                let event = TuiEvent::TunerAdded(device_info.clone());
                let _ = sender.send(event);
            }
            OutputMode::Headless => {}
        }
    }

    pub fn send_tuner_removed(&self, device_id: &crate::hardware::DeviceId) {
        match self {
            OutputMode::Tui(sender) => {
                let event = TuiEvent::TunerRemoved(device_id.clone());
                let _ = sender.send(event);
            }
            OutputMode::Headless => {}
        }
    }
}

pub struct DiscoverySetup {
    pub discovery_handle: thread::JoinHandle<()>,
    pub discovery_forwarder: thread::JoinHandle<()>,
    pub discovery_rx: mpsc::Receiver<crate::discovery::Event>,
}

/// Start discovery service that monitors for device add/remove events
pub fn start_discovery_service(
    output_mode: OutputMode,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    scheduler: Arc<crate::task::TaskScheduler>,
    pool: Arc<Pool>,
    tuner_entities: Arc<std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
    device_entities: Arc<std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
) -> Result<DiscoverySetup> {
    let backends = vec![
        crate::hardware::types::Backend::Usb,
        crate::hardware::types::Backend::Soapy,
    ];
    let mut discovery_service = discovery::create(
        backends,
        DiscoveryMode::Auto,
        scheduler,
        pool,
        tuner_entities,
        device_entities,
    );

    let (discovery_sender, discovery_receiver) = mpsc::channel();
    let (discovery_tx, discovery_rx) = mpsc::channel();

    let discovery_forwarder = {
        let shutdown = shutdown_coordinator.clone();

        thread::spawn(move || {
            info!("Discovery forwarder thread started");
            loop {
                if shutdown.is_shutdown() {
                    info!("Discovery forwarder shutting down");
                    break;
                }

                match discovery_receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(event) => {
                        debug!(event_type = ?std::mem::discriminant(&event), "Discovery forwarder received event");

                        match &event {
                            discovery::Event::Added(device_info) => {
                                debug!(device_id = ?device_info.id, "Forwarding TunerAdded event");
                                output_mode.send_tuner_added(device_info);
                                if discovery_tx.send(event.clone()).is_err() {
                                    debug!("Failed to send discovery event (channel closed)");
                                    return;
                                }
                            }
                            discovery::Event::Removed(device_id) => {
                                debug!(device_id = ?device_id, "Forwarding TunerRemoved event");
                                output_mode.send_tuner_removed(device_id);
                                if discovery_tx.send(event.clone()).is_err() {
                                    debug!("Failed to send discovery event (channel closed)");
                                    return;
                                }
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        debug!("Discovery channel disconnected");
                        break;
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
        discovery_rx,
    })
}
