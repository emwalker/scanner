use crate::core::types::Result;
use crate::discovery::{self, DiscoveryMode};
use crate::hardware::pool::Pool;
use crate::shutdown::ShutdownCoordinator;
use crate::ui::TuiEvent;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use tracing::debug;

pub struct DiscoverySetup {
    pub discovery_handle: thread::JoinHandle<()>,
    pub discovery_forwarder: thread::JoinHandle<()>,
}

/// Start discovery service that monitors for device add/remove events
pub fn start_discovery_service(
    tui_event_sender: mpsc::Sender<TuiEvent>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    scheduler: Arc<crate::task::TaskScheduler>,
    pool: Arc<Pool>,
) -> Result<DiscoverySetup> {
    let backends = vec![
        crate::hardware::types::Backend::Usb,
        crate::hardware::types::Backend::Soapy,
    ];
    let mut discovery_service = discovery::create(backends, DiscoveryMode::Auto, scheduler, pool);

    let (discovery_sender, discovery_receiver) = mpsc::channel();

    let discovery_forwarder = {
        let tui_sender = tui_event_sender.clone();
        let shutdown = shutdown_coordinator.clone();

        thread::spawn(move || {
            debug!("Discovery forwarder thread started");
            loop {
                if shutdown.is_shutdown() {
                    debug!("Discovery forwarder shutting down");
                    break;
                }

                match discovery_receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(event) => {
                        debug!(event_type = ?std::mem::discriminant(&event), "Discovery forwarder received event");

                        match &event {
                            discovery::Event::Added(device_info) => {
                                debug!(device_id = ?device_info.id, "Forwarding TunerAdded event to TUI");
                                let tui_event = TuiEvent::TunerAdded(device_info.clone());
                                if tui_sender.send(tui_event).is_err() {
                                    debug!("Failed to send TunerAdded to TUI (channel closed)");
                                    return;
                                }
                            }
                            discovery::Event::Removed(device_id) => {
                                debug!(device_id = ?device_id, "Forwarding TunerRemoved event to TUI");
                                let tui_event = TuiEvent::TunerRemoved(device_id.clone());
                                if tui_sender.send(tui_event).is_err() {
                                    debug!("Failed to send TunerRemoved to TUI (channel closed)");
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
    })
}
