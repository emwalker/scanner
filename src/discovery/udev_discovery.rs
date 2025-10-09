use super::{
    common,
    enumerator::MultiEnumerator,
    polling::Polling,
    service::{Event, Service},
};
use crate::hardware;
use nix::poll::{PollFd, PollFlags, poll};
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use udev::{EventType, MonitorBuilder};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(150);

pub struct Udev {
    enumerator: MultiEnumerator,
    known_devices: HashMap<hardware::DeviceId, hardware::DeviceInfo>,
    pending_rescan: bool,
}

impl Udev {
    pub fn new(enumerator: MultiEnumerator) -> Self {
        Self {
            enumerator,
            known_devices: HashMap::new(),
            pending_rescan: false,
        }
    }

    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
        let devices = self.enumerator.enumerate();
        let mut current_devices = HashMap::new();

        for device in devices {
            current_devices.insert(device.id.clone(), device);
        }

        let (added, removed) = common::detect_changes(&self.known_devices, &current_devices);

        for device in added {
            debug!(tuner_id = ?device.id, "new device detected");
            event_tx
                .send(Event::Added(device.clone()))
                .map_err(|_| ())?;
        }

        for id in removed {
            debug!(tuner_id = ?id, "device removed");
            event_tx.send(Event::Removed(id.clone())).map_err(|_| ())?;
        }

        self.known_devices = current_devices;
        Ok(())
    }
}

impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        let socket = match MonitorBuilder::new()
            .and_then(|m| m.match_subsystem("usb"))
            .and_then(|m| m.listen())
        {
            Ok(s) => s,
            Err(e) => {
                debug!(error = %e, "failed to create udev monitor, falling back to polling");
                let enumerator = std::mem::replace(
                    &mut self.enumerator,
                    MultiEnumerator {
                        enumerators: Vec::new(),
                    },
                );
                let mut polling = Polling::new(enumerator, Duration::from_secs(3));
                return polling.run(event_tx, cancel);
            }
        };

        if self.rescan_devices(&event_tx).is_err() {
            return;
        }

        use std::os::unix::io::AsFd;
        let mut fds = [PollFd::new(socket.as_fd(), PollFlags::POLLIN)];
        let mut last_event_time = Instant::now();

        loop {
            if cancel.is_cancelled() {
                break;
            }

            match poll(&mut fds, 100u16) {
                Ok(n) if n > 0 => {
                    while let Some(event) = socket.iter().next() {
                        match event.event_type() {
                            EventType::Add | EventType::Remove => {
                                debug!(event_type = ?event.event_type(), "USB event detected");
                                self.pending_rescan = true;
                                last_event_time = Instant::now();
                            }
                            _ => {}
                        }
                    }
                }
                Ok(_) => {
                    if self.pending_rescan && last_event_time.elapsed() >= DEBOUNCE_DURATION {
                        self.pending_rescan = false;
                        if self.rescan_devices(&event_tx).is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    debug!(error = ?e, "poll error");
                    break;
                }
            }
        }
    }
}
