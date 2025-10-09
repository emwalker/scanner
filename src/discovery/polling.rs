use super::{
    common,
    enumerator::MultiEnumerator,
    service::{Event, Service},
};
use crate::hardware;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::debug;

pub struct Polling {
    enumerator: MultiEnumerator,
    known_devices: HashMap<hardware::DeviceId, hardware::DeviceInfo>,
    poll_interval: Duration,
}

impl Polling {
    pub fn new(enumerator: MultiEnumerator, poll_interval: Duration) -> Self {
        Self {
            enumerator,
            known_devices: HashMap::new(),
            poll_interval,
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

impl Service for Polling {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        if self.rescan_devices(&event_tx).is_err() {
            return;
        }

        loop {
            if cancel.is_cancelled() {
                break;
            }

            // Sleep in small chunks to ensure responsive shutdown
            let sleep_chunk = Duration::from_millis(100);
            let mut remaining = self.poll_interval;
            while remaining > Duration::ZERO {
                if cancel.is_cancelled() {
                    return;
                }
                let to_sleep = remaining.min(sleep_chunk);
                std::thread::sleep(to_sleep);
                remaining = remaining.saturating_sub(to_sleep);
            }

            if cancel.is_cancelled() {
                break;
            }

            if self.rescan_devices(&event_tx).is_err() {
                break;
            }
        }
    }
}
