use std::{
    collections::HashMap,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

use nix::poll::{PollFd, PollFlags, poll};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use udev::{EventType, MonitorBuilder};

use super::{
    polling::Polling,
    service::{Event, Service},
    tracker::DeviceTracker,
};
use crate::{
    hardware::{pool::Pool, types::Backend},
    task::{DeviceEnumerationTask, Task, TaskScheduler},
};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(150);

pub struct Udev {
    scheduler: Arc<TaskScheduler>,
    pool: Arc<Pool>,
    backends: Vec<Backend>,
    pending_rescan: bool,
    trackers: HashMap<Backend, Arc<Mutex<DeviceTracker>>>,
    tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
    device_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
}

impl Udev {
    pub fn new(
        scheduler: Arc<TaskScheduler>,
        pool: Arc<Pool>,
        backends: Vec<Backend>,
        tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
        device_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
    ) -> Self {
        let trackers = backends
            .iter()
            .map(|backend| (backend.clone(), Arc::new(Mutex::new(DeviceTracker::new()))))
            .collect();

        Self {
            scheduler,
            pool,
            backends,
            pending_rescan: false,
            trackers,
            tuner_entities,
            device_entities,
        }
    }

    fn submit_enumeration_tasks(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
        for backend in &self.backends {
            debug!(backend = ?backend, "Submitting device enumeration task");

            let tracker = self
                .trackers
                .get(backend)
                .cloned()
                .expect("Tracker should exist for backend");

            let task = DeviceEnumerationTask::with_shared_entities(
                backend.clone(),
                self.pool.clone(),
                event_tx.clone(),
                Some(tracker),
                self.tuner_entities.clone(),
                self.device_entities.clone(),
            );

            self.scheduler
                .submit(Task::DeviceEnumeration(task))
                .map_err(|e| {
                    debug!(error = ?e, "Failed to submit enumeration task");
                })?;
        }

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
                let mut polling = Polling::new(
                    self.scheduler.clone(),
                    self.pool.clone(),
                    self.backends.clone(),
                    Duration::from_secs(3),
                    self.tuner_entities.clone(),
                    self.device_entities.clone(),
                );
                return polling.run(event_tx, cancel);
            }
        };

        if self.submit_enumeration_tasks(&event_tx).is_err() {
            return;
        }

        debug!("Udev discovery service started, monitoring USB subsystem");

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
                                debug!(
                                    event_type = ?event.event_type(),
                                    devpath = ?event.devpath(),
                                    devtype = ?event.devtype(),
                                    subsystem = ?event.subsystem(),
                                    sysname = ?event.sysname(),
                                    syspath = ?event.syspath(),
                                    "USB event detected, will trigger re-enumeration"
                                );
                                self.pending_rescan = true;
                                last_event_time = Instant::now();
                            }
                            _ => {}
                        }
                    }
                }
                Ok(_) => {
                    if self.pending_rescan && last_event_time.elapsed() >= DEBOUNCE_DURATION {
                        debug!("USB event debounce period elapsed, triggering re-enumeration");
                        self.pending_rescan = false;
                        if self.submit_enumeration_tasks(&event_tx).is_err() {
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
