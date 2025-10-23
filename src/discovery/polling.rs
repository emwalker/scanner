use std::{
    collections::HashMap,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

use tokio_util::sync::CancellationToken;
use tracing::debug;

use super::{
    service::{Event, Service},
    tracker::DeviceTracker,
};
use crate::{
    hardware::{pool::Pool, types::Backend},
    task::{DeviceEnumerationTask, Task, TaskScheduler},
};

pub struct Polling {
    scheduler: Arc<TaskScheduler>,
    pool: Arc<Pool>,
    backends: Vec<Backend>,
    poll_interval: Duration,
    trackers: HashMap<Backend, Arc<Mutex<DeviceTracker>>>,
    tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
    device_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
}

impl Polling {
    pub fn new(
        scheduler: Arc<TaskScheduler>,
        pool: Arc<Pool>,
        backends: Vec<Backend>,
        poll_interval: Duration,
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
            poll_interval,
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

impl Service for Polling {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        if self.submit_enumeration_tasks(&event_tx).is_err() {
            return;
        }

        loop {
            if cancel.is_cancelled() {
                break;
            }

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

            if self.submit_enumeration_tasks(&event_tx).is_err() {
                break;
            }
        }
    }
}
