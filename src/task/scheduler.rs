//! Task scheduler with backend serialization

use super::{Task, TaskHandle, TaskId, TaskPriority, TaskStatus, TaskType};
use crate::core::types::{Result, ScannerError};
use crate::hardware::pool::Pool;
use crate::hardware::types::Backend;
use crate::shutdown::ShutdownCoordinator;
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::warn;

/// Task with priority for queue ordering
#[allow(dead_code)]
struct PrioritizedTask {
    task: Task,
    task_id: TaskId,
    priority: TaskPriority,
    submitted_at: Instant,
}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for PrioritizedTask {}

/// Information about a running task
#[allow(dead_code)]
struct RunningTaskInfo {
    task_type: TaskType,
    description: String,
    started_at: Instant,
    cancel_token: tokio_util::sync::CancellationToken,
}

/// Schedules tasks to available tuners with backend API serialization
#[allow(dead_code)]
pub struct TaskScheduler {
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    running_tasks: Arc<DashMap<TaskId, RunningTaskInfo>>,
    backend_queues: Arc<Mutex<HashMap<Backend, VecDeque<PrioritizedTask>>>>,
    backend_semaphores: Arc<DashMap<Backend, Arc<tokio::sync::Semaphore>>>,
}

impl TaskScheduler {
    #[allow(dead_code)]
    pub fn new(pool: Arc<Pool>, shutdown_coordinator: Arc<ShutdownCoordinator>) -> Self {
        Self {
            pool,
            shutdown_coordinator,
            running_tasks: Arc::new(DashMap::new()),
            backend_queues: Arc::new(Mutex::new(HashMap::new())),
            backend_semaphores: Arc::new(DashMap::new()),
        }
    }

    /// Determine backend from task
    #[allow(dead_code)]
    fn determine_backend(&self, task: &Task) -> Backend {
        task.backend()
    }

    /// Acquire backend semaphore permit (ensures serialized API access)
    #[allow(dead_code)]
    fn acquire_backend_permit(&self, backend: &Backend) -> Arc<tokio::sync::Semaphore> {
        self.backend_semaphores
            .entry(backend.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Semaphore::new(1)))
            .clone()
    }

    /// Submit task for execution, returns handle for per-task control
    #[allow(dead_code)]
    pub fn submit(&self, mut task: Task) -> Result<TaskHandle> {
        let task_id = TaskId::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let backend = self.determine_backend(&task);
        let semaphore = self.acquire_backend_permit(&backend);

        self.running_tasks.insert(
            task_id,
            RunningTaskInfo {
                task_type: task.task_type(),
                description: task.description(),
                started_at: Instant::now(),
                cancel_token: cancel_token.clone(),
            },
        );

        let running_tasks = self.running_tasks.clone();
        let shutdown_token = cancel_token.clone();

        std::thread::spawn(move || {
            let permit = loop {
                if shutdown_token.is_cancelled() {
                    running_tasks.remove(&task_id);
                    return;
                }

                match semaphore.clone().try_acquire_owned() {
                    Ok(permit) => break permit,
                    Err(tokio::sync::TryAcquireError::NoPermits) => {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    }
                    Err(tokio::sync::TryAcquireError::Closed) => {
                        warn!(task_id = ?task_id, "Backend semaphore closed");
                        running_tasks.remove(&task_id);
                        return;
                    }
                }
            };

            if shutdown_token.is_cancelled() {
                drop(permit);
                running_tasks.remove(&task_id);
                return;
            }

            task.on_start();
            let result = task.run(shutdown_token);
            match result {
                Ok(()) => task.on_complete(),
                Err(ref e) => task.on_error(e),
            }

            drop(permit);
            running_tasks.remove(&task_id);
        });

        Ok(TaskHandle::new(task_id, cancel_token))
    }

    /// Stop a running task
    #[allow(dead_code)]
    pub fn stop(&self, task_id: TaskId) -> Result<()> {
        if let Some(info) = self.running_tasks.get(&task_id) {
            info.cancel_token.cancel();
            Ok(())
        } else {
            Err(ScannerError::Custom(format!(
                "Task {:?} not found",
                task_id
            )))
        }
    }

    /// Get status of all running tasks
    #[allow(dead_code)]
    pub fn status(&self) -> Vec<TaskStatus> {
        self.running_tasks
            .iter()
            .map(|entry| TaskStatus {
                task_id: *entry.key(),
                task_type: entry.value().task_type,
                description: entry.value().description.clone(),
                tuner_id: None,
                running_duration: entry.value().started_at.elapsed(),
            })
            .collect()
    }

    /// Shutdown all tasks
    #[allow(dead_code)]
    pub fn shutdown(&self) {
        for entry in self.running_tasks.iter() {
            entry.value().cancel_token.cancel();
        }
    }
}
