//! Task scheduler with backend serialization

use super::{Task, TaskContinuation, TaskHandle, TaskId, TaskPriority, TaskStatus, TaskType};
use crate::core::types::{Result, ScannerError};
use crate::hardware::pool::Pool;
use crate::hardware::types::Backend;
use crate::shutdown::ShutdownCoordinator;
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, warn};

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
    completion_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<TaskId>>>>,
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
            completion_tx: Arc::new(Mutex::new(None)),
        }
    }

    #[allow(dead_code)]
    pub fn set_completion_channel(&self, tx: std::sync::mpsc::Sender<TaskId>) {
        *self.completion_tx.lock().unwrap() = Some(tx);
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
        let completion_tx = self.completion_tx.clone();

        std::thread::spawn(move || {
            task.on_start();

            // Task execution loop - allows cooperative yielding
            // Create a Tokio runtime for async operations in this thread
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime");

            let mut permit = match runtime.block_on(semaphore.clone().acquire_owned()) {
                Ok(permit) => permit,
                Err(_) => {
                    warn!(task_id = ?task_id, "Backend semaphore closed");
                    running_tasks.remove(&task_id);
                    return;
                }
            };

            loop {
                if shutdown_token.is_cancelled() {
                    drop(permit);
                    running_tasks.remove(&task_id);
                    return;
                }

                let result = task.run(shutdown_token.clone());

                match result {
                    Ok(TaskContinuation::Complete) => {
                        task.on_complete();
                        if let Some(tx) = completion_tx.lock().unwrap().as_ref() {
                            let _ = tx.send(task_id);
                        }
                        break;
                    }
                    Ok(TaskContinuation::Resubmit) => {
                        // Task has more work but yields to allow fairness
                        // Drop permit explicitly to release backend immediately
                        debug!(task_id = ?task_id, "Task yielding backend semaphore");
                        drop(permit);

                        // Check for cancellation before reacquiring
                        if shutdown_token.is_cancelled() {
                            running_tasks.remove(&task_id);
                            return;
                        }

                        // Reacquire permit with FIFO fairness
                        permit = match runtime.block_on(semaphore.clone().acquire_owned()) {
                            Ok(permit) => {
                                debug!(task_id = ?task_id, "Task reacquired backend semaphore");
                                permit
                            }
                            Err(_) => {
                                warn!(task_id = ?task_id, "Backend semaphore closed during reacquisition");
                                running_tasks.remove(&task_id);
                                return;
                            }
                        };

                        // Continue loop - will call run() again
                        continue;
                    }
                    Ok(TaskContinuation::ResubmitAfter(delay)) => {
                        // Task has more work but needs to delay before retrying
                        // Drop permit to release backend, sleep, then reacquire
                        debug!(task_id = ?task_id, delay_ms = delay.as_millis(), "Task yielding backend semaphore with delay");
                        drop(permit);

                        // Sleep AFTER releasing semaphore to allow other tasks to run
                        std::thread::sleep(delay);

                        // Check for cancellation before reacquiring
                        if shutdown_token.is_cancelled() {
                            running_tasks.remove(&task_id);
                            return;
                        }

                        // Reacquire permit with FIFO fairness
                        permit = match runtime.block_on(semaphore.clone().acquire_owned()) {
                            Ok(permit) => {
                                debug!(task_id = ?task_id, "Task reacquired backend semaphore after delay");
                                permit
                            }
                            Err(_) => {
                                warn!(task_id = ?task_id, "Backend semaphore closed during reacquisition");
                                running_tasks.remove(&task_id);
                                return;
                            }
                        };

                        // Continue loop - will call run() again
                        continue;
                    }
                    Err(ref e) => {
                        task.on_error(e);
                        if let Some(tx) = completion_tx.lock().unwrap().as_ref() {
                            let _ = tx.send(task_id);
                        }
                        break;
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::types::MockTaskTrait;
    use std::sync::mpsc;

    struct MockTask {
        id: usize,
        yields: usize,
        yield_count: usize,
        order_tx: mpsc::Sender<(usize, &'static str)>,
    }

    impl MockTask {
        fn new(id: usize, yields: usize, order_tx: mpsc::Sender<(usize, &'static str)>) -> Self {
            Self {
                id,
                yields,
                yield_count: 0,
                order_tx,
            }
        }
    }

    impl MockTaskTrait for MockTask {
        fn backend(&self) -> Backend {
            Backend::Mock
        }

        fn run(
            &mut self,
            _shutdown: tokio_util::sync::CancellationToken,
        ) -> Result<TaskContinuation> {
            self.order_tx.send((self.id, "run")).unwrap();

            // Sleep on first run to allow other tasks to queue on semaphore
            // Use a longer sleep to ensure other tasks have time to start their threads,
            // create their Tokio runtimes, and reach the acquire_owned() call
            if self.yield_count == 0 && self.yields > 0 {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }

            if self.yield_count < self.yields {
                self.yield_count += 1;
                Ok(TaskContinuation::Resubmit)
            } else {
                Ok(TaskContinuation::Complete)
            }
        }

        fn description(&self) -> String {
            format!("MockTask {}", self.id)
        }

        fn on_start(&mut self) {
            self.order_tx.send((self.id, "start")).unwrap();
        }

        fn on_complete(&mut self) {
            self.order_tx.send((self.id, "complete")).unwrap();
        }

        fn on_error(&mut self, _error: &ScannerError) {}
    }

    /// Regression test for SDRplay hot-plug removal issue.
    ///
    /// This test verifies that when a task yields the backend semaphore (via TaskContinuation::Resubmit),
    /// tasks that are already queued waiting for the semaphore get priority over the yielding task
    /// reacquiring. This FIFO ordering is critical for device enumeration to run promptly when
    /// a scan task yields between windows.
    ///
    /// Before the fix (using try_acquire_owned with busy-wait):
    /// - Task 1 would immediately reacquire: [1, 1, ...]
    /// - Enumeration tasks couldn't get semaphore access during scan
    /// - Device removal events wouldn't reach TUI until scan completed
    ///
    /// After the fix (using async acquire_owned with FIFO):
    /// - Queued tasks run before task 1 reacquires: [1, 2, ...]  or [1, 3, ...]
    /// - Enumeration can run between scan windows
    /// - Device removal events reach TUI promptly
    #[test]
    fn test_fifo_semaphore_ordering() {
        use std::time::Duration;

        // Create a semaphore with 1 permit for the Mock backend
        let pool = Arc::new(Pool::new_unfiltered());
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
        let scheduler = TaskScheduler::new(pool.clone(), shutdown_coordinator);

        let (order_tx, order_rx) = mpsc::channel();

        // Task 1: Acquires, yields once
        let task1 = MockTask::new(1, 1, order_tx.clone());

        // Submit task 1 first
        scheduler
            .submit(Task::Mock(Box::new(task1)))
            .expect("Failed to submit task 1");

        // Wait for task 1 to start running
        loop {
            match order_rx.recv_timeout(Duration::from_millis(100)) {
                Ok((1, "run")) => {
                    // Task 1 is running and will sleep for 200ms
                    // Submit tasks 2 & 3 now while task 1 holds the semaphore
                    std::thread::sleep(Duration::from_millis(10));

                    let task2 = MockTask::new(2, 0, order_tx.clone());
                    let task3 = MockTask::new(3, 0, order_tx.clone());

                    scheduler
                        .submit(Task::Mock(Box::new(task2)))
                        .expect("Failed to submit task 2");
                    scheduler
                        .submit(Task::Mock(Box::new(task3)))
                        .expect("Failed to submit task 3");
                    break;
                }
                Ok(_) => continue,
                Err(_) => panic!("Task 1 never started running"),
            }
        }

        // Wait longer to ensure tasks 2 & 3 have started their threads and are blocked on acquire_owned()
        // Task 1 is sleeping for 200ms, we've already waited 10ms, so wait another 150ms
        // This gives tasks 2 & 3 a total of 150ms to reach acquire_owned() while task 1 still holds the permit
        std::thread::sleep(Duration::from_millis(150));

        // Collect all run events for 500ms
        let start = std::time::Instant::now();
        let mut acquisition_order = vec![1]; // We already saw task 1's first run
        while start.elapsed() < Duration::from_millis(500) {
            match order_rx.recv_timeout(Duration::from_millis(10)) {
                Ok((id, "run")) => acquisition_order.push(id),
                Ok(_) => {}
                Err(_) => {}
            }
        }

        eprintln!("Acquisition order: {:?}", acquisition_order);

        // With FIFO: task 1 runs, yields, then tasks 2 & 3 (queued) run before task 1 reacquires
        // Expected: [1, 2, 3, 1] or [1, 3, 2, 1] (tasks 2 & 3 both before task 1's reacquire)
        //
        // Without FIFO (bug): task 1 reacquires immediately before queued tasks
        // Would see: [1, 1, ...]

        assert!(
            acquisition_order.len() >= 3,
            "Should see at least 3 acquisitions, got {:?}",
            acquisition_order
        );

        assert_eq!(acquisition_order[0], 1, "Task 1 should acquire first");

        // The critical test: task 1 should NOT run twice consecutively
        // If FIFO works, tasks 2 or 3 should run before task 1's second run
        assert!(
            acquisition_order[1] != 1,
            "FIFO failure: Task 1 reacquired immediately after yielding (position 1). \
             With FIFO, queued tasks 2 or 3 should run first. Order: {:?}",
            acquisition_order
        );

        scheduler.shutdown();
    }
}
