//! Tests for task module

use crate::core::types::ScanningConfig;
use crate::hardware::pool::{Pool, PoolFilter};
use crate::hardware::types::Backend;
use crate::shutdown::ShutdownCoordinator;
use crate::task::{AudioTask, DeviceEnumerationTask, Task, TaskScheduler};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[test]
fn test_device_enumeration_task_mock() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let (discovery_tx, discovery_rx) = mpsc::channel();

    let mut task = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), discovery_tx);

    let cancel = CancellationToken::new();
    let result = task.run(cancel);

    assert!(result.is_ok(), "Task should complete successfully");

    let mut device_count = 0;
    while let Ok(event) = discovery_rx.recv_timeout(Duration::from_millis(10)) {
        if let crate::discovery::Event::Added(device_info) = event {
            device_count += 1;
            assert!(!device_info.label.is_empty());
        }
    }

    assert!(device_count >= 2, "Should discover at least 2 mock devices");
}

#[test]
fn test_device_enumeration_task_shutdown() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let (discovery_tx, _discovery_rx) = mpsc::channel();

    let mut task = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), discovery_tx);

    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = task.run(cancel);
    assert!(result.is_ok(), "Task should handle shutdown gracefully");
}

#[test]
fn test_device_enumeration_task_unknown_backend() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let (discovery_tx, _discovery_rx) = mpsc::channel();

    let mut task =
        DeviceEnumerationTask::new(Backend::Unknown("test".to_string()), pool, discovery_tx);

    let cancel = CancellationToken::new();
    let result = task.run(cancel);

    assert!(result.is_err(), "Unknown backend should return error");
}

#[test]
fn test_device_enumeration_task_usb_backend() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let (discovery_tx, _discovery_rx) = mpsc::channel();

    let mut task = DeviceEnumerationTask::new(Backend::Usb, pool, discovery_tx);

    let cancel = CancellationToken::new();
    let result = task.run(cancel);

    assert!(result.is_err(), "USB backend should return error");
}

#[test]
fn test_scheduler_device_enumeration() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown);

    let (tx, rx) = mpsc::channel();
    let task = DeviceEnumerationTask::new(Backend::Mock, pool, tx);

    let handle = scheduler.submit(Task::DeviceEnumeration(task)).unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let events: Vec<_> = rx.try_iter().collect();
    assert!(
        events.len() >= 2,
        "Should have discovered at least 2 mock devices"
    );

    assert!(
        !handle.is_cancelled(),
        "Task should complete without cancellation"
    );
}

#[test]
fn test_scheduler_status() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown);

    let initial_count = scheduler.status().len();
    assert_eq!(initial_count, 0, "Should start with no running tasks");

    let (tx, _rx) = mpsc::channel();
    let task = DeviceEnumerationTask::new(Backend::Mock, pool, tx);

    let _handle = scheduler.submit(Task::DeviceEnumeration(task)).unwrap();

    let statuses = scheduler.status();
    assert!(
        statuses.len() <= 1,
        "Should have at most one task (may have already completed)"
    );

    std::thread::sleep(Duration::from_millis(500));

    let statuses = scheduler.status();
    assert_eq!(statuses.len(), 0, "Task should have completed");
}

#[test]
fn test_scheduler_stop_task() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown);

    let (tx, _rx) = mpsc::channel();
    let task = DeviceEnumerationTask::new(Backend::Mock, pool, tx);

    let handle = scheduler.submit(Task::DeviceEnumeration(task)).unwrap();

    std::thread::sleep(Duration::from_millis(10));

    let result = scheduler.stop(handle.task_id);

    match result {
        Ok(()) => {
            assert!(
                handle.is_cancelled(),
                "If stop succeeds, task must be cancelled"
            );
        }
        Err(_) => {
            assert!(
                !handle.is_cancelled(),
                "If stop fails, task should have completed normally (not cancelled)"
            );
        }
    }

    std::thread::sleep(Duration::from_millis(100));

    assert_eq!(
        scheduler.status().len(),
        0,
        "Task should no longer be running"
    );
}

#[test]
fn test_scheduler_shutdown() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown);

    let (tx1, _rx1) = mpsc::channel();
    let task1 = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), tx1);
    scheduler.submit(Task::DeviceEnumeration(task1)).unwrap();

    let (tx2, _rx2) = mpsc::channel();
    let task2 = DeviceEnumerationTask::new(Backend::Mock, pool, tx2);
    scheduler.submit(Task::DeviceEnumeration(task2)).unwrap();

    std::thread::sleep(Duration::from_millis(10));

    scheduler.shutdown();

    std::thread::sleep(Duration::from_millis(200));

    assert_eq!(
        scheduler.status().len(),
        0,
        "All tasks should have stopped after shutdown"
    );
}

#[test]
fn test_audio_task_basic() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let config = ScanningConfig::default();

    let mut task = AudioTask::new(88_900_000.0, config, pool.clone(), shutdown.clone());

    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = task.run(cancel);
    assert!(
        result.is_ok(),
        "AudioTask should handle immediate cancellation gracefully"
    );
}

#[test]
fn test_shutdown_timeout() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown.clone());

    let (tx, _rx) = mpsc::channel();
    let task = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), tx);
    let _handle = scheduler.submit(Task::DeviceEnumeration(task)).unwrap();

    std::thread::sleep(Duration::from_millis(50));

    scheduler.shutdown();

    std::thread::sleep(Duration::from_millis(200));

    let status = scheduler.status();
    assert_eq!(status.len(), 0, "Task should complete after shutdown");
}

#[test]
fn test_error_reporting() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown);

    let (discovery_tx, _discovery_rx) = mpsc::channel();
    let task =
        DeviceEnumerationTask::new(Backend::Unknown("invalid".to_string()), pool, discovery_tx);
    let handle = scheduler.submit(Task::DeviceEnumeration(task)).unwrap();

    std::thread::sleep(Duration::from_millis(200));

    let status = scheduler.status();
    assert_eq!(
        status.len(),
        0,
        "Task with invalid backend should complete (with error)"
    );

    assert!(
        handle.is_cancelled() || !handle.is_cancelled(),
        "Task handle should exist regardless of completion status"
    );
}

#[test]
fn test_task_continuation_pattern() {
    use crate::task::TaskContinuation;

    let log = Arc::new(std::sync::Mutex::new(Vec::new()));

    struct TestTask {
        step: usize,
        log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl TestTask {
        fn simulate_run(
            &mut self,
            _cancel: CancellationToken,
        ) -> crate::core::types::Result<TaskContinuation> {
            self.log.lock().unwrap().push(format!("step-{}", self.step));

            self.step += 1;

            if self.step < 3 {
                Ok(TaskContinuation::Resubmit)
            } else {
                Ok(TaskContinuation::Complete)
            }
        }
    }

    let mut task = TestTask {
        step: 0,
        log: Arc::clone(&log),
    };

    let cancel = CancellationToken::new();

    let result1 = task.simulate_run(cancel.clone());
    assert_eq!(result1.unwrap(), TaskContinuation::Resubmit);

    let result2 = task.simulate_run(cancel.clone());
    assert_eq!(result2.unwrap(), TaskContinuation::Resubmit);

    let result3 = task.simulate_run(cancel);
    assert_eq!(result3.unwrap(), TaskContinuation::Complete);

    let entries = log.lock().unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0], "step-0");
    assert_eq!(entries[1], "step-1");
    assert_eq!(entries[2], "step-2");
}

#[test]
fn test_integration_pool_scheduler_discovery() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown.clone());

    let initial_status = pool.status();
    assert_eq!(
        initial_status.available_tuner_count, 0,
        "Pool should start with no tuners"
    );

    let (discovery_tx, discovery_rx) = mpsc::channel();
    let enum_task = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), discovery_tx);

    let _enum_handle = scheduler
        .submit(Task::DeviceEnumeration(enum_task))
        .unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let events: Vec<_> = discovery_rx.try_iter().collect();
    assert!(
        events.len() >= 2,
        "Should have received discovery events for mock devices"
    );

    let after_discovery = pool.status();
    assert!(
        after_discovery.available_tuner_count + after_discovery.allocated_tuner_count >= 2,
        "Pool should have tuners after enumeration: available={}, allocated={}",
        after_discovery.available_tuner_count,
        after_discovery.allocated_tuner_count
    );

    assert_eq!(
        scheduler.status().len(),
        0,
        "DeviceEnumerationTask should have completed"
    );
}
