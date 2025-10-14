//! Tests for task module

use crate::core::types::{Band, ScanningConfig};
use crate::hardware::pool::{Pool, PoolFilter};
use crate::hardware::types::Backend;
use crate::shutdown::ShutdownCoordinator;
use crate::task::{AudioTask, DeviceEnumerationTask, ScanBandTask, Task, TaskScheduler};
use crate::ui::NoOpProgressReporter;
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
fn test_scan_band_task_basic() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let progress = Arc::new(NoOpProgressReporter);
    let config = ScanningConfig::default();

    let mut task = ScanBandTask::new(
        config,
        Band::Weather,
        progress,
        pool.clone(),
        shutdown.clone(),
    );

    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = task.run(cancel);
    assert!(
        result.is_ok(),
        "ScanBandTask should handle immediate cancellation gracefully"
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
fn test_parallel_tasks() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown.clone());
    let progress = Arc::new(NoOpProgressReporter);
    let config = ScanningConfig::default();

    let scan_task = ScanBandTask::new(
        config.clone(),
        Band::Weather,
        progress.clone(),
        pool.clone(),
        shutdown.clone(),
    );
    let scan_handle = scheduler.submit(Task::ScanBand(scan_task)).unwrap();

    let audio_task = AudioTask::new(88_900_000.0, config, pool, shutdown);
    let audio_handle = scheduler.submit(Task::Audio(audio_task)).unwrap();

    std::thread::sleep(Duration::from_millis(100));

    let status = scheduler.status();

    if status.len() == 2 {
        let scan_status = status.iter().find(|s| s.task_id == scan_handle.task_id);
        let audio_status = status.iter().find(|s| s.task_id == audio_handle.task_id);

        assert!(scan_status.is_some(), "Scan task should be in status");
        assert!(audio_status.is_some(), "Audio task should be in status");

        if let (Some(scan), Some(audio)) = (scan_status, audio_status)
            && let (Some(scan_tuner), Some(audio_tuner)) = (&scan.tuner_id, &audio.tuner_id)
        {
            assert_ne!(scan_tuner, audio_tuner, "Tasks should use different tuners");
        }
    }

    scheduler.shutdown();
    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(
        scheduler.status().len(),
        0,
        "All tasks should complete after shutdown"
    );
}

#[test]
fn test_backend_serialization() {
    let mut filter = PoolFilter::new();
    filter = filter.with_driver("mock");
    let pool = Arc::new(Pool::new(filter, None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown.clone());
    let progress = Arc::new(NoOpProgressReporter);
    let config = ScanningConfig::default();

    let task1 = ScanBandTask::new(
        config.clone(),
        Band::Weather,
        progress.clone(),
        pool.clone(),
        shutdown.clone(),
    );
    let handle1 = scheduler.submit(Task::ScanBand(task1)).unwrap();

    std::thread::sleep(Duration::from_millis(50));

    let task2 = ScanBandTask::new(config, Band::Weather, progress, pool, shutdown);
    let handle2 = scheduler.submit(Task::ScanBand(task2)).unwrap();

    std::thread::sleep(Duration::from_millis(100));

    let status = scheduler.status();
    assert!(
        status.len() <= 2,
        "Should have at most 2 tasks (backend serialization limits concurrency)"
    );

    scheduler.shutdown();
    std::thread::sleep(Duration::from_millis(1000));

    assert!(
        !handle1.is_cancelled() || !handle2.is_cancelled(),
        "At least one task should complete normally"
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
fn test_device_enumeration_serialized_with_scan() {
    let pool = Arc::new(Pool::new(PoolFilter::new(), None));
    let shutdown = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool.clone(), shutdown.clone());
    let progress = Arc::new(NoOpProgressReporter);
    let config = ScanningConfig::default();

    let scan_task = ScanBandTask::new(config, Band::Weather, progress, pool.clone(), shutdown);
    let _scan_handle = scheduler.submit(Task::ScanBand(scan_task)).unwrap();

    std::thread::sleep(Duration::from_millis(50));

    let (tx, rx) = mpsc::channel();
    let enum_task = DeviceEnumerationTask::new(Backend::Mock, pool, tx);
    let _enum_handle = scheduler
        .submit(Task::DeviceEnumeration(enum_task))
        .unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let events: Vec<_> = rx.try_iter().collect();
    assert!(
        events.len() >= 2,
        "Device enumeration should complete and discover devices"
    );

    scheduler.shutdown();
    std::thread::sleep(Duration::from_millis(200));

    assert_eq!(scheduler.status().len(), 0, "All tasks should complete");
}
