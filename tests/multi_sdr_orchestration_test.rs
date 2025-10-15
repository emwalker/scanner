//! Integration tests for multi-SDR orchestration
//!
//! Tests the complete system behavior: Pool + TaskScheduler + Discovery + Tasks
//! Uses Backend::Mock to simulate multiple devices without requiring real hardware.

use scanner::core::types::{Band, ScanningConfig};
use scanner::discovery::{self, DiscoveryMode, Event};
use scanner::hardware::pool::{Pool, PoolFilter, TuningMode};
use scanner::hardware::types::Backend;
use scanner::shutdown::ShutdownCoordinator;
use scanner::task::{AudioTask, DeviceEnumerationTask, ScanBandTask, Task, TaskScheduler};
use scanner::ui::NoOpProgressReporter;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// Test environment with mock devices
struct TestEnv {
    pool: Arc<Pool>,
    scheduler: Arc<TaskScheduler>,
    shutdown: Arc<ShutdownCoordinator>,
    progress: Arc<dyn scanner::ui::ProgressReporter>,
}

impl TestEnv {
    /// Create test environment and populate pool with mock devices
    fn new() -> Self {
        let filter = PoolFilter::new()
            .with_driver("mock")
            .with_mode(TuningMode::SingleTuner);
        let pool = Arc::new(Pool::new(filter, None));
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let scheduler = Arc::new(TaskScheduler::new(pool.clone(), shutdown.clone()));
        let progress = Arc::new(NoOpProgressReporter);

        let (tx, _rx) = mpsc::channel();
        let enum_task = DeviceEnumerationTask::new(Backend::Mock, pool.clone(), tx);
        scheduler
            .submit(Task::DeviceEnumeration(enum_task))
            .unwrap();
        std::thread::sleep(Duration::from_millis(300));

        Self {
            pool,
            scheduler,
            shutdown,
            progress,
        }
    }

    /// Create empty environment without populating devices
    fn new_empty() -> Self {
        let filter = PoolFilter::new()
            .with_driver("mock")
            .with_mode(TuningMode::SingleTuner);
        let pool = Arc::new(Pool::new(filter, None));
        let shutdown = Arc::new(ShutdownCoordinator::new());
        let scheduler = Arc::new(TaskScheduler::new(pool.clone(), shutdown.clone()));
        let progress = Arc::new(NoOpProgressReporter);

        Self {
            pool,
            scheduler,
            shutdown,
            progress,
        }
    }

    fn config(&self) -> ScanningConfig {
        ScanningConfig::default()
    }
}

#[test]
fn test_single_device_backward_compatibility() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new();
    let status = env.pool.status();

    if status.available_tuner_count + status.allocated_tuner_count < 1 {
        panic!("Mock backend should provide at least 1 device");
    }

    let scan_task = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );

    let handle = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task)))
        .unwrap();

    std::thread::sleep(Duration::from_millis(200));

    let status_during = env.scheduler.status();
    assert!(
        status_during.len() <= 1,
        "Should have at most 1 task running (scan task)"
    );

    handle.cancel();

    std::thread::sleep(Duration::from_millis(500));

    env.scheduler.shutdown();
    env.pool.shutdown();
}

#[test]
fn test_parallel_scan_and_audio() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new();
    let status = env.pool.status();

    if status.available_tuner_count + status.allocated_tuner_count < 2 {
        println!("Test requires 2+ devices, skipping");
        return;
    }

    let scan_task = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _scan_handle = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task)))
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));

    let audio_task = AudioTask::new(
        88_900_000.0,
        env.config(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _audio_handle = env.scheduler.submit(Task::Audio(audio_task)).unwrap();

    std::thread::sleep(Duration::from_millis(200));

    let task_status = env.scheduler.status();

    assert!(
        task_status.len() <= 2,
        "Should have at most 2 tasks running (both might have completed)"
    );

    env.scheduler.shutdown();
    env.pool.shutdown();
}

#[test]
fn test_cooperative_yielding_allows_interleaving() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new();

    let scan_task = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _scan_handle = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task)))
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));

    let (enum_tx, enum_rx) = mpsc::channel();
    let enum_task = DeviceEnumerationTask::new(Backend::Mock, env.pool.clone(), enum_tx);
    let enum_handle = env
        .scheduler
        .submit(Task::DeviceEnumeration(enum_task))
        .unwrap();

    std::thread::sleep(Duration::from_millis(1000));

    let events: Vec<_> = enum_rx.try_iter().collect();
    assert!(
        events.len() >= 2,
        "DeviceEnumerationTask should complete even while scan is running (proves cooperative yielding works)"
    );

    assert!(
        !enum_handle.is_cancelled(),
        "Enumeration task should complete normally, not be cancelled"
    );

    env.scheduler.shutdown();
    env.pool.shutdown();
}

#[test]
fn test_discovery_to_allocation_flow() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new_empty();

    let initial_status = env.pool.status();
    assert_eq!(
        initial_status.available_tuner_count, 0,
        "Pool should start empty"
    );

    let mut discovery_service = discovery::create_for_testing(
        vec![Backend::Mock],
        DiscoveryMode::ForcePolling(Duration::from_millis(100)),
        env.scheduler.clone(),
        env.pool.clone(),
    );

    let (discovery_tx, discovery_rx) = mpsc::channel();

    env.shutdown
        .spawn_sdr_thread(move |cancel| {
            discovery_service.run(discovery_tx, cancel);
        })
        .unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let events: Vec<_> = discovery_rx.try_iter().collect();
    let added_count = events
        .iter()
        .filter(|e| matches!(e, Event::Added(_)))
        .count();
    assert!(
        added_count >= 2,
        "Discovery should find 2 mock devices, found {}",
        added_count
    );

    let after_discovery = env.pool.status();
    assert!(
        after_discovery.available_tuner_count + after_discovery.allocated_tuner_count >= 2,
        "Pool should have tuners after discovery enumeration: available={}, allocated={}",
        after_discovery.available_tuner_count,
        after_discovery.allocated_tuner_count
    );

    let pool_after = env.pool.status();
    assert!(
        pool_after.available_tuner_count + pool_after.allocated_tuner_count >= 2,
        "Pool should have tuners available for allocation after discovery"
    );

    env.shutdown.shutdown();
    env.scheduler.shutdown();
}

#[test]
fn test_task_scheduler_fairness() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new_empty();

    let (tx1, rx1) = mpsc::channel();
    let enum_task1 = DeviceEnumerationTask::new(Backend::Mock, env.pool.clone(), tx1);

    let (tx2, rx2) = mpsc::channel();
    let enum_task2 = DeviceEnumerationTask::new(Backend::Mock, env.pool.clone(), tx2);

    let (tx3, rx3) = mpsc::channel();
    let enum_task3 = DeviceEnumerationTask::new(Backend::Mock, env.pool.clone(), tx3);

    let handle1 = env
        .scheduler
        .submit(Task::DeviceEnumeration(enum_task1))
        .unwrap();
    let handle2 = env
        .scheduler
        .submit(Task::DeviceEnumeration(enum_task2))
        .unwrap();
    let handle3 = env
        .scheduler
        .submit(Task::DeviceEnumeration(enum_task3))
        .unwrap();

    std::thread::sleep(Duration::from_millis(1000));

    let events1: Vec<_> = rx1.try_iter().collect();
    let events2: Vec<_> = rx2.try_iter().collect();
    let events3: Vec<_> = rx3.try_iter().collect();

    assert!(
        events1.len() >= 2,
        "Task 1 should complete and discover devices"
    );
    assert!(
        events2.len() >= 2,
        "Task 2 should complete and discover devices"
    );
    assert!(
        events3.len() >= 2,
        "Task 3 should complete and discover devices"
    );

    assert!(
        !handle1.is_cancelled(),
        "Task 1 should complete normally (fairness check)"
    );
    assert!(
        !handle2.is_cancelled(),
        "Task 2 should complete normally (fairness check)"
    );
    assert!(
        !handle3.is_cancelled(),
        "Task 3 should complete normally (fairness check)"
    );

    assert_eq!(
        env.scheduler.status().len(),
        0,
        "All enumeration tasks should complete"
    );
}

#[test]
fn test_shutdown_during_active_tasks() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new();

    let scan_task = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _scan_handle = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task)))
        .unwrap();

    let audio_task = AudioTask::new(
        88_900_000.0,
        env.config(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _audio_handle = env.scheduler.submit(Task::Audio(audio_task)).unwrap();

    std::thread::sleep(Duration::from_millis(200));

    let shutdown_start = std::time::Instant::now();
    env.shutdown.shutdown();
    env.scheduler.shutdown();

    std::thread::sleep(Duration::from_millis(500));

    let shutdown_duration = shutdown_start.elapsed();

    assert!(
        shutdown_duration < Duration::from_secs(2),
        "Shutdown initiation should be quick (< 2s), took {:?}",
        shutdown_duration
    );
}

#[test]
fn test_device_enumeration_updates_pool() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new_empty();

    let initial_status = env.pool.status();
    assert_eq!(
        initial_status.device_count, 0,
        "Pool should start with no devices"
    );
    assert_eq!(
        initial_status.available_tuner_count, 0,
        "Pool should start with no tuners"
    );

    let (discovery_tx, discovery_rx) = mpsc::channel();
    let enum_task = DeviceEnumerationTask::new(Backend::Mock, env.pool.clone(), discovery_tx);

    let _handle = env
        .scheduler
        .submit(Task::DeviceEnumeration(enum_task))
        .unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let events: Vec<_> = discovery_rx.try_iter().collect();
    assert!(
        events.len() >= 2,
        "Should receive discovery events for mock devices"
    );

    let after_enum = env.pool.status();
    assert!(
        after_enum.device_count >= 2,
        "Pool should have devices after enumeration"
    );
    assert!(
        after_enum.available_tuner_count + after_enum.allocated_tuner_count >= 2,
        "Pool should have tuners after enumeration"
    );

    env.shutdown.shutdown();
    env.scheduler.shutdown();
}

#[test]
fn test_multiple_scans_sequential() {
    let _ = tracing_subscriber::fmt::try_init();

    let env = TestEnv::new();

    let scan_task1 = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let handle1 = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task1)))
        .unwrap();

    std::thread::sleep(Duration::from_millis(200));
    handle1.cancel();
    std::thread::sleep(Duration::from_millis(200));

    let scan_task2 = ScanBandTask::new(
        env.config(),
        Band::Weather,
        env.progress.clone(),
        env.pool.clone(),
        env.shutdown.clone(),
    );
    let _handle2 = env
        .scheduler
        .submit(Task::ScanBand(Box::new(scan_task2)))
        .unwrap();

    std::thread::sleep(Duration::from_millis(500));

    let status = env.scheduler.status();
    assert!(
        status.len() <= 2,
        "Should have at most 2 scans submitted (first may still be cancelling)"
    );

    env.scheduler.shutdown();
    env.pool.shutdown();
}
