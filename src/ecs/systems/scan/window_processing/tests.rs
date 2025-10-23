//! Tests for WindowProcessingSystem serial window processing

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::WindowProcessingSystem;
use crate::{
    core::types::ScanningConfig,
    ecs::{
        EntityWorld, ScanPauseState, ScanType, TaskId, TunerAllocationQueue, WindowEntity,
        components::{
            scan::{ScanConfigComponent, ScanProgressComponent},
            window::{WindowAllocationComponent, WindowId},
        },
        system::{System, SystemContext},
        test_helpers::create_test_pool_with_entities,
    },
    hardware::{
        DeviceId,
        pool::{PoolFilter, TunerId, TuningMode},
        types::Backend,
    },
    shutdown::ShutdownCoordinator,
};

/// Test that windows are processed serially when one window is still Active
#[test]
fn test_serial_window_processing_waits_for_active() {
    let config = Arc::new(ScanningConfig::default());
    let pool_filter = PoolFilter::new().with_mode(TuningMode::SingleTuner);
    let (pool, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(pool_filter, None);
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let system = WindowProcessingSystem::new(config.clone(), pool, shutdown);

    // Create task with 3 windows
    let task_id = TaskId::new("scan_1");
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6,      // freq_min
        90.0e6,      // freq_max
        1.0e6,       // step_size
        2_000_000.0, // sample_rate
        24.0,        // gain_db
        3.0,         // duration_per_window
        3,           // scanning_windows
    );

    let mut progress = ScanProgressComponent::new(3);
    progress.state = ScanPauseState::Scanning;

    // Mark window 0 as completed so next_window_to_process returns window 1
    let window_id_0 = WindowId::new(task_id.clone(), 0);
    progress.complete_window_at(window_id_0.clone());

    // Create window entities
    let window_entities = Arc::new(std::sync::RwLock::new(EntityWorld::new()));

    {
        let mut windows = window_entities.write().unwrap();

        // Window 0: Active (still analyzing signals, but marked complete in progress)
        let mut window_0 = WindowEntity::new(window_id_0, task_id.clone(), 88.0e6);
        let tuner_id = TunerId {
            device_id: DeviceId::Driver {
                backend: Backend::Soapy,
                driver: "test".to_string(),
                serial: "123".to_string(),
            },
            channel_index: 0,
        };
        window_0.allocation.start_active(tuner_id.clone(), 5);
        windows.insert(window_0);

        // Window 1: Pending (not started yet)
        let window_id_1 = WindowId::new(task_id.clone(), 1);
        let window_1 = WindowEntity::new(window_id_1, task_id.clone(), 89.0e6);
        windows.insert(window_1);

        // Window 2: Pending
        let window_id_2 = WindowId::new(task_id.clone(), 2);
        let window_2 = WindowEntity::new(window_id_2, task_id.clone(), 90.0e6);
        windows.insert(window_2);
    }

    // Create SystemContext with window entities and allocation queue
    let allocation_queue: Arc<Mutex<TunerAllocationQueue>> = Arc::new(Mutex::new(VecDeque::new()));
    let context = SystemContext::new()
        .with_window_entities(window_entities.clone())
        .with_tuner_allocation_queue(allocation_queue.clone());

    // Call handle_no_allocation - should NOT request next window while window 0 is Active
    system.handle_no_allocation(&scan_config, &progress, &task_id, &context);

    // Verify: Window 1 should NOT have allocation requested
    {
        let windows = window_entities.read().unwrap();
        let window_id_1 = WindowId::new(task_id.clone(), 1);
        let window_1 = windows.get(&window_id_1).unwrap();

        assert!(
            !window_1.allocation.is_requested(),
            "Window 1 should not be requested while Window 0 is Active, but allocation state is: \
             {:?}",
            window_1.allocation
        );
    }
}

/// Test that next window IS requested when previous window is Complete
#[test]
fn test_serial_window_processing_proceeds_when_complete() {
    let config = Arc::new(ScanningConfig::default());
    let pool_filter = PoolFilter::new().with_mode(TuningMode::SingleTuner);
    let (pool, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(pool_filter, None);
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let system = WindowProcessingSystem::new(config.clone(), pool, shutdown);

    // Create task with 2 windows
    let task_id = TaskId::new("scan_1");
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6,
        89.0e6,
        1.0e6,
        2_000_000.0,
        24.0,
        3.0,
        2,
    );

    let mut progress = ScanProgressComponent::new(2);
    progress.state = ScanPauseState::Scanning;

    // Mark window 0 as completed
    let window_id_0 = WindowId::new(task_id.clone(), 0);
    progress.complete_window_at(window_id_0.clone());

    // Create window entities
    let window_entities = Arc::new(std::sync::RwLock::new(EntityWorld::new()));

    {
        let mut windows = window_entities.write().unwrap();

        // Window 0: Complete (ready for deallocation, tuner released)
        let mut window_0 = WindowEntity::new(window_id_0, task_id.clone(), 88.0e6);
        let tuner_id = TunerId {
            device_id: DeviceId::Driver {
                backend: Backend::Soapy,
                driver: "test".to_string(),
                serial: "123".to_string(),
            },
            channel_index: 0,
        };
        window_0.allocation = WindowAllocationComponent::Complete { tuner_id };
        windows.insert(window_0);

        // Window 1: Pending (ready to start)
        let window_id_1 = WindowId::new(task_id.clone(), 1);
        let window_1 = WindowEntity::new(window_id_1, task_id.clone(), 89.0e6);
        windows.insert(window_1);
    }

    // Create SystemContext with window entities and allocation queue
    let allocation_queue: Arc<Mutex<TunerAllocationQueue>> = Arc::new(Mutex::new(VecDeque::new()));
    let context = SystemContext::new()
        .with_window_entities(window_entities.clone())
        .with_tuner_allocation_queue(allocation_queue.clone());

    // Call handle_no_allocation - SHOULD request next window since window 0 is Complete
    system.handle_no_allocation(&scan_config, &progress, &task_id, &context);

    // Verify: Window 1 SHOULD have allocation requested
    {
        let windows = window_entities.read().unwrap();
        let window_id_1 = WindowId::new(task_id.clone(), 1);
        let window_1 = windows.get(&window_id_1).unwrap();

        assert!(
            window_1.allocation.is_requested(),
            "Window 1 should be requested when Window 0 is Complete"
        );
    }
}

#[test]
fn test_system_respects_global_pause() {
    use std::sync::{Mutex, RwLock};

    use crate::ecs::{
        GlobalPauseState, ScanTaskData, TaskEntity,
        system::{System, SystemContext},
    };

    let config = Arc::new(ScanningConfig::default());
    let pool_filter = PoolFilter::new().with_mode(TuningMode::SingleTuner);
    let (pool, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(pool_filter, None);
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let mut system = WindowProcessingSystem::new(config.clone(), pool, shutdown);
    system.enable();

    // Create task with scanning state
    let task_id = TaskId::new("scan_1");
    let mut task =
        TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 2);
    let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
    progress.state = ScanPauseState::Scanning;

    let task_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));
    {
        let mut tasks = task_entities.write().unwrap();
        tasks.insert(task);
    }

    let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
    {
        let mut windows = window_entities.write().unwrap();
        // Window 0: Pending (ready to be processed)
        let window_id_0 = WindowId::new(task_id.clone(), 0);
        let window_0 = WindowEntity::new(window_id_0, task_id.clone(), 88.0e6);
        windows.insert(window_0);
    }

    // Create global pause resource in Paused state
    let global_pause = Arc::new(Mutex::new(GlobalPauseState::Paused {
        had_active_scans: true,
        playing_stations: vec![],
    }));

    let mut context = SystemContext::new()
        .with_task_entities(task_entities)
        .with_window_entities(window_entities.clone())
        .with_global_pause_resource(global_pause);

    // Run the system - should return early due to global pause
    // WITHOUT the fix, this would process the window and request allocation
    system.run(&mut context).unwrap();

    // Verify: Window should NOT have been processed (no allocation requested)
    {
        let windows = window_entities.read().unwrap();
        let window_id_0 = WindowId::new(task_id.clone(), 0);
        let window_0 = windows.get(&window_id_0).unwrap();

        assert!(
            !window_0.allocation.is_requested(),
            "BUG: Window allocation should NOT be requested during global pause. This is part of \
             the pause/resume bug - WindowProcessingSystem should not spawn new work while paused."
        );
    }
}

/// Integration test: WindowProcessingSystem should NOT process finished workers
///
/// This test demonstrates the bug where WindowProcessingSystem interferes with
/// WindowWorkerCompletionSystem by consuming finished worker tasks.
///
/// The proper ECS pattern:
/// - WindowWorkerCompletionSystem is responsible for processing finished workers
/// - WindowProcessingSystem should only handle scan coordination (window allocation, state
///   transitions)
///
/// Without the fix, this test FAILS because:
/// 1. WindowProcessingSystem finds the finished worker via `task.take()`
/// 2. WindowProcessingSystem processes it and removes it from the window
/// 3. WindowWorkerCompletionSystem then finds windows_with_tasks=0
///
/// With the fix, this test PASSES because:
/// 1. WindowProcessingSystem skips finished workers
/// 2. WindowWorkerCompletionSystem is the only system that processes them
#[test]
fn test_window_processing_does_not_consume_finished_workers() {
    use std::{sync::RwLock, time::Instant};

    use tokio_util::sync::CancellationToken;

    let config = Arc::new(ScanningConfig::default());
    let pool_filter = PoolFilter::new().with_mode(TuningMode::SingleTuner);
    let (pool, tuner_entities, _device_entities) =
        create_test_pool_with_entities(pool_filter, None);
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let mut processing_system =
        WindowProcessingSystem::new(config.clone(), pool.clone(), shutdown.clone());
    processing_system.enable();

    // Create a scan task
    let task_id = TaskId::new("scan_1");
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6,
        90.0e6,
        1.0e6,
        2_000_000.0,
        24.0,
        3.0,
        3,
    );

    let mut task = crate::ecs::TaskEntity::new_scan_with_defaults(
        task_id.clone(),
        crate::ecs::ScanTaskData::Placeholder,
        3,
    );

    let crate::ecs::TaskComponents::Scan {
        config: task_config,
        progress,
        ..
    } = &mut task.components;
    *task_config = scan_config.clone();
    progress.state = ScanPauseState::Scanning;

    let task_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));
    task_entities.write().unwrap().insert(task);

    // Create window with a FINISHED worker
    let window_id = WindowId::new(task_id.clone(), 0);
    let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);

    // Simulate a finished worker with NoSignals outcome (simpler, no Segment needed)
    let finished_worker_handle = std::thread::spawn(|| {
        Ok(crate::ecs::components::scan::WindowWorkerResult {
            window_index: 0,
            outcome: crate::ecs::components::scan::WindowWorkerOutcome::NoSignals {
                center_freq: 88.0e6,
                reason: "test - no signals detected".to_string(),
            },
            completed_at: Instant::now(),
        })
    });

    // Wait for worker to finish
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert!(
        finished_worker_handle.is_finished(),
        "Worker should be finished before test"
    );

    window.task = Some(crate::ecs::components::scan::WindowWorkerComponent {
        window_index: 0,
        task_handle: finished_worker_handle,
        cancellation_token: CancellationToken::new(),
        started_at: Instant::now(),
        cancelling: false,
    });

    let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
    window_entities.write().unwrap().insert(window);

    let allocation_queue = Arc::new(Mutex::new(VecDeque::new()));

    let mut context = SystemContext::new()
        .with_task_entities(task_entities)
        .with_window_entities(window_entities.clone())
        .with_tuner_entities(tuner_entities)
        .with_tuner_allocation_queue(allocation_queue);

    // Run WindowProcessingSystem
    processing_system.run(&mut context).unwrap();

    // ASSERTION: The worker task should STILL be present in the window
    // WindowProcessingSystem should NOT have consumed it
    {
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).unwrap();

        assert!(
            window.task.is_some(),
            "WindowProcessingSystem should NOT consume finished workers - that's \
             WindowWorkerCompletionSystem's job!"
        );

        // Verify the worker is still finished (not a different worker)
        assert!(
            window.task.as_ref().unwrap().task_handle.is_finished(),
            "The worker task should still be the finished one"
        );
    }
}
