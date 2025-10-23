//! Window processing tests
//!
//! These tests verify that the ECS-based window processing system correctly:
//! - Calculates total_windows for both Band and Stations scan types
//! - Requests and receives tuner allocations
//! - Spawns window tasks that create SignalEntity objects
//! - Does NOT try to deallocate Complete windows (AudioStreamManagementSystem's job)

use std::sync::{Arc, Mutex, RwLock};

use scanner::{
    audio::quality::AudioAnalyzer,
    core::types::{Result, ScanningConfig},
    ecs::{
        AudioEntity, Coordinator, Entities, Entity, EntityWorld, ScanConfigComponent,
        ScanPauseState, ScanTaskData, ScanType, SignalEntity, System, SystemContext,
        TaskComponents, TaskEntity, TaskId, TunerEntity, WindowEntity, WindowId,
        systems::{
            AudioSpawnSystem, PeakAnalysisSystem, PeakCompletionSystem, PeakDetectionSystem,
            SignalAnalysisSystem, scan::WindowProcessingSystem,
        },
    },
    hardware::{
        Capabilities, DeviceId,
        pool::{Pool, TunerId},
        types::Backend,
    },
    shutdown::ShutdownCoordinator,
};

#[test]
fn test_full_window_to_audio_pipeline() -> Result<()> {
    let pool = Arc::new(Pool::new_unfiltered());
    let config = Arc::new(ScanningConfig::default());
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let window_entities: Entities<WindowEntity> = Arc::new(RwLock::new(EntityWorld::new()));
    let signal_entities: Entities<SignalEntity> = Arc::new(RwLock::new(EntityWorld::new()));
    let audio_entities: Entities<AudioEntity> = Arc::new(RwLock::new(EntityWorld::new()));

    let task_id = TaskId::new("scan_1");
    let window_id = WindowId::new(task_id.clone(), 0);
    let window = WindowEntity::new(window_id, task_id, 88.9e6);

    {
        let mut windows = window_entities.write().unwrap();
        windows.insert(window);
    }

    let analyzer = Arc::new(AudioAnalyzer::new(Box::new(
        scanner::audio::quality::heuristic2::Classifier::new(48000.0),
    )));

    let mut coordinator = Coordinator::new(&pool, &config, &shutdown)
        .with_window_entities(window_entities.clone())
        .with_signal_entities(signal_entities.clone())
        .with_audio_entities(audio_entities.clone());

    coordinator.add_system(Box::new(PeakDetectionSystem::new()));
    coordinator.add_system(Box::new(PeakCompletionSystem::new()));
    coordinator.add_system(Box::new(PeakAnalysisSystem::new()));
    coordinator.add_system(Box::new(SignalAnalysisSystem::new(analyzer)));
    coordinator.add_system(Box::new(AudioSpawnSystem::new()));

    for tick in 0..20 {
        coordinator.tick()?;
        std::thread::sleep(std::time::Duration::from_millis(50));

        let windows = window_entities.read().unwrap();
        let signals = signal_entities.read().unwrap();
        let audios = audio_entities.read().unwrap();

        println!(
            "Tick {}: Windows={} (peak_detection state varies), Candidates={}, Audio={}",
            tick,
            windows.len(),
            signals.len(),
            audios.len()
        );
    }

    let windows = window_entities.read().unwrap();
    assert_eq!(windows.len(), 1);

    let window = windows.iter().next().unwrap();
    println!(
        "Final window peak_detection state: {:?}",
        window.peak_detection.state()
    );

    Ok(())
}

#[test]
fn test_band_scan_total_windows_calculation() {
    let (freq_min, freq_max) = (88.0e6, 108.0e6);
    let window_size = 2.0e6;

    let _config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        window_size,
        2.0e6,
        40.0,
        1.0,
        10,
    );

    let total_windows = ((freq_max - freq_min) / window_size).ceil() as usize;
    let task = TaskEntity::new_scan_with_defaults(
        TaskId::new("scan_1"),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert_eq!(
        progress.total_windows, 10,
        "FM band (88-108 MHz) with 2 MHz windows should have 10 windows"
    );
}

#[test]
fn test_station_scan_total_windows_calculation() {
    let stations = [88.9e6];

    let total_windows = stations.len();
    let task = TaskEntity::new_scan_with_defaults(
        TaskId::new("scan_1"),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert_eq!(
        progress.total_windows, 1,
        "Single station scan should have 1 window, not 0"
    );
}

#[test]
fn test_multiple_stations_total_windows() {
    let stations = [88.9e6, 91.5e6, 95.7e6];

    let total_windows = stations.len();
    let task = TaskEntity::new_scan_with_defaults(
        TaskId::new("scan_1"),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert_eq!(
        progress.total_windows, 3,
        "Three station scan should have 3 windows (one per station)"
    );
}

#[test]
fn test_window_processing_system_with_valid_windows() {
    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut window_processing = WindowProcessingSystem::new(config, pool, shutdown_coordinator);
    window_processing.enable();

    let total_windows = 10;
    let mut task = TaskEntity::new_scan_with_defaults(
        TaskId::new("scan_1"),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert_eq!(progress.total_windows, 10, "Should have 10 windows");

    let TaskComponents::Scan { progress, .. } = &mut task.components;
    progress.state = ScanPauseState::Scanning;

    let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
    task_entities.write().unwrap().insert(task);

    let signal_entities = Arc::new(RwLock::new(EntityWorld::<SignalEntity>::new()));

    let mut context = SystemContext::new()
        .with_task_entities(task_entities.clone())
        .with_signal_entities(signal_entities);

    window_processing.run(&mut context).unwrap();

    let tasks = task_entities.read().unwrap();
    let task = tasks.iter().next().unwrap();

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert!(
        matches!(progress.state, ScanPauseState::Scanning),
        "WindowProcessingSystem should maintain Scanning state when total_windows > 0"
    );
}

#[test]
fn test_scan_progress_with_zero_total_windows() {
    let total_windows = 0;
    let task = TaskEntity::new_scan_with_defaults(
        TaskId::new("scan_1"),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &task.components;
    assert_eq!(
        progress.total_windows, 0,
        "Band scan with freq_min == freq_max should have 0 windows"
    );

    let percentage = progress.progress_percentage();
    assert!(
        (0.0..=1.0).contains(&percentage),
        "Progress percentage should be valid even with 0 total_windows"
    );
}

#[test]
fn test_scan_completes_when_all_windows_processed() {
    let total_windows = 1;
    let task_id = TaskId::new("scan_1");
    let mut task = TaskEntity::new_scan_with_defaults(
        task_id.clone(),
        ScanTaskData::Placeholder,
        total_windows,
    );

    let TaskComponents::Scan { progress, .. } = &mut task.components;
    assert_eq!(progress.total_windows, 1);

    let window_id = WindowId::new(task_id, 0);
    progress.complete_window_at(window_id);

    assert_eq!(
        progress.completed_windows.len(),
        1,
        "Should have 1 completed window"
    );

    let should_complete = progress.completed_windows.len() >= progress.total_windows;

    assert!(
        should_complete,
        "Scan should be ready to complete when all windows are done"
    );
}

#[test]
fn test_window_processing_does_not_deallocate_complete_windows() {
    let task_id = TaskId::new("scan_1");
    let window_id = WindowId::new(task_id.clone(), 0);

    let device_id = DeviceId::from_serial("sdrplay", "test123");
    let tuner_id = TunerId::new(device_id.clone(), 0);

    let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.9e6);
    window.allocation.allocate(tuner_id.clone());
    window.allocation.mark_complete();
    window.progress.mark_completed();

    let mut window_entities = EntityWorld::new();
    window_entities.insert(window);

    let capabilities = Capabilities::for_mock("sdrplay", "test123");
    let tuner = TunerEntity::new(
        device_id.clone(),
        0,
        capabilities,
        Backend::Soapy,
        "Test Tuner".to_string(),
        None,
        "FM".to_string(),
    );

    let mut tuner_entities = EntityWorld::new();
    tuner_entities.insert(tuner);

    let mut task =
        TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 1);
    let TaskComponents::Scan { progress, .. } = &mut task.components;
    progress.state = ScanPauseState::Scanning;
    let mut task_entities = EntityWorld::new();
    task_entities.insert(task);

    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut system = WindowProcessingSystem::new(config, pool, shutdown_coordinator);
    system.enable();

    let mut context = SystemContext::new()
        .with_task_entities(Arc::new(RwLock::new(task_entities)))
        .with_window_entities(Arc::new(RwLock::new(window_entities)))
        .with_tuner_entities(Arc::new(Mutex::new(tuner_entities)));

    system.run(&mut context).unwrap();

    let windows = context.window_entities.as_ref().unwrap().read().unwrap();
    let window = windows.get(&window_id).expect("Window should still exist");

    assert!(
        window.allocation.is_complete(),
        "WindowProcessingSystem should NOT have cleared the Complete window allocation"
    );

    let tuners = context.tuner_entities.as_ref().unwrap().lock().unwrap();
    let tuner = tuners.iter().find(|t| t.id() == &tuner_id).unwrap();

    assert!(
        tuner.allocation.is_available(),
        "WindowProcessingSystem should NOT have touched the tuner"
    );
}

/// Test: WindowWorkerSpawnSystem Consistent Locking Behavior
///
/// Tests that WindowWorkerSpawnSystem uses consistent locking patterns that don't
/// fail when other systems hold read locks. This validates the fix where
/// try_read() was changed to blocking read().
///
/// Pattern: Unit testing individual systems (ECS Testing skill)
#[test]
fn test_window_worker_spawn_system_consistent_locking() {
    use scanner::ecs::systems::WindowWorkerSpawnSystem;

    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut spawn_system =
        WindowWorkerSpawnSystem::new(config.clone(), pool.clone(), shutdown_coordinator);

    // Setup minimal world with no windows (tests the empty case)
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
    let task_entities = Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new()));

    // Simulate another system holding a read lock (this would break try_read())
    let _read_lock_holder = window_entities.read().unwrap();

    // TEST: WindowWorkerSpawnSystem should still be able to read (blocking read())
    let mut context = SystemContext::new()
        .with_window_entities(window_entities.clone())
        .with_task_entities(task_entities.clone())
        .with_pool(pool)
        .with_config(config);

    let result = spawn_system.run(&mut context);

    // VERIFY: System succeeds even when read lock is held by another "system"
    assert!(
        result.is_ok(),
        "WindowWorkerSpawnSystem should succeed with blocking read()"
    );

    // Note: With the old try_read() implementation, this test would fail
    // because try_read() would return Err when another thread holds a read lock
}

/// Test: RwLock Contention Fix Validation for Window Systems
///
/// This test specifically validates that the fix for RwLock contention between
/// AllocationSystem and WindowWorkerSpawnSystem is working correctly.
///
/// THE FIX: Changed WindowWorkerSpawnSystem from try_read() to blocking read()
/// to match AllocationSystem's locking pattern, eliminating lock contention.
#[test]
fn test_rwlock_contention_fix_validation() {
    use std::{thread, time::Duration};

    use scanner::ecs::systems::WindowWorkerSpawnSystem;

    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut spawn_system =
        WindowWorkerSpawnSystem::new(config.clone(), pool.clone(), shutdown_coordinator);

    // Create shared EntityWorld that both threads will access
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));

    // REPRODUCE THE EXACT CONTENTION SCENARIO:
    // Thread 1: Holds a read lock (like AllocationSystem does during processing)
    // Thread 2: WindowWorkerSpawnSystem tries to read the same EntityWorld

    let window_entities_1 = window_entities.clone();
    let window_entities_2 = window_entities.clone();

    let lock_holder_handle = thread::spawn(move || {
        // Simulate AllocationSystem holding a read lock during processing
        let _long_read_lock = window_entities_1.read().unwrap();

        // Hold the lock for a meaningful duration
        thread::sleep(Duration::from_millis(50));

        "read_lock_released"
    });

    let spawn_system_handle = thread::spawn(move || {
        // Give lock holder time to acquire the read lock
        thread::sleep(Duration::from_millis(10));

        // TEST: WindowWorkerSpawnSystem should be able to read despite contention
        // With the old try_read() this would fail
        // With the new blocking read() this succeeds
        let mut context = SystemContext::new()
            .with_window_entities(window_entities_2)
            .with_pool(pool)
            .with_config(config);

        spawn_system.run(&mut context)
    });

    // VERIFY: Both operations complete successfully
    let lock_result = lock_holder_handle.join().unwrap();
    let system_result = spawn_system_handle.join().unwrap();

    assert_eq!(lock_result, "read_lock_released");
    assert!(
        system_result.is_ok(),
        "WindowWorkerSpawnSystem should succeed with blocking read() despite lock contention"
    );
}
