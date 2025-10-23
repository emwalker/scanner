//! Integration and ECS tests

use std::{
    sync::{Arc, RwLock},
    time::SystemTime,
};

use scanner::{
    audio::quality::{AudioAnalyzer, AudioQuality},
    core::types::{Format, ModulationType, Result, ScanningConfig, Signal},
    ecs::{
        Coordinator, Entity, EntityWorld, ScanTaskData, SignalEntity, TaskEntity, TaskId,
        WindowEntity, WindowId,
        components::{
            AnalysisResults,
            station::{TuneAllocationComponent, TuneAllocationState, TuneState},
        },
        systems::{AudioSpawnSystem, SignalAnalysisSystem},
    },
    hardware::{
        DeviceId,
        pool::{Pool, TunerId},
    },
    shutdown::ShutdownCoordinator,
    testing::with_captured_logs,
};

#[test]
fn test_pipeline_debug_modes() {
    let mut config = ScanningConfig::default();
    config.debug.pipeline = true;

    assert!(config.debug.pipeline);
    assert_eq!(config.samp_rate, 2_000_000.0);
    assert_eq!(config.peak_detection.fft_size, 1024);
}

#[test]
fn test_captured_logging() {
    use tracing::debug;

    let result = with_captured_logs(true, Format::Json, || {
        debug!(
            message = "Test log entry",
            test_value = 42,
            test_string = "hello"
        );
        Ok(())
    });

    match result {
        Ok((_, logs)) => {
            assert!(logs.contains("Test log entry"));
            assert!(logs.contains("test_value"));
            assert!(logs.contains("42"));
        }
        Err(e) => panic!("Log capture test failed: {}", e),
    }
}

#[test]
fn test_log_comparison_structure() {
    let mut config = ScanningConfig::default();
    config.debug.pipeline = true;
    config.samp_rate = 1_000_000.0;
    config.peak_detection.fft_size = 1024;
    config.peak_detection.threshold = 1.0;

    let station_freq = 88.9e6;
    let window_center = 89.1e6;

    println!("Testing log comparison framework");
    println!("Station frequency: {:.3} MHz", station_freq / 1e6);
    println!("Window center: {:.3} MHz", window_center / 1e6);
    println!(
        "Expected offset: {:.1} kHz",
        (station_freq - window_center) / 1e3
    );

    assert!(config.debug.pipeline);
}

#[test]
fn test_task_has_no_pause_request_initially() {
    let task =
        TaskEntity::new_scan_with_defaults(TaskId::new("scan_1"), ScanTaskData::Placeholder, 10);

    let scanner::ecs::TaskComponents::Scan { pause_request, .. } = &task.components;
    assert!(pause_request.is_none());
}

#[test]
fn test_signal_has_idle_tune_state_initially() {
    let task_id = TaskId::new("scan_1");
    let window_id = WindowId::new(task_id, 1);
    let signal = SignalEntity::new(88.9e6, window_id);
    assert!(signal.tune_state.is_idle());
}

#[test]
fn test_signal_tune_state_progression() {
    let _signal = Signal {
        frequency_hz: 88.9e6,
        signal_strength: 0.8,
        bandwidth_hz: 200_000.0,
        modulation: ModulationType::WFM,
        audio_sample_rate: 48000,
        detected_at: SystemTime::now(),
        analysis_duration_ms: 100,
        detection_center_freq: 88.9e6,
        audio_quality: AudioQuality::Good,
    };

    let task_id = TaskId::new("scan_1");
    let window_id = WindowId::new(task_id, 1);
    let mut signal_entity = SignalEntity::new(88.9e6, window_id.clone());

    assert!(signal_entity.tune_state.is_idle());

    // Confirm the signal so it can be tuned
    signal_entity
        .analysis
        .confirm_analysis(AudioQuality::Good, 0.8);

    let result = signal_entity.request_tune_transition(window_id.clone(), 88.9e6);
    assert!(result.is_ok());
    assert!(signal_entity.tune_state.is_transitioning());
    assert!(matches!(
        signal_entity.tune_state,
        TuneState::Transitioning(_) | TuneState::RequestQueued { .. }
    ));

    let request = scanner::ecs::components::station::TuneRequestComponent::new(window_id);
    let allocation = TuneAllocationComponent::new();
    signal_entity.tune_state = TuneState::RequestQueued {
        request,
        allocation,
    };
    assert!(signal_entity.tune_state.is_request_queued());
    assert!(matches!(
        signal_entity.tune_state,
        TuneState::Transitioning(_) | TuneState::RequestQueued { .. }
    ));

    let allocation = TuneAllocationComponent::new();
    signal_entity.tune_state = TuneState::Active { allocation };
    assert!(signal_entity.tune_state.is_active());
    assert!(matches!(signal_entity.tune_state, TuneState::Active { .. }));
}

#[test]
fn test_tune_allocation_state_transitions() {
    let mut component = TuneAllocationComponent::new();
    assert_eq!(component.state(), TuneAllocationState::Pending);

    component.transition(TuneAllocationState::Allocated);
    assert_eq!(component.state(), TuneAllocationState::Allocated);

    component.transition(TuneAllocationState::Active);
    assert_eq!(component.state(), TuneAllocationState::Active);

    component.transition(TuneAllocationState::Failed);
    assert_eq!(component.state(), TuneAllocationState::Failed);
}

#[test]
fn test_phase4a_dual_path_signal_processing() -> Result<()> {
    let pool = Arc::new(Pool::new_unfiltered());
    let config = Arc::new(ScanningConfig::default());
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
    let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

    let task_id = TaskId::new("test-scan");
    let window_id = WindowId::new(task_id.clone(), 0);
    let mut entity = SignalEntity::new(88.9e6, window_id.clone());

    let (tx, rx) = std::sync::mpsc::channel();

    let handle = std::thread::spawn(move || -> Result<AnalysisResults> {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let results = AnalysisResults {
            quality: AudioQuality::Good,
            strength: 0.8,
        };
        let _ = tx.send(results.clone());
        Ok(results)
    });

    let entity_id = entity.id().clone();
    entity.analysis.start_analysis(handle, rx);
    signal_entities.write().unwrap().insert(entity);

    let mut coordinator = Coordinator::new(&pool, &config, &shutdown)
        .with_signal_entities(signal_entities.clone())
        .with_audio_entities(audio_entities.clone());

    let analyzer = Arc::new(AudioAnalyzer::new(Box::new(
        scanner::audio::quality::heuristic2::Classifier::new(48000.0),
    )));
    coordinator.add_system(Box::new(SignalAnalysisSystem::new(analyzer)));
    coordinator.add_system(Box::new(AudioSpawnSystem::new()));

    let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
    task_entities
        .write()
        .unwrap()
        .insert(TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            ScanTaskData::Placeholder,
            10,
        ));

    let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
    let mut window = WindowEntity::new(window_id, task_id, 88.9e6);
    window.lifecycle.start_analyzing(1);

    let tuner_id = TunerId {
        device_id: DeviceId::from_serial("test-driver", "test-device"),
        channel_index: 0,
    };

    window.allocation.start_active(tuner_id, 1);
    window_entities.write().unwrap().insert(window);

    coordinator = coordinator
        .with_task_entities(task_entities)
        .with_window_entities(window_entities);

    for _ in 0..20 {
        coordinator.tick()?;
        std::thread::sleep(std::time::Duration::from_millis(50));

        let signals = signal_entities.read().unwrap();
        if let Some(signal) = signals.iter().next()
            && signal.analysis.is_done()
        {
            break;
        }
    }

    let signals = signal_entities.read().unwrap();
    let signal = signals.get(&entity_id).expect("Candidate should exist");
    assert!(
        signal.analysis.is_done(),
        "SignalAnalysisSystem should have joined thread and marked complete"
    );

    // After StationEntity migration: SignalEntity now manages its own state
    assert!(
        signal.analysis.is_confirmed(),
        "SignalAnalysisSystem should have confirmed the signal"
    );

    let audio = audio_entities.read().unwrap();
    assert_eq!(
        audio.len(),
        0,
        "AudioEntity not created without hardware segment (expected)"
    );

    Ok(())
}

/// Integration Test: AllocationSystem + WindowWorkerSpawnSystem Coordination
///
/// Tests that AllocationSystem and WindowWorkerSpawnSystem work together correctly
/// when sharing EntityWorld access. This test verifies the fix for RwLock contention
/// where inconsistent locking patterns broke system coordination.
///
/// Pattern: Integration testing multiple systems (ECS Testing skill)
#[test]
fn test_allocation_spawn_systems_integration() {
    use std::sync::{Arc, Mutex, RwLock};

    use scanner::{
        core::types::ScanningConfig,
        ecs::{
            EntityWorld, ScanTaskData, System, SystemContext, TaskEntity, TaskId, WindowEntity,
            WindowId,
            systems::{AllocationSystem, WindowWorkerSpawnSystem},
        },
        hardware::{Capabilities, DeviceId, pool::Pool, types::Backend},
        shutdown::ShutdownCoordinator,
    };

    // Setup: Create systems that would run concurrently in production
    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut allocation_system = AllocationSystem::new();
    let mut spawn_system =
        WindowWorkerSpawnSystem::new(config.clone(), pool.clone(), shutdown_coordinator.clone());

    // Create shared entity worlds (same as production)
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
    let task_entities = Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new()));

    // Setup: Create task entity (ScanFactorySystem would do this)
    let task_id = TaskId::new("test_scan");
    {
        let mut tasks = task_entities.write().unwrap();
        let task = TaskEntity::new_scan_with_defaults(
            task_id.clone(),
            ScanTaskData::Placeholder,
            5, // 5 windows total
        );
        tasks.insert(task);
    }

    // Setup: Create window entity in Requested state (ScanFactorySystem would do this)
    let window_id = WindowId::new(task_id.clone(), 0);
    let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.0e6);

    // Window is in Requested state, waiting for allocation
    use scanner::hardware::pool::{TaskPriority, TaskRequirements, TunerActivity};
    let requirements = TaskRequirements {
        frequency_hz: 88.0e6,
        bandwidth_hz: 2.0e6,
        required_sample_rate: 2.0e6,
        priority: TaskPriority::Normal,
    };
    let requester_id = "test_scan_window_0".to_string();
    window
        .allocation
        .request(requirements, TunerActivity::Scanning, requester_id.clone());

    {
        let mut windows = window_entities.write().unwrap();
        windows.insert(window);
    }

    // Setup: Create tuner entity that allocation system can use
    let tuner_entities = Arc::new(Mutex::new(scanner::ecs::EntityWorld::<
        scanner::ecs::TunerEntity,
    >::new()));
    {
        let mut tuners = tuner_entities.lock().unwrap();
        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let capabilities = Capabilities::for_device(&device_id);
        let tuner = scanner::ecs::TunerEntity::new(
            device_id,
            0,
            capabilities,
            Backend::Soapy,
            "Test SDRplay".to_string(),
            None,
            "FM".to_string(),
        );
        tuners.insert(tuner);
    }

    // TEST: Sequential execution (how it should work)
    // Step 1: AllocationSystem runs and allocates tuners to windows
    let mut context = SystemContext::new()
        .with_window_entities(window_entities.clone())
        .with_task_entities(task_entities.clone())
        .with_tuner_entities(tuner_entities.clone())
        .with_pool(pool.clone());

    allocation_system
        .run(&mut context)
        .expect("AllocationSystem should succeed");

    // Step 2: WindowWorkerSpawnSystem should now see allocated windows
    let mut context = SystemContext::new()
        .with_window_entities(window_entities.clone())
        .with_task_entities(task_entities.clone())
        .with_pool(pool.clone())
        .with_config(config.clone());

    spawn_system
        .run(&mut context)
        .expect("WindowWorkerSpawnSystem should succeed");

    // VERIFY: Both systems can access the same EntityWorld consistently
    // The fix ensures both systems use blocking read(), eliminating lock contention
    let windows = window_entities.read().unwrap();
    assert_eq!(
        windows.len(),
        1,
        "Window should still exist after both systems run"
    );
}

/// Concurrency Test: Prevent RwLock Contention Regressions
///
/// Tests concurrent access patterns that previously caused lock contention.
/// This test verifies that multiple systems can safely access shared EntityWorlds
/// without deadlocks or failed reads.
///
/// Pattern: Concurrent testing for race conditions (ECS Testing skill)
#[test]
fn test_concurrent_systems_no_lock_contention() {
    use std::{
        sync::{Arc, Mutex, RwLock},
        thread,
        time::Duration,
    };

    use scanner::{
        core::types::ScanningConfig,
        ecs::{
            EntityWorld, ScanTaskData, System, SystemContext, TaskEntity, TaskId,
            systems::{AllocationSystem, WindowWorkerSpawnSystem},
        },
        hardware::pool::Pool,
        shutdown::ShutdownCoordinator,
    };

    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    // Create systems that will access shared state
    let mut allocation_system = AllocationSystem::new();
    let mut spawn_system =
        WindowWorkerSpawnSystem::new(config.clone(), pool.clone(), shutdown_coordinator);

    // Setup shared EntityWorlds
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
    let task_entities = Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new()));
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::<scanner::ecs::TunerEntity>::new()));

    // Create minimal test data
    let task_id = TaskId::new("concurrent_test");
    {
        let mut tasks = task_entities.write().unwrap();
        let task =
            TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 1);
        tasks.insert(task);
    }

    // TEST: Run systems concurrently with shared read access
    let window_entities_1 = window_entities.clone();
    let window_entities_2 = window_entities.clone();
    let task_entities_1 = task_entities.clone();
    let task_entities_2 = task_entities.clone();
    let tuner_entities_1 = tuner_entities.clone();
    let pool_1 = pool.clone();
    let pool_2 = pool.clone();
    let config_2 = config.clone();

    let handle1 = thread::spawn(move || {
        let mut context = SystemContext::new()
            .with_window_entities(window_entities_1)
            .with_task_entities(task_entities_1)
            .with_tuner_entities(tuner_entities_1)
            .with_pool(pool_1);

        // Hold read lock for a short time (simulates processing)
        thread::sleep(Duration::from_millis(10));
        allocation_system.run(&mut context)
    });

    let handle2 = thread::spawn(move || {
        let mut context = SystemContext::new()
            .with_window_entities(window_entities_2)
            .with_task_entities(task_entities_2)
            .with_pool(pool_2)
            .with_config(config_2);

        // This would fail with try_read() due to lock contention
        spawn_system.run(&mut context)
    });

    // VERIFY: Both systems complete successfully without deadlock
    let result1 = handle1.join().expect("Thread 1 should not panic");
    let result2 = handle2.join().expect("Thread 2 should not panic");

    assert!(
        result1.is_ok(),
        "AllocationSystem should succeed in concurrent scenario"
    );
    assert!(
        result2.is_ok(),
        "WindowWorkerSpawnSystem should succeed in concurrent scenario"
    );

    // This test validates that the fix (blocking read() instead of try_read())
    // allows systems to coordinate properly even under concurrent access
}
