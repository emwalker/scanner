//! Integration tests for ECS systems with Pool

use std::sync::{Arc, Mutex, RwLock};

use crate::{
    core::signals::ModulationType,
    ecs::{
        AudioEntity, DeviceEntity, EntityWorld, ScanTaskData, Scheduler, TaskEntity, TaskId,
        components::{Priority, WindowId},
        entity::Entity,
        system::{System, SystemContext},
        systems::{
            AllocationRequest, AllocationSystem, AudioCoordinationSystem, DiscoverySystem,
            ScanRequestProcessorSystem,
        },
    },
    hardware::pool::{Pool, PoolFilter},
};

#[test]
fn test_discovery_system_with_pool() {
    let pool = Pool::new_unfiltered();

    let mut context = SystemContext::new().with_tuner_entities(Arc::clone(&pool.tuner_entities));

    let mut system = DiscoverySystem::new();
    let result = system.run(&mut context);

    assert!(result.is_ok());
}

#[test]
fn test_scheduler_with_multiple_systems() {
    let mut scheduler = Scheduler::new();
    scheduler.add_system(Box::new(DiscoverySystem::new()));
    scheduler.add_system(Box::new(AllocationSystem::new()));

    let pool = Pool::new_unfiltered();
    let mut context = SystemContext::new().with_tuner_entities(Arc::clone(&pool.tuner_entities));

    let result = scheduler.run(&mut context);
    assert!(result.is_ok());
    assert_eq!(scheduler.system_count(), 2);
}

#[test]
fn test_allocation_system_integration() {
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let device_entities = Arc::new(Mutex::new(EntityWorld::<DeviceEntity>::new()));
    let pool = Arc::new(Pool::with_entity_worlds(
        PoolFilter::allow_all(),
        None,
        tuner_entities.clone(),
        device_entities,
    ));

    let mut context = SystemContext::new()
        .with_tuner_entities(tuner_entities)
        .with_pool(pool);

    let mut system = AllocationSystem::new();
    system.request_allocation(AllocationRequest {
        requester_id: "test_scan".to_string(),
        frequency_hz: 88_900_000.0,
        sample_rate_hz: 2_000_000.0,
        priority: Priority::Medium,
        for_audio: false,
        filter: None,
        allocated_count: 0,
    });

    let result = system.run(&mut context);
    assert!(result.is_ok());
}

// Phase 4: Integration tests for request component flows

fn create_test_task(task_id: &str, total_windows: usize) -> TaskEntity {
    TaskEntity::new_scan_with_defaults(
        TaskId::new(task_id.to_string()),
        ScanTaskData::Placeholder,
        total_windows,
    )
}

#[test]
fn test_scan_request_processor_pause_flow() {
    let mut world = EntityWorld::new();
    let mut task = create_test_task("test-scan", 10);
    let task_id = task.id().clone();

    let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
    let window_id = WindowId::new(task_id, 0);
    progress.start_window(window_id);
    assert!(progress.is_scanning());

    // TUI sets pause request
    task.request_pause(5);

    let crate::ecs::TaskComponents::Scan { pause_request, .. } = &task.components;
    assert!(pause_request.is_some());

    world.insert(task);
    let task_entities = Arc::new(RwLock::new(world));

    // System processes request
    let mut context = SystemContext::new().with_task_entities(task_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = task_entities.read().unwrap();
    for task in entities.iter() {
        let crate::ecs::TaskComponents::Scan {
            pause_request,
            progress,
            ..
        } = &task.components;
        assert!(pause_request.is_none(), "Request should be cleared");
        assert!(progress.is_paused(), "Scan should be paused");
    }
}

#[test]
fn test_scan_request_processor_resume_flow() {
    let mut world = EntityWorld::new();
    let mut task = create_test_task("test-scan", 10);
    let task_id = task.id().clone();

    // Start paused
    let crate::ecs::TaskComponents::Scan { progress, .. } = &mut task.components;
    let window_id = WindowId::new(task_id, 5);
    progress.pause(window_id);
    assert!(progress.is_paused());

    // TUI sets resume request
    task.request_resume(5);

    let crate::ecs::TaskComponents::Scan { resume_request, .. } = &task.components;
    assert!(resume_request.is_some());

    world.insert(task);
    let task_entities = Arc::new(RwLock::new(world));

    // System processes request
    let mut context = SystemContext::new().with_task_entities(task_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = task_entities.read().unwrap();
    for task in entities.iter() {
        let crate::ecs::TaskComponents::Scan {
            resume_request,
            progress,
            ..
        } = &task.components;
        assert!(resume_request.is_none(), "Request should be cleared");
        assert!(progress.is_scanning(), "Scan should be scanning");
    }
}

#[test]
fn test_audio_coordination_stop_listening_flow() {
    use std::time::SystemTime;

    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, Signal},
        ecs::{TaskId, components::window::WindowId},
    };

    let signal = Signal {
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

    let mut audio_world = EntityWorld::new();
    let mut audio = AudioEntity::new(signal, 88.9e6, None);

    // TUI sets stop_listening request
    audio.request_stop_listening();
    assert!(audio.stop_listening_request.is_some());
    assert!(audio.is_playing());

    audio_world.insert(audio);

    let mut signal_world = EntityWorld::new();

    // Add SignalEntity that the AudioCoordinationSystem needs
    let task_id = TaskId::new("test_task");
    let window_id = WindowId::new(task_id, 0);
    let signal_entity = crate::ecs::SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
    signal_world.insert(signal_entity);

    let audio_entities = Arc::new(RwLock::new(audio_world));
    let signal_entities = Arc::new(RwLock::new(signal_world));

    // System processes request
    let mut context = SystemContext::new()
        .with_audio_entities(audio_entities.clone())
        .with_signal_entities(signal_entities);

    let mut system = AudioCoordinationSystem::new();
    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = audio_entities.read().unwrap();
    for audio in entities.iter() {
        assert!(
            audio.stop_listening_request.is_none(),
            "Request should be cleared"
        );
        assert!(!audio.is_playing(), "Audio should be stopped");
    }
}

#[test]
fn test_dual_read_both_paths_work() {
    let mut world = EntityWorld::new();
    let mut task = create_test_task("test-scan", 10);

    // TUI sets request component (ECS path)
    task.request_pause(5);

    let crate::ecs::TaskComponents::Scan { pause_request, .. } = &task.components;
    assert!(pause_request.is_some());

    // Old command path would also be sent (but we're just testing ECS here)

    world.insert(task);
    let task_entities = Arc::new(RwLock::new(world));

    // System processes ECS request
    let mut context = SystemContext::new().with_task_entities(task_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    system.run(&mut context).unwrap();

    // Verify ECS path worked
    let entities = task_entities.read().unwrap();
    for task in entities.iter() {
        let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
        assert!(progress.is_paused(), "ECS path should pause scan");
    }
}

#[test]
fn test_scheduler_with_request_processor() {
    let mut scheduler = Scheduler::new();
    scheduler.add_system(Box::new(ScanRequestProcessorSystem::new()));
    scheduler.add_system(Box::new(AudioCoordinationSystem::new()));

    let mut scan_world = EntityWorld::new();
    let mut scan = create_test_task("test-scan", 10);
    scan.request_pause(5);
    scan_world.insert(scan);

    let task_entities = Arc::new(RwLock::new(scan_world));
    let audio_entities = Arc::new(RwLock::new(EntityWorld::<AudioEntity>::new()));

    let mut context = SystemContext::new()
        .with_task_entities(task_entities.clone())
        .with_audio_entities(audio_entities);

    let result = scheduler.run(&mut context);
    assert!(result.is_ok());

    let entities = task_entities.read().unwrap();
    for task in entities.iter() {
        let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
        assert!(
            progress.is_paused(),
            "Request should be processed by scheduler"
        );
    }
}

/// Integration test: Segment lifetime with signalAnalysisSpawnSystem
///
/// Tests the complete flow of the Segment lifetime fix:
/// 1. Window worker completes and stores Segment (does NOT call mark_all_spawned)
/// 2. SignalAnalysisSpawnSystem spawns signals and calls mark_all_spawned
/// 3. Segment survives for signals to subscribe
/// 4. AudioStreamManagementSystem cleans up Segment after signals complete
///
/// This reproduces and verifies the fix for the bug where window 7-8 showed
/// "No audio" timeouts because Segment was dropped before signals spawned.
#[test]
fn test_segment_survives_until_signals_spawn_and_complete() {
    use tokio::sync::broadcast;

    use crate::{
        core::config::ScanningConfig,
        ecs::{
            SignalEntity, WindowEntity, components::AnalysisInputComponent,
            systems::SignalAnalysisSpawnSystem,
        },
        hardware::{DeviceId, pool::TunerId},
    };

    // Setup: Create window with 3 signals ready to spawn
    let task_id = TaskId::new("test-scan");
    let window_id = WindowId::new(task_id.clone(), 0);
    let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 0.0);

    // Simulate window worker completed: allocation is Active but all_spawned=false
    let tuner_id = TunerId {
        device_id: DeviceId::from_serial("mock", "test-123"),
        channel_index: 0,
    };
    window.allocation.start_active(tuner_id.clone(), 3);
    window.progress.mark_completed();

    // Verify: all_work_complete() should return FALSE because all_spawned=false
    assert!(
        !window.allocation.all_work_complete(),
        "BEFORE FIX: Window would incorrectly have all_spawned=true here, causing Segment to \
         drop. AFTER FIX: all_spawned=false, Segment survives."
    );

    // Create 3 signals with analysis inputs ready to spawn
    let config = Arc::new(ScanningConfig::default());
    let mut signals = Vec::new();
    for i in 0..3 {
        let mut signal = SignalEntity::new(
            88.0e6 + (i as f64) * 0.1e6,
            window_id.clone(),
            ModulationType::WFM,
        );

        // Give signal an analysis input so it's ready to spawn
        let (_tx_refining, sdr_rx_refining) = broadcast::channel(32);
        let (_tx_detection, sdr_rx_detection) = broadcast::channel(32);
        signal.analysis_input = Some(AnalysisInputComponent {
            config: config.clone(),
            center_freq: 88.0e6,
            window_id: window_id.clone(),
            sdr_rx_refining,
            sdr_rx_detection,
            pause_signal: None,
        });

        signals.push(signal);
    }

    // Setup entity worlds
    let window_entities = Arc::new(RwLock::new(EntityWorld::new()));
    window_entities.write().unwrap().insert(window);

    let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));
    {
        let mut entities = signal_entities.write().unwrap();
        for signal in signals {
            entities.insert(signal);
        }
    }

    // Execute: Run signalAnalysisSpawnSystem to spawn signals
    let mut spawn_system = SignalAnalysisSpawnSystem::new();
    let mut context = SystemContext::new()
        .with_signal_entities(signal_entities.clone())
        .with_window_entities(window_entities.clone());

    let result = spawn_system.run(&mut context);
    assert!(result.is_ok(), "signalAnalysisSpawnSystem should succeed");

    // Verify: After spawning, signals_analyzing > 0, so all_work_complete() = false
    // The GREEN FIX ensures mark_all_spawned() was called, but we can't check it directly
    // (no public accessor). Instead, we verify the important behavior:
    // all_work_complete() returns false while signals are analyzing
    let windows = window_entities.read().unwrap();
    let window = windows.get(&window_id).expect("Window should exist");

    assert!(
        !window.allocation.all_work_complete(),
        "GREEN FIX: Segment should NOT be dropped yet - signals are still analyzing. Even though \
         mark_all_spawned() was called, signals_analyzing > 0."
    );

    // Cleanup: Complete all signal analysis
    drop(windows);
    {
        let mut windows = window_entities.write().unwrap();
        let window = windows.get_mut(&window_id).unwrap();
        // Simulate all signals completing
        for _ in 0..3 {
            window.allocation.complete_analysis();
        }
    }

    // Verify: NOW all_work_complete() returns true (all_spawned=true, signals_analyzing=0)
    let windows = window_entities.read().unwrap();
    let window = windows.get(&window_id).unwrap();
    assert!(
        window.allocation.all_work_complete(),
        "After signals finish, Segment can be cleaned up"
    );
}
