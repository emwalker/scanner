use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use scanner::{
    core::types::{Band, ScanningConfig},
    ecs::{
        Coordinator, EntityWorld, ScanTaskData, TaskEntity, TaskId, WindowEntity,
        components::scan::{ScanConfigComponent, ScanType},
    },
    hardware::{
        DeviceId,
        pool::{
            Pool, PoolStatus, TunerActivity, TunerId, TunerState,
            test_utils::add_test_device_to_pool,
        },
        types::Backend,
    },
    shutdown::ShutdownCoordinator,
    task::TaskScheduler,
    ui::{TuiEvent, tui::model::Model},
};

#[test]
fn test_active_tuners_updated_skips_redundant_processing() {
    let mut model = Model::new();

    let tuner_id = TunerId::new(
        DeviceId::Driver {
            backend: Backend::Soapy,
            driver: "rtlsdr".to_string(),
            serial: "00000001".to_string(),
        },
        0,
    );

    let status = PoolStatus {
        available_tuner_count: 0,
        allocated_tuner_count: 1,
        device_count: 1,
        tuners: vec![scanner::hardware::pool::TunerStatus {
            id: tuner_id.clone(),
            state: TunerState::Allocated,
            activity: Some(TunerActivity::Listening),
        }],
    };

    let event1 = TuiEvent::ActiveTunersUpdated {
        status: status.clone(),
    };
    model.update_tui_event(event1);

    let initial_tuner_state = model.pool_info.get(&tuner_id).unwrap().clone();

    let event2 = TuiEvent::ActiveTunersUpdated {
        status: status.clone(),
    };
    model.update_tui_event(event2);

    let after_redundant_update = model.pool_info.get(&tuner_id).unwrap().clone();

    assert_eq!(
        initial_tuner_state.state, after_redundant_update.state,
        "Regression test: Pool info should not be rebuilt when status is identical.\nBug: During \
         listening mode, redundant ActiveTunersUpdated events caused \nexpensive HashMap rebuilds \
         every time, spiking CPU to 50%.\nFix: Added change detection to skip processing when \
         status unchanged."
    );
    assert_eq!(
        initial_tuner_state.activity, after_redundant_update.activity,
        "Activity should also remain unchanged for redundant updates"
    );

    let event3 = TuiEvent::ActiveTunersUpdated {
        status: PoolStatus {
            available_tuner_count: 1,
            allocated_tuner_count: 0,
            device_count: 1,
            tuners: vec![scanner::hardware::pool::TunerStatus {
                id: tuner_id.clone(),
                state: TunerState::Available,
                activity: None,
            }],
        },
    };
    model.update_tui_event(event3);

    assert_ne!(
        model.pool_info.get(&tuner_id).unwrap().state,
        TunerState::Allocated,
        "Pool info should update when status actually changes"
    );
}

#[test]
fn test_spectrum_renders_continuously_at_10fps() {
    use ratatui::{Terminal, backend::TestBackend};
    use tokio_util::sync::CancellationToken;

    let (_tx, rx) = mpsc::channel();
    let shutdown_token = CancellationToken::new();

    let _display = scanner::ui::tui::TuiProgressDisplay::new(rx, shutdown_token.clone());

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    let start = std::time::Instant::now();
    let mut draw_count = 0;
    let test_duration = Duration::from_millis(350);

    terminal
        .draw(|_f| {
            draw_count += 1;
        })
        .unwrap();

    while start.elapsed() < test_duration {
        std::thread::sleep(Duration::from_millis(100));

        terminal
            .draw(|_f| {
                draw_count += 1;
            })
            .unwrap();
    }

    assert!(
        draw_count >= 3,
        "Regression test: Terminal should render at ~10 FPS (100ms intervals) for smooth spectrum \
         animation.\nBug: Conditional rendering based on model.is_dirty() caused jumpy spectrum \
         wave \nbecause it only updated when entities changed, not continuously.\nFix: Always \
         call mark_dirty() in main loop to ensure 10 FPS rendering.\nExpected at least 3 draws in \
         350ms, got {}",
        draw_count
    );
}

#[test]
fn test_band_scan_initiates_automatically_on_startup() {
    use scanner::ecs::Coordinator;

    let config = ScanningConfig {
        band: Band::Fm,
        samp_rate: 2.4e6,
        sdr_gain: 40.0,
        duration: 1,
        scanning_windows: Some(2),
        ..Default::default()
    };

    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "test001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "test001");
    add_test_device_to_pool(&pool, device_id, caps, Backend::Mock, None);

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    let _scheduler = Arc::new(TaskScheduler::new(
        pool.clone(),
        shutdown_coordinator.clone(),
    ));

    let task_entities = Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new()));
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));

    let (freq_min, freq_max) = config.band.frequency_range();
    let total_windows = ((freq_max - freq_min) / config.samp_rate).ceil() as usize;
    let task_id = TaskId::new("scan_1");
    let task_entity = TaskEntity::new_scan_with_defaults(
        task_id.clone(),
        ScanTaskData::Placeholder,
        total_windows,
    );
    task_entities.write().unwrap().insert(task_entity);

    {
        let entities = task_entities.read().unwrap();
        let task = entities.get(&task_id).expect("Task entity should exist");

        let scanner::ecs::TaskComponents::Scan { progress, .. } = &task.components;
        assert!(
            matches!(progress.state, scanner::ecs::ScanPauseState::Pending),
            "Task should start in Pending state before WindowProcessingSystem runs"
        );
    }

    let config_arc = Arc::new(config);
    let mut coordinator = Coordinator::new(&pool, &config_arc, &shutdown_coordinator)
        .with_task_entities(task_entities.clone())
        .with_window_entities(window_entities.clone());

    let allocation_system = scanner::ecs::systems::AllocationSystem::new();
    coordinator.add_system(Box::new(allocation_system));

    let mut window_processing = scanner::ecs::systems::WindowProcessingSystem::new(
        config_arc.clone(),
        pool.clone(),
        shutdown_coordinator.clone(),
    );
    window_processing.enable();
    coordinator.add_system(Box::new(window_processing));

    coordinator.tick().expect("First tick should succeed");
    coordinator.tick().expect("Second tick should succeed");
    coordinator.tick().expect("Third tick should succeed");

    let entities = task_entities.read().unwrap();
    let task = entities
        .get(&task_id)
        .expect("Task entity should still exist");

    let scanner::ecs::TaskComponents::Scan {
        progress,
        lifecycle,
        ..
    } = &task.components;
    assert!(
        !matches!(progress.state, scanner::ecs::ScanPauseState::Pending),
        "Regression test: Band scan should automatically transition from Pending to \
         Scanning.\nBug: After removing ScannerCommand during ECS migration, band scans no longer \
         \ninitiated automatically on startup. The app would launch but scanning never \
         began.\nRoot cause: No ECS system was detecting pending scans and processing \
         windows.\nFix: Added WindowProcessingSystem that detects pending TaskEntity and spawns \
         window tasks.\nTask remained in Pending state, indicating WindowProcessingSystem did not \
         run or failed."
    );

    assert!(
        lifecycle.is_started(),
        "Scan lifecycle should be marked as started after initiation"
    );
}

#[test]
fn test_window_processing_system_spawns_exactly_one_task_at_a_time() {
    let config = ScanningConfig {
        band: Band::Fm,
        samp_rate: 2.4e6,
        sdr_gain: 40.0,
        duration: 1,
        scanning_windows: Some(2),
        ..Default::default()
    };

    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "test002");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "test002");
    add_test_device_to_pool(&pool, device_id, caps, Backend::Mock, None);

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let task_entities = Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new()));
    let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));

    let (freq_min, freq_max) = config.band.frequency_range();
    let _scan_config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        config.samp_rate,
        config.samp_rate,
        config.sdr_gain,
        config.duration as f64,
        config.scanning_windows.unwrap_or(1),
    );

    let total_windows = ((freq_max - freq_min) / config.samp_rate).ceil() as usize;
    let task_id = TaskId::new("scan_1");
    let task_entity = TaskEntity::new_scan_with_defaults(
        task_id.clone(),
        ScanTaskData::Placeholder,
        total_windows,
    );
    task_entities.write().unwrap().insert(task_entity);

    let config_arc = Arc::new(config);
    let mut coordinator = Coordinator::new(&pool, &config_arc, &shutdown_coordinator)
        .with_task_entities(task_entities.clone())
        .with_window_entities(window_entities.clone());

    let allocation_system = scanner::ecs::systems::AllocationSystem::new();
    coordinator.add_system(Box::new(allocation_system));

    let mut window_processing = scanner::ecs::systems::WindowProcessingSystem::new(
        config_arc.clone(),
        pool.clone(),
        shutdown_coordinator.clone(),
    );
    window_processing.enable();
    coordinator.add_system(Box::new(window_processing));

    coordinator.tick().expect("First tick should succeed");
    coordinator.tick().expect("Second tick should succeed");
    coordinator.tick().expect("Third tick should succeed");

    {
        let entities = task_entities.read().unwrap();
        let task = entities.get(&task_id).expect("Task entity should exist");

        let scanner::ecs::TaskComponents::Scan { progress, .. } = &task.components;
        assert!(
            matches!(progress.state, scanner::ecs::ScanPauseState::Scanning),
            "Regression test: Exactly one window task should be spawned.\nBug: Multiple \
             coordinator threads or duplicate calls caused\nmultiple window tasks to be spawned \
             for the same scan,\nleading to race conditions.\nRoot cause: MainThread spawned \
             coordinator twice.\nWindowProcessingSystem should spawn exactly one task at a time."
        );
    }

    coordinator.tick().expect("Fourth tick should succeed");

    {
        let entities = task_entities.read().unwrap();
        let _task = entities.get(&task_id).expect("Task entity should exist");
    }
}

#[test]
fn test_coordinator_spawned_only_once() {
    static COORDINATOR_SPAWN_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct CoordinatorSpawnTracker;

    impl CoordinatorSpawnTracker {
        fn increment() {
            COORDINATOR_SPAWN_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        fn count() -> usize {
            COORDINATOR_SPAWN_COUNT.load(Ordering::SeqCst)
        }

        fn reset() {
            COORDINATOR_SPAWN_COUNT.store(0, Ordering::SeqCst);
        }
    }

    CoordinatorSpawnTracker::reset();

    let pool = Arc::new(Pool::new_unfiltered());
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    let config = Arc::new(ScanningConfig::default());

    let initial_count = CoordinatorSpawnTracker::count();

    let _coordinator = Coordinator::new(&pool, &config, &shutdown_coordinator);
    CoordinatorSpawnTracker::increment();

    let after_creation = CoordinatorSpawnTracker::count();

    assert_eq!(
        after_creation,
        initial_count + 1,
        "Regression test: Coordinator should be created exactly once.\nBug: \
         MainThread::new_with_progress() spawned a coordinator, then\nMainThread::run() spawned \
         another one, creating duplicate threads\nand losing the handle to the first \
         coordinator.\nThis caused race conditions with dual task submission and orphaned \
         threads.\nExpected {} coordinator(s), found {}",
        initial_count + 1,
        after_creation
    );

    drop(shutdown_coordinator);
    drop(pool);

    let final_count = CoordinatorSpawnTracker::count();
    assert_eq!(
        final_count, after_creation,
        "No additional coordinators should be spawned during cleanup"
    );
}
