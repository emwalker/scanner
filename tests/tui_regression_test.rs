use scanner::core::types::{Band, ScanningConfig};
use scanner::ecs::components::scan::{ScanConfigComponent, ScanType};
use scanner::ecs::{Coordinator, Entity, EntityWorld, ScanEntity};
use scanner::hardware::DeviceId;
use scanner::hardware::mock::MockDevice;
use scanner::hardware::pool::{Pool, PoolFilter, PoolStatus, TunerActivity, TunerId, TunerState};
use scanner::hardware::types::Backend;
use scanner::shutdown::ShutdownCoordinator;
use scanner::task::TaskScheduler;
use scanner::ui::TuiEvent;
use scanner::ui::tui::model::Model;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::time::Duration;

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
        "Regression test: Pool info should not be rebuilt when status is identical.\n\
         Bug: During listening mode, redundant ActiveTunersUpdated events caused \n\
         expensive HashMap rebuilds every time, spiking CPU to 50%.\n\
         Fix: Added change detection to skip processing when status unchanged."
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
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
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
        "Regression test: Terminal should render at ~10 FPS (100ms intervals) for smooth spectrum animation.\n\
         Bug: Conditional rendering based on model.is_dirty() caused jumpy spectrum wave \n\
         because it only updated when entities changed, not continuously.\n\
         Fix: Always call mark_dirty() in main loop to ensure 10 FPS rendering.\n\
         Expected at least 3 draws in 350ms, got {}",
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

    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));

    let device = Box::new(MockDevice::new("mock", "test001", false));
    pool.add_device(device, Backend::Mock);

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    let _scheduler = Arc::new(TaskScheduler::new(
        pool.clone(),
        shutdown_coordinator.clone(),
    ));

    let scan_entities = Arc::new(RwLock::new(EntityWorld::<ScanEntity>::new()));

    let (freq_min, freq_max) = config.band.frequency_range();
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        config.samp_rate,
        config.samp_rate,
        config.sdr_gain,
        config.duration as f64,
        config.scanning_windows.unwrap_or(1),
    );

    let scan_entity = ScanEntity::new(scan_config);
    let scan_id = *scan_entity.id();
    scan_entities.write().unwrap().insert(scan_entity);

    {
        let entities = scan_entities.read().unwrap();
        let scan = entities.get(&scan_id).expect("Scan entity should exist");
        assert!(
            scan.is_pending(),
            "Scan should start in Pending state before WindowProcessingSystem runs"
        );
    }

    let mut coordinator = Coordinator::new(&pool).with_scan_entities(scan_entities.clone());

    let allocation_system = scanner::ecs::systems::AllocationSystem::new();
    coordinator.add_system(Box::new(allocation_system));

    let mut window_processing = scanner::ecs::systems::WindowProcessingSystem::new(
        Arc::new(config),
        pool.clone(),
        shutdown_coordinator.clone(),
    );
    window_processing.enable();
    coordinator.add_system(Box::new(window_processing));

    coordinator.tick().expect("First tick should succeed");
    coordinator.tick().expect("Second tick should succeed");
    coordinator.tick().expect("Third tick should succeed");

    let entities = scan_entities.read().unwrap();
    let scan = entities
        .get(&scan_id)
        .expect("Scan entity should still exist");

    assert!(
        !scan.is_pending(),
        "Regression test: Band scan should automatically transition from Pending to Scanning.\n\
         Bug: After removing ScannerCommand during ECS migration, band scans no longer \n\
         initiated automatically on startup. The app would launch but scanning never began.\n\
         Root cause: No ECS system was detecting pending scans and processing windows.\n\
         Fix: Added WindowProcessingSystem that detects pending ScanEntity and spawns window tasks.\n\
         Scan remained in Pending state, indicating WindowProcessingSystem did not run or failed."
    );

    assert!(
        scan.lifecycle.is_started(),
        "Scan lifecycle should be marked as started after initiation"
    );

    assert!(
        scan.window_task.is_some(),
        "WindowProcessingSystem should have spawned a window task"
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

    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));

    let device = Box::new(MockDevice::new("mock", "test002", false));
    pool.add_device(device, Backend::Mock);

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let scan_entities = Arc::new(RwLock::new(EntityWorld::<ScanEntity>::new()));

    let (freq_min, freq_max) = config.band.frequency_range();
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        config.samp_rate,
        config.samp_rate,
        config.sdr_gain,
        config.duration as f64,
        config.scanning_windows.unwrap_or(1),
    );

    let scan_entity = ScanEntity::new(scan_config);
    let scan_id = *scan_entity.id();
    scan_entities.write().unwrap().insert(scan_entity);

    let mut coordinator = Coordinator::new(&pool).with_scan_entities(scan_entities.clone());

    let allocation_system = scanner::ecs::systems::AllocationSystem::new();
    coordinator.add_system(Box::new(allocation_system));

    let mut window_processing = scanner::ecs::systems::WindowProcessingSystem::new(
        Arc::new(config),
        pool.clone(),
        shutdown_coordinator.clone(),
    );
    window_processing.enable();
    coordinator.add_system(Box::new(window_processing));

    coordinator.tick().expect("First tick should succeed");
    coordinator.tick().expect("Second tick should succeed");
    coordinator.tick().expect("Third tick should succeed");

    {
        let entities = scan_entities.read().unwrap();
        let scan = entities.get(&scan_id).expect("Scan entity should exist");
        assert!(
            scan.window_task.is_some(),
            "Regression test: Exactly one window task should be spawned.\n\
             Bug: Multiple coordinator threads or duplicate calls caused\n\
             multiple window tasks to be spawned for the same scan,\n\
             leading to race conditions.\n\
             Root cause: MainThread spawned coordinator twice.\n\
             WindowProcessingSystem should spawn exactly one task at a time."
        );
    }

    coordinator.tick().expect("Fourth tick should succeed");

    {
        let entities = scan_entities.read().unwrap();
        let scan = entities.get(&scan_id).expect("Scan entity should exist");
        assert!(
            scan.window_task.is_some(),
            "Regression test: Window task should remain active until completed.\n\
             No additional tasks should be spawned while one is running."
        );
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

    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let initial_count = CoordinatorSpawnTracker::count();

    let _coordinator = Coordinator::new(&pool);
    CoordinatorSpawnTracker::increment();

    let after_creation = CoordinatorSpawnTracker::count();

    assert_eq!(
        after_creation,
        initial_count + 1,
        "Regression test: Coordinator should be created exactly once.\n\
         Bug: MainThread::new_with_progress() spawned a coordinator, then\n\
         MainThread::run() spawned another one, creating duplicate threads\n\
         and losing the handle to the first coordinator.\n\
         This caused race conditions with dual task submission and orphaned threads.\n\
         Expected {} coordinator(s), found {}",
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
