use super::*;
use crate::audio::quality::AudioAnalyzer;
use crate::core::types::ScanningConfig;
use crate::hardware::pool::TunerActivity;
use crate::hardware::{Backend, DeviceId};
use std::sync::Arc;

fn create_test_config() -> ScanningConfig {
    let mut config = ScanningConfig::default();
    config.audio.buffer_size = 8192;
    config.audio.analyzer = AudioAnalyzer::mock();
    config.scanning_windows = Some(2);
    config.peak_detection.fft_size = 1024;
    config.peak_detection.scan_duration = 1.5;
    config
}

fn create_test_tuner_id() -> DeviceId {
    DeviceId::from_serial("mock", "test123")
}

fn create_test_backend() -> Arc<dyn crate::hardware::Backend> {
    Arc::new(crate::hardware::mock::Mock)
}

#[test]
fn test_main_thread_creation() {
    let config = create_test_config();
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let result = MainThread::new(Arc::new(config), backend, shutdown_coordinator);
    assert!(result.is_ok());
    let _main_thread = result.unwrap().start();
}

#[test]
fn test_main_thread_run_with_mock_tuner() {
    let config = create_test_config();

    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let _main_thread = MainThread::new(Arc::new(config), backend, shutdown_coordinator).unwrap();

    // Mock backend will fail when trying to open the device
    // This test just verifies MainThread construction works with new API
}

#[test]
fn test_pool_initialization() {
    let config = create_test_config();

    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(Arc::new(config), backend, shutdown_coordinator).unwrap();

    let pool_status = main_thread.pool.status();
    assert_eq!(pool_status.device_count, 0, "Pool should start empty");
    assert_eq!(
        pool_status.available_tuner_count, 0,
        "Pool should have no tuners initially"
    );
}

#[test]
fn test_pool_shutdown_on_drop() {
    let config = create_test_config();

    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(Arc::new(config), backend, shutdown_coordinator).unwrap();

    let pool_clone = Arc::clone(&main_thread.pool);
    assert!(!pool_clone.is_shutdown(), "Pool should not be shutdown");

    drop(main_thread);

    assert!(
        pool_clone.is_shutdown(),
        "Pool should be shutdown after MainThread drop"
    );
}

#[test]
fn test_pool_device_population() {
    let tuner_id = DeviceId::from_serial("mock", "12345");

    let filter = PoolFilter::new()
        .with_driver("mock")
        .with_mode(TuningMode::SingleTuner);
    let pool = Pool::new(filter, None);

    let mock_backend = crate::hardware::Mock;
    let pool_tuner_id = crate::hardware::pool::TunerId::new(tuner_id, 0);
    let device = mock_backend.open_tuner(&pool_tuner_id).unwrap();

    pool.add_device(device, crate::hardware::types::Backend::Mock)
        .unwrap();

    let status = pool.status();
    assert_eq!(status.device_count, 1, "Pool should have one device");
    assert_eq!(status.available_tuner_count, 1, "Mock device has 1 tuner");
}

#[test]
fn test_pool_acquire_and_use() {
    let tuner_id = DeviceId::from_serial("mock", "12345");

    let filter = PoolFilter::new()
        .with_driver("mock")
        .with_mode(TuningMode::SingleTuner);
    let pool = Pool::new(filter, None);

    let mock_backend = crate::hardware::Mock;
    let pool_tuner_id = crate::hardware::pool::TunerId::new(tuner_id, 0);
    let device = mock_backend.open_tuner(&pool_tuner_id).unwrap();
    pool.add_device(device, crate::hardware::types::Backend::Mock)
        .unwrap();

    let pool = Arc::new(pool);

    // Acquire tuner from pool
    let requirements = crate::hardware::pool::TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200_000.0,
        required_sample_rate: 2_400_000.0,
        priority: crate::hardware::pool::TaskPriority::Normal,
    };

    let pooled_tuner = pool
        .acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    // Verify tuner was acquired
    let status = pool.status();
    assert_eq!(status.available_tuner_count, 0, "Tuner should be allocated");
    assert_eq!(status.allocated_tuner_count, 1, "One tuner allocated");

    // Use the tuner to add source to graph
    let mut graph = rustradio::graph::Graph::new();
    let _stream = pooled_tuner
        .add_source_to_graph(&mut graph, 88.9e6, 2_400_000.0, 40.0)
        .unwrap();

    // Drop tuner - should return to pool automatically
    drop(pooled_tuner);

    let status = pool.status();
    assert_eq!(
        status.available_tuner_count, 1,
        "Tuner should be returned to pool"
    );
    assert_eq!(status.allocated_tuner_count, 0, "No tuners allocated");
}

#[test]
fn test_coordinator_thread_lifecycle() {
    use std::sync::atomic::Ordering;

    let config = create_test_config();

    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(Arc::new(config), backend, shutdown_coordinator)
        .unwrap()
        .start();

    assert!(
        main_thread.coordinator_handle.is_some(),
        "Coordinator should be spawned after start()"
    );

    std::thread::sleep(std::time::Duration::from_millis(250));

    assert!(
        !main_thread.coordinator_shutdown.load(Ordering::SeqCst),
        "Coordinator should still be running"
    );

    drop(main_thread);
}

#[test]
fn test_coordinator_shutdown_on_drop() {
    use std::sync::atomic::Ordering;

    let config = create_test_config();

    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(Arc::new(config), backend, shutdown_coordinator).unwrap();

    let shutdown_flag = Arc::clone(&main_thread.coordinator_shutdown);

    std::thread::sleep(std::time::Duration::from_millis(50));

    assert!(
        !shutdown_flag.load(Ordering::SeqCst),
        "Coordinator should be running"
    );

    drop(main_thread);

    assert!(
        shutdown_flag.load(Ordering::SeqCst),
        "Coordinator should be shut down after drop"
    );
}

#[test]
fn test_tui_event_sender_wired_to_ui_update_system() {
    use crate::ecs::Coordinator;
    use crate::hardware::types::{Backend, Capabilities};

    let (tui_sender, tui_receiver) = std::sync::mpsc::channel();

    let pool = Arc::new(crate::hardware::pool::Pool::new(
        crate::hardware::pool::PoolFilter::new().with_driver("sdrplay"),
        None,
    ));

    let config = Arc::new(crate::core::types::ScanningConfig::default());
    let shutdown = Arc::new(crate::shutdown::ShutdownCoordinator::new());

    let scan_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));
    let station_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));
    let audio_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));
    let candidate_entities = Arc::new(RwLock::new(crate::ecs::EntityWorld::new()));

    let device_id = DeviceId::from_driver(Backend::Soapy, "sdrplay", "test123");
    let tuner_id = crate::hardware::pool::TunerId {
        device_id: device_id.clone(),
        channel_index: 0,
    };

    {
        let mut entities = pool.tuner_entities.lock().unwrap();
        entities.insert(crate::ecs::TunerEntity::new(
            device_id,
            0,
            Capabilities::for_mock("sdrplay", "test123"),
            Backend::Soapy,
        ));
    }

    let mut coordinator = Coordinator::new(&pool, &config, &shutdown)
        .with_scan_entities(scan_entities)
        .with_station_entities(station_entities)
        .with_audio_entities(audio_entities)
        .with_candidate_entities(candidate_entities);

    let mut ui_update_system = crate::ecs::systems::UIUpdateSystem::new();
    ui_update_system = ui_update_system.with_tui_event_sender(tui_sender);
    coordinator.add_system(Box::new(ui_update_system));

    {
        let mut entities = pool.tuner_entities.lock().unwrap();
        let entity_id = tuner_id.clone();
        let tuner = entities.get_mut(&entity_id).unwrap();
        tuner.allocation.allocate("test_requester".to_string());
        tuner.status.activity = crate::ecs::components::TunerActivity::Scanning;
    }

    coordinator.tick().unwrap();

    let mut received_scanning_event = false;
    while let Ok(event) = tui_receiver.try_recv() {
        if let crate::ui::TuiEvent::ActiveTunersUpdated { status } = event {
            for tuner in &status.tuners {
                if tuner.id == tuner_id && tuner.activity == Some(TunerActivity::Scanning) {
                    received_scanning_event = true;
                    break;
                }
            }
        }
    }

    assert!(
        received_scanning_event,
        "UIUpdateSystem should send ActiveTunersUpdated events with Scanning activity when TUI event sender is configured"
    );
}
