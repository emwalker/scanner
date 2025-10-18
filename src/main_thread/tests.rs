use super::*;
use crate::audio::quality::AudioAnalyzer;
use crate::core::types::ScanningConfig;
use crate::hardware::pool::TunerActivity;
use crate::hardware::{Backend, DeviceId};
use std::sync::{Arc, Mutex};

// Mock implementations for testing
#[derive(Default)]
pub struct MockConsoleWriter {
    messages: Arc<Mutex<Vec<String>>>,
}

impl MockConsoleWriter {
    pub fn new() -> Self {
        Self {
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn messages(&self) -> Vec<String> {
        self.messages.lock().unwrap().clone()
    }
}

impl ConsoleWriter for MockConsoleWriter {
    fn write_info(&self, message: &str) {
        self.messages
            .lock()
            .unwrap()
            .push(format!("INFO: {}", message));
    }

    fn write_debug(&self, message: &str) {
        self.messages
            .lock()
            .unwrap()
            .push(format!("DEBUG: {}", message));
    }
}

pub struct MockLogger {
    init_called: Arc<Mutex<bool>>,
}

impl MockLogger {
    pub fn new() -> Self {
        Self {
            init_called: Arc::new(Mutex::new(false)),
        }
    }
}

impl Logger for MockLogger {
    fn init(&self) -> Result<()> {
        *self.init_called.lock().unwrap() = true;
        Ok(())
    }
}

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
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    );
    assert!(main_thread.is_ok());
}

#[test]
fn test_main_thread_run_with_mock_tuner() {
    let config = create_test_config();
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let _main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    // Mock backend will fail when trying to open the device
    // This test just verifies MainThread construction works with new API
}

#[test]
fn test_console_output() {
    let config = create_test_config();
    let console_writer = Arc::new(MockConsoleWriter::new());
    let console_clone = Arc::clone(&console_writer);
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    // This would normally call SoapySDR and process windows, but we can test the console output pattern
    main_thread.console_writer.write_info("Test message");

    let messages = console_clone.messages();
    assert_eq!(messages, vec!["INFO: Test message"]);
}

#[test]
fn test_parse_stations() {
    let config = create_test_config();
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    let stations = main_thread
        .parse_stations("88.9e6,101.5e6,107.3e6")
        .unwrap();
    assert_eq!(stations, vec![88.9e6, 101.5e6, 107.3e6]);
}

#[test]
fn test_parse_stations_invalid() {
    let config = create_test_config();
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    let result = main_thread.parse_stations("88.9e6,invalid,107.3e6");
    assert!(result.is_err());
}

#[test]
fn test_pool_initialization() {
    let config = create_test_config();
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

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
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let _tuner_id = create_test_tuner_id();
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

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
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    assert!(
        main_thread.coordinator_handle.is_none(),
        "Coordinator should not be spawned yet"
    );

    main_thread.spawn_coordinator();

    assert!(
        main_thread.coordinator_handle.is_some(),
        "Coordinator should be spawned"
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
    let console_writer = Arc::new(MockConsoleWriter::new());
    let logger = Arc::new(MockLogger::new());
    let backend = create_test_backend();
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator,
    )
    .unwrap();

    let shutdown_flag = Arc::clone(&main_thread.coordinator_shutdown);

    main_thread.spawn_coordinator();
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
fn test_worker_channels_creation() {
    let (channels, handle) = WorkerChannels::new();

    assert!(channels.event_rx.try_recv().is_err(), "No events yet");

    handle
        .event_tx
        .send(WorkerEvent::ScanStarted {
            scan_id: ScanId::new(),
        })
        .unwrap();

    assert!(channels.event_rx.try_recv().is_ok(), "Should receive event");

    channels.command_tx.send(WorkerCommand::PauseScan).unwrap();

    assert!(
        handle.command_rx.try_recv().is_ok(),
        "Should receive command"
    );
}
