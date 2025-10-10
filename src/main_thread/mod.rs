mod audio_coordinator;
mod commands;
mod state_manager;

use crate::audio::session::AudioSession;
use crate::core::types::{ConsoleWriter, Logger, Result, ScannerError, ScanningConfig};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::scanner_state::{PauseSignal, ScannerState};
use crate::scanning::window::Window;
use crate::shutdown::ShutdownCoordinator;
use crate::signal;
use crate::ui::{NoOpProgressReporter, ProgressReporter, ScannerCommand, TuiEvent};
use audio_coordinator::TuneParams;
use commands::CommandHandler;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use tracing::{debug, info};

pub struct MainThread {
    config: ScanningConfig,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    _backend: Arc<dyn crate::hardware::Backend>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    scanner_state: ScannerState,
    pause_signal: PauseSignal,
    current_playing: Option<TuneParams>,
    pool: Arc<Pool>,
}

impl MainThread {
    pub fn new(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        backend: Arc<dyn crate::hardware::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let filter = PoolFilter::new()
            .with_driver("sdrplay")
            .with_mode(TuningMode::SingleTuner);
        let pool = Pool::new(filter);

        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            progress_reporter: Arc::new(NoOpProgressReporter),
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            pool: Arc::new(pool),
        })
    }

    pub fn new_with_progress(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        backend: Arc<dyn crate::hardware::Backend>,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
        pool: Arc<Pool>,
    ) -> Result<Self> {
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            _backend: backend,
            progress_reporter,
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            pool,
        })
    }

    pub fn with_command_receiver(mut self, receiver: Receiver<ScannerCommand>) -> Self {
        self.command_receiver = Some(receiver);
        self
    }

    pub fn with_tui_event_sender(mut self, sender: Sender<TuiEvent>) -> Self {
        self.tui_event_sender = Some(sender);
        self
    }

    fn send_active_tuners_update(&self) {
        if let Some(ref sender) = self.tui_event_sender {
            let status = self.pool.status();
            let event = TuiEvent::ActiveTunersUpdated { status };
            let _ = sender.send(event);
        }
    }

    pub fn run(mut self, stations: Option<String>) -> Result<()> {
        // Logging is now initialized in main() before SDR operations
        // Pool is already populated with initial device by scanner.rs

        // Verify pool is populated
        let pool_status = self.pool.status();
        debug!(
            device_count = pool_status.device_count,
            available_tuners = pool_status.available_tuner_count,
            "Pool status at startup"
        );

        self.console_writer.write_info("Scanning for stations ...");

        // Send initial active tuners state (tuner is scanning)
        self.send_active_tuners_update();

        if let Some(stations_str) = stations {
            self.scan_stations(&stations_str)?;
        } else {
            self.scan_band()?;
        }

        self.console_writer.write_info("Scan complete.");
        Ok(())
    }

    fn parse_stations(&self, stations_str: &str) -> Result<Vec<f64>> {
        stations_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
            .collect()
    }

    fn scan_stations(&self, stations_str: &str) -> Result<()> {
        let stations = self.parse_stations(stations_str)?;
        debug!(
            message = "Scanning stations",
            stations = format!("{:?}", stations)
        );
        let _total_stations = stations.len();

        // Create a separate window for each station, using the station frequency as center frequency
        for (station_idx, station_freq) in stations.into_iter().enumerate() {
            debug!(
                "Processing station {} of {} at {:.1} MHz",
                station_idx + 1,
                _total_stations,
                station_freq / 1e6
            );

            // Create a window for this specific station frequency (pool-based)
            let window = Window::for_station(
                station_freq,
                station_idx + 1,
                _total_stations,
                self.pool.clone(),
                self.config.clone(),
                self.progress_reporter.clone(),
                self.shutdown_coordinator.clone(),
            );

            // Process using pool-based flow
            window.process_with_pool()?;
        }

        Ok(())
    }

    fn process_commands(
        &mut self,
        window_num: usize,
        _total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        let mut commands = Vec::new();
        if let Some(receiver) = &self.command_receiver {
            while let Ok(command) = receiver.try_recv() {
                commands.push(command);
            }
        }

        for command in commands {
            let mut handler = CommandHandler::new(commands::CommandHandlerConfig {
                scanner_state: &mut self.scanner_state,
                pause_signal: &self.pause_signal,
                pool: &self.pool,
                config: &self.config,
                shutdown_coordinator: &self.shutdown_coordinator,
                progress_reporter: &self.progress_reporter,
                tui_event_sender: &self.tui_event_sender,
                current_playing: &mut self.current_playing,
            });
            handler.handle_command(command, window_num, audio_session)?;
        }
        Ok(())
    }

    fn check_and_handle_command(
        &mut self,
        window_num: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        if let Some(receiver) = &self.command_receiver
            && let Ok(command) = receiver.try_recv()
        {
            let mut handler = CommandHandler::new(commands::CommandHandlerConfig {
                scanner_state: &mut self.scanner_state,
                pause_signal: &self.pause_signal,
                pool: &self.pool,
                config: &self.config,
                shutdown_coordinator: &self.shutdown_coordinator,
                progress_reporter: &self.progress_reporter,
                tui_event_sender: &self.tui_event_sender,
                current_playing: &mut self.current_playing,
            });
            handler.handle_command(command, window_num, audio_session)?;
        }
        Ok(())
    }

    fn process_commands_with_pause_check(
        &mut self,
        window_num: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<bool> {
        self.process_commands(window_num, total_windows, audio_session)?;

        if self.scanner_state.is_paused() {
            return Ok(true);
        }

        self.check_and_handle_command(window_num, audio_session)?;
        Ok(self.scanner_state.is_paused())
    }

    fn scan_band(&mut self) -> Result<()> {
        signal::clear_processed_frequencies();

        let window_centers = self.config.band.windows(
            self.config.samp_rate,
            self.config.signal_processing.window_overlap,
        );
        debug!(
            "Scanning {} windows across {:?} band",
            window_centers.len(),
            self.config.band
        );

        let windows_to_process = match self.config.scanning_windows {
            Some(n) => n.min(window_centers.len()),
            None => window_centers.len(),
        };

        let mut i: usize = 0;
        let mut audio_session: Option<AudioSession> = None;

        loop {
            if self.shutdown_coordinator.is_shutdown() {
                self.scanner_state.shutdown();
            }

            let control = match &self.scanner_state.mode {
                crate::scanner_state::ScanMode::ShuttingDown => {
                    debug!("Shutdown requested, stopping band scanning");
                    state_manager::LoopControl::Break
                }
                crate::scanner_state::ScanMode::ScanComplete { .. } => {
                    self.check_and_handle_command(windows_to_process, &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::ScanCompletePaused { .. } => {
                    self.process_commands(
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Paused { .. } => {
                    if !i.is_multiple_of(50) {
                        debug!(
                            iteration = i,
                            total = windows_to_process,
                            "Paused - waiting for commands"
                        );
                    }
                    self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Listening { .. } => {
                    self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Scanning => {
                    if i >= windows_to_process {
                        debug!("Scan band complete - all windows processed");
                        self.scanner_state.mark_scan_complete(windows_to_process);
                        state_manager::LoopControl::Continue
                    } else {
                        debug!(
                            iteration = i,
                            total = windows_to_process,
                            "Start of scan loop iteration"
                        );

                        if self.process_commands_with_pause_check(
                            i + 1,
                            window_centers.len(),
                            &mut audio_session,
                        )? {
                            state_manager::LoopControl::Continue
                        } else {
                            let center_freq = window_centers[i];
                            self.process_window(i + 1, center_freq, window_centers.len())?;
                            self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;

                            debug!(
                                completed_window = i + 1,
                                next_window = i + 2,
                                remaining = windows_to_process - i - 1,
                                "Window complete, advancing to next"
                            );

                            state_manager::LoopControl::Advance
                        }
                    }
                }
            };

            match control {
                state_manager::LoopControl::Break => break,
                state_manager::LoopControl::Continue => continue,
                state_manager::LoopControl::Advance => i += 1,
            }
        }

        Ok(())
    }

    fn process_window(
        &self,
        window_num: usize,
        center_freq: f64,
        total_windows: usize,
    ) -> Result<()> {
        debug!(
            window = window_num,
            total = total_windows,
            "Processing window"
        );

        let window = Window::new(crate::scanning::window::WindowConfig {
            center_freq,
            window_num,
            total_windows,
            tuner_provider: self.pool.clone(),
            config: self.config.clone(),
            progress_reporter: self.progress_reporter.clone(),
            shutdown_coordinator: self.shutdown_coordinator.clone(),
            pause_signal: Some(self.pause_signal.clone()),
        });

        window.process_with_pool()?;

        debug!(
            completed_window = window_num,
            next_window = window_num + 1,
            "Window complete"
        );
        Ok(())
    }
}

// Default implementations for production use
pub struct DefaultConsoleWriter;

impl ConsoleWriter for DefaultConsoleWriter {
    fn write_info(&self, message: &str) {
        info!("{}", message);
    }

    fn write_debug(&self, message: &str) {
        debug!("{}", message);
    }
}

impl Drop for MainThread {
    fn drop(&mut self) {
        self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests {
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
            config,
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
            config,
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
            config,
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
            config,
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
            config,
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
            config,
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
            config,
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
        let pool = Pool::new(filter);

        let mock_backend = crate::hardware::Mock;
        let device = mock_backend.open_device(&tuner_id).unwrap();

        pool.add_device(device, "Mock".to_string()).unwrap();

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
        let pool = Pool::new(filter);

        let mock_backend = crate::hardware::Mock;
        let device = mock_backend.open_device(&tuner_id).unwrap();
        pool.add_device(device, "Mock".to_string()).unwrap();

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
}
