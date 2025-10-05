use crate::scanner_state::{PauseSignal, ScannerState};
use crate::sdr::Device;
use crate::shutdown::ShutdownCoordinator;
use crate::terminal::{NoOpProgressReporter, ProgressReporter};
use crate::types::{ConsoleWriter, Logger, Result, ScannerError, ScanningConfig};
use crate::window::Window;
use crate::{fm, soapy};
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Clone)]
struct TuneParams {
    candidate_id: String,
    window_id: usize,
    center_frequency: f64,
    candidate_frequency: f64,
    signal_strength: Option<f64>,
    audio_quality: Option<crate::audio_quality::AudioQuality>,
}

/// Tracks which tuners are available and actively allocated to different functions
#[derive(Clone, Debug)]
pub struct ActiveTuners {
    /// Pool of tuners that can be allocated (relatively static, changes with discovery)
    pub available: Vec<crate::sdr::TunerId>,
    /// Tuners currently allocated to scanning
    pub scanning: Vec<crate::sdr::TunerId>,
    /// Tuners currently allocated to listening
    pub listening: Vec<crate::sdr::TunerId>,
}

impl ActiveTuners {
    fn new(selected_tuner_id: crate::sdr::TunerId) -> Self {
        Self {
            available: vec![selected_tuner_id.clone()],
            scanning: vec![selected_tuner_id],
            listening: vec![],
        }
    }
}

pub struct MainThread {
    config: ScanningConfig,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    selected_tuner_id: crate::sdr::TunerId,
    _backend: Arc<dyn crate::sdr::Backend>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    command_receiver: Option<std::sync::mpsc::Receiver<crate::terminal::ScannerCommand>>,
    tui_event_sender: Option<std::sync::mpsc::Sender<crate::terminal::TuiEvent>>,
    scanner_state: ScannerState,
    pause_signal: PauseSignal,
    current_playing: Option<TuneParams>,
    active_tuners: ActiveTuners,
}

impl MainThread {
    pub fn new(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        selected_tuner_id: crate::sdr::TunerId,
        backend: Arc<dyn crate::sdr::Backend>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let active_tuners = ActiveTuners::new(selected_tuner_id.clone());
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            selected_tuner_id,
            _backend: backend,
            progress_reporter: Arc::new(NoOpProgressReporter),
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            active_tuners,
        })
    }

    pub fn new_with_progress(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        selected_tuner_id: crate::sdr::TunerId,
        backend: Arc<dyn crate::sdr::Backend>,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let active_tuners = ActiveTuners::new(selected_tuner_id.clone());
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            selected_tuner_id,
            _backend: backend,
            progress_reporter,
            shutdown_coordinator,
            command_receiver: None,
            tui_event_sender: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
            current_playing: None,
            active_tuners,
        })
    }

    pub fn with_command_receiver(
        mut self,
        receiver: std::sync::mpsc::Receiver<crate::terminal::ScannerCommand>,
    ) -> Self {
        self.command_receiver = Some(receiver);
        self
    }

    pub fn with_tui_event_sender(
        mut self,
        sender: std::sync::mpsc::Sender<crate::terminal::TuiEvent>,
    ) -> Self {
        self.tui_event_sender = Some(sender);
        self
    }

    fn send_active_tuners_update(&self) {
        if let Some(ref sender) = self.tui_event_sender {
            let event = crate::terminal::TuiEvent::ActiveTunersUpdated {
                available: self.active_tuners.available.clone(),
                scanning: self.active_tuners.scanning.clone(),
                listening: self.active_tuners.listening.clone(),
            };
            let _ = sender.send(event);
        }
    }

    pub fn run(mut self, stations: Option<String>) -> Result<()> {
        // Logging is now initialized in main() before SDR operations

        // Open the selected device
        let device = self.open_device()?;
        self.console_writer.write_info("Scanning for stations ...");

        // Send initial active tuners state (tuner is scanning)
        self.send_active_tuners_update();

        if let Some(stations_str) = stations {
            self.scan_stations(&device, &stations_str)?;
        } else {
            self.scan_band(&device)?;
        }

        self.console_writer.write_info("Scan complete.");
        Ok(())
    }

    fn open_device(&self) -> Result<soapy::Device> {
        // For now, convert the new TunerId back to the old soapy::Device format
        // This maintains compatibility with existing code while we migrate
        let device_string = match &self.selected_tuner_id {
            crate::sdr::TunerId::Backend { backend, serial } => {
                // Handle RSPduo format: serial can be "1234:master" or just "1234"
                if let Some((actual_serial, mode)) = serial.split_once(':') {
                    format!("driver={},serial={},mode={}", backend, actual_serial, mode)
                } else {
                    format!("driver={},serial={}", backend, serial)
                }
            }
            crate::sdr::TunerId::Usb { .. } => {
                return Err(crate::types::ScannerError::Custom(
                    "USB device IDs not supported for legacy soapy::Device".to_string(),
                ));
            }
        };
        Ok(soapy::Device(device_string))
    }

    fn parse_stations(&self, stations_str: &str) -> Result<Vec<f64>> {
        stations_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
            .collect()
    }

    fn handle_tune_command(
        &self,
        device: &soapy::Device,
        audio_session: &mut crate::audio_session::AudioSession,
        params: TuneParams,
    ) -> Result<()> {
        debug!(
            candidate_id = ?params.candidate_id,
            candidate_mhz = params.candidate_frequency / 1e6,
            center_mhz = params.center_frequency / 1e6,
            signal_strength = ?params.signal_strength,
            audio_quality = ?params.audio_quality,
            "Tuning to candidate"
        );

        let segment = device.tune(&self.config, params.center_frequency)?;

        let signal = crate::types::Signal {
            frequency_hz: params.candidate_frequency,
            signal_strength: params.signal_strength.unwrap_or(0.1) as f32,
            bandwidth_hz: 200_000.0,
            modulation: crate::types::ModulationType::WFM,
            audio_sample_rate: self.config.audio_sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 0,
            detection_center_freq: params.center_frequency,
            audio_quality: params
                .audio_quality
                .unwrap_or(crate::audio_quality::AudioQuality::Unknown),
        };

        tracing::info!(
            "playing {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );

        audio_session.tune_to_station(&signal, segment, &self.config)?;

        debug!(
            candidate_id = ?params.candidate_id,
            event_type = "AudioPlaybackStarted",
            "MainThread: Sending AudioPlaybackStarted event to TUI"
        );

        self.progress_reporter
            .report(crate::terminal::ProgressEvent {
                event_type: crate::terminal::ProgressEventType::AudioPlaybackStarted,
                frequency_hz: params.candidate_frequency,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: params.center_frequency,
                    window_id: params.window_id,
                },
                candidate_id: Some(params.candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
                tuner_id: device.tuner_id().ok(),
            });

        Ok(())
    }

    fn handle_command(
        &mut self,
        command: crate::terminal::ScannerCommand,
        device: &soapy::Device,
        window_num: usize,
        _total_windows: usize,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<Option<crate::terminal::ScannerCommand>> {
        match command {
            crate::terminal::ScannerCommand::Pause => {
                debug!(window = window_num, "Scanner paused, creating AudioSession");
                self.pause_signal.pause();
                self.scanner_state.handle_pause(window_num);

                *audio_session = Some(crate::audio_session::AudioSession::new(
                    &self.config,
                    self.shutdown_coordinator.clone(),
                )?);
                debug!("AudioSession created for browse mode");

                // Send Paused event to TUI so it knows scanning has stopped and can now tune
                if let Some(ref sender) = self.tui_event_sender {
                    let tuner_id = device
                        .tuner_id()
                        .ok()
                        .unwrap_or_else(|| crate::sdr::TunerId::from_serial("unknown", "0"));
                    let _ = sender.send(crate::terminal::TuiEvent::Paused { tuner_id });
                }

                Ok(None)
            }
            crate::terminal::ScannerCommand::ResumeScan => {
                debug!(
                    window = window_num,
                    "Scanner resuming - exiting selection mode and continuing scan"
                );
                self.pause_signal.unpause();
                let _next_window = self.scanner_state.handle_resume();

                // Move tuner from listening back to scanning
                self.active_tuners.listening.clear();
                self.active_tuners.scanning = self.active_tuners.available.clone();
                self.send_active_tuners_update();

                *audio_session = None;
                debug!("AudioSession dropped, returning to scan mode");

                Ok(None)
            }
            crate::terminal::ScannerCommand::TuneToCandidate {
                candidate_id,
                window_id,
                center_frequency,
                candidate_frequency,
                signal_strength,
                audio_quality,
            } => {
                debug!(
                    candidate_id = ?candidate_id,
                    window_id = window_id,
                    candidate_frequency_mhz = candidate_frequency / 1e6,
                    "MainThread: Received TuneToCandidate command"
                );
                self.scanner_state.handle_tune(window_num);

                // Move tuner from scanning to listening
                self.active_tuners.scanning.clear();
                self.active_tuners.listening = self.active_tuners.available.clone();
                self.send_active_tuners_update();

                if let Some(session) = audio_session {
                    let params = TuneParams {
                        candidate_id,
                        window_id,
                        center_frequency,
                        candidate_frequency,
                        signal_strength,
                        audio_quality,
                    };
                    self.handle_tune_command(device, session, params.clone())?;
                    self.current_playing = Some(params);
                    Ok(None)
                } else {
                    debug!("TuneToCandidate received but no AudioSession exists");
                    Ok(None)
                }
            }
            crate::terminal::ScannerCommand::StopListening => {
                debug!("Stopped listening, returning to browsing mode");
                self.scanner_state.handle_stop_listening();

                if let Some(session) = audio_session {
                    session.stop_current_station();
                }

                // Send AudioPlaybackCompleted event if we were playing something
                if let Some(params) = self.current_playing.take() {
                    self.progress_reporter
                        .report(crate::terminal::ProgressEvent {
                            event_type: crate::terminal::ProgressEventType::AudioPlaybackCompleted,
                            frequency_hz: params.candidate_frequency,
                            metadata: crate::window::WindowMetadata {
                                center_frequency_hz: params.center_frequency,
                                window_id: params.window_id,
                            },
                            candidate_id: Some(params.candidate_id),
                            audio_quality: params.audio_quality,
                            signal_strength: params.signal_strength,
                            timestamp: std::time::Instant::now(),
                            tuner_id: device.tuner_id().ok(),
                        });
                }

                Ok(None)
            }
        }
    }

    fn scan_stations(&self, device: &soapy::Device, stations_str: &str) -> Result<()> {
        let stations = self.parse_stations(stations_str)?;
        debug!(
            message = "Scanning stations",
            stations = format!("{:?}", stations)
        );
        let total_stations = stations.len();

        // Create a separate window for each station, using the station frequency as center frequency
        for (station_idx, station_freq) in stations.into_iter().enumerate() {
            debug!(
                "Processing station {} of {} at {:.1} MHz",
                station_idx + 1,
                total_stations,
                station_freq / 1e6
            );

            // Create a window for this specific station frequency
            let segment = device.tune(&self.config, station_freq)?;
            let window = Window::for_station(
                station_freq,
                station_idx + 1,
                total_stations,
                device.clone(),
                self.config.clone(),
                self.progress_reporter.clone(),
                self.shutdown_coordinator.clone(),
            );

            // Process using the full band scanning pipeline (peak detection, candidates, etc.)
            window.process(&*segment)?;
        }

        Ok(())
    }

    fn process_commands(
        &mut self,
        device: &soapy::Device,
        window_num: usize,
        total_windows: usize,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<()> {
        let mut commands = Vec::new();
        if let Some(receiver) = &self.command_receiver {
            while let Ok(command) = receiver.try_recv() {
                commands.push(command);
            }
        }

        for command in commands {
            let next_cmd =
                self.handle_command(command, device, window_num, total_windows, audio_session)?;
            if let Some(cmd) = next_cmd {
                let _ =
                    self.handle_command(cmd, device, window_num, total_windows, audio_session)?;
            }
        }
        Ok(())
    }

    fn check_and_handle_command(
        &mut self,
        device: &soapy::Device,
        window_num: usize,
        total_windows: usize,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<()> {
        if let Some(receiver) = &self.command_receiver
            && let Ok(command) = receiver.try_recv()
        {
            let next_cmd =
                self.handle_command(command, device, window_num, total_windows, audio_session)?;
            if let Some(cmd) = next_cmd {
                let _ =
                    self.handle_command(cmd, device, window_num, total_windows, audio_session)?;
            }
        }
        Ok(())
    }

    fn handle_post_scan_waiting(
        &mut self,
        device: &soapy::Device,
        windows_to_process: usize,
        total_windows: usize,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<bool> {
        self.check_and_handle_command(device, windows_to_process, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(true)
    }

    fn handle_post_scan_browse_mode(
        &mut self,
        device: &soapy::Device,
        windows_to_process: usize,
        total_windows: usize,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<bool> {
        self.process_commands(device, windows_to_process, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(true)
    }

    fn scan_band(&mut self, device: &soapy::Device) -> Result<()> {
        fm::clear_processed_frequencies();

        let window_centers = self
            .config
            .band
            .windows(self.config.samp_rate, self.config.window_overlap);
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
        let mut audio_session: Option<crate::audio_session::AudioSession> = None;

        loop {
            // Check for shutdown FIRST - compiler will force us to handle this in all match arms
            if self.shutdown_coordinator.is_shutdown() {
                self.scanner_state.shutdown();
            }

            // Exhaustive state machine - compiler enforces handling all states
            match &self.scanner_state.mode {
                crate::scanner_state::ScanMode::ShuttingDown => {
                    debug!("Shutdown requested, stopping band scanning");
                    break;
                }
                crate::scanner_state::ScanMode::ScanComplete { .. } => {
                    if !self.handle_post_scan_waiting(
                        device,
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )? {
                        break;
                    }
                    continue;
                }
                crate::scanner_state::ScanMode::ScanCompletePaused { .. } => {
                    if !self.handle_post_scan_browse_mode(
                        device,
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )? {
                        break;
                    }
                    continue;
                }
                crate::scanner_state::ScanMode::Paused { .. } => {
                    if !i.is_multiple_of(50) {
                        debug!(
                            iteration = i,
                            total = windows_to_process,
                            "Paused - waiting for commands"
                        );
                    }

                    self.process_commands(device, i + 1, window_centers.len(), &mut audio_session)?;

                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                crate::scanner_state::ScanMode::Listening { .. } => {
                    self.process_commands(device, i + 1, window_centers.len(), &mut audio_session)?;

                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                crate::scanner_state::ScanMode::Scanning => {
                    // Check if we've scanned all windows
                    if i >= windows_to_process {
                        debug!("Scan band complete - all windows processed");
                        self.scanner_state.mark_scan_complete(windows_to_process);
                        continue;
                    }

                    debug!(
                        iteration = i,
                        total = windows_to_process,
                        "Start of scan loop iteration"
                    );

                    self.process_commands(device, i + 1, window_centers.len(), &mut audio_session)?;

                    // After processing commands, check if we transitioned to paused
                    if self.scanner_state.is_paused() {
                        continue;
                    }

                    debug!(
                        window = i + 1,
                        total = windows_to_process,
                        "Processing window"
                    );

                    self.check_and_handle_command(
                        device,
                        i + 1,
                        window_centers.len(),
                        &mut audio_session,
                    )?;
                    if self.scanner_state.is_paused() {
                        continue;
                    }

                    let center_freq = window_centers[i];
                    let window = Window::new(crate::window::WindowConfig {
                        center_freq,
                        window_num: i + 1,
                        total_windows: window_centers.len(),
                        device: device.clone(),
                        config: self.config.clone(),
                        progress_reporter: self.progress_reporter.clone(),
                        shutdown_coordinator: self.shutdown_coordinator.clone(),
                        pause_signal: Some(self.pause_signal.clone()),
                    });
                    let segment = device.tune(&self.config, center_freq)?;

                    self.check_and_handle_command(
                        device,
                        i + 1,
                        window_centers.len(),
                        &mut audio_session,
                    )?;
                    if self.scanner_state.is_paused() {
                        continue;
                    }

                    window.process(&*segment)?;

                    self.process_commands(device, i + 1, window_centers.len(), &mut audio_session)?;

                    debug!(
                        completed_window = i + 1,
                        next_window = i + 2,
                        remaining = windows_to_process - i - 1,
                        "Window complete, advancing to next"
                    );
                    i += 1;
                }
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScanningConfig;
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

        pub fn get_messages(&self) -> Vec<String> {
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
        ScanningConfig {
            audio_buffer_size: 8192,
            scanning_windows: Some(2),
            fft_size: 1024,
            peak_scan_duration: 1.5,
            audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),
            ..Default::default()
        }
    }

    fn create_test_tuner_id() -> crate::sdr::TunerId {
        crate::sdr::TunerId::from_serial("mock", "test123")
    }

    fn create_test_backend() -> Arc<dyn crate::sdr::Backend> {
        Arc::new(crate::sdr::mock::Mock)
    }

    #[test]
    fn test_main_thread_creation() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let tuner_id = create_test_tuner_id();
        let backend = create_test_backend();
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let main_thread = MainThread::new(
            config,
            console_writer,
            logger,
            tuner_id,
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
        let tuner_id = create_test_tuner_id();
        let backend = create_test_backend();
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let _main_thread = MainThread::new(
            config,
            console_writer,
            logger,
            tuner_id,
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
        let tuner_id = create_test_tuner_id();
        let backend = create_test_backend();
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let main_thread = MainThread::new(
            config,
            console_writer,
            logger,
            tuner_id,
            backend,
            shutdown_coordinator,
        )
        .unwrap();

        // This would normally call SoapySDR and process windows, but we can test the console output pattern
        main_thread.console_writer.write_info("Test message");

        let messages = console_clone.get_messages();
        assert_eq!(messages, vec!["INFO: Test message"]);
    }

    #[test]
    fn test_parse_stations() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let tuner_id = create_test_tuner_id();
        let backend = create_test_backend();
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let main_thread = MainThread::new(
            config,
            console_writer,
            logger,
            tuner_id,
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
        let tuner_id = create_test_tuner_id();
        let backend = create_test_backend();
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let main_thread = MainThread::new(
            config,
            console_writer,
            logger,
            tuner_id,
            backend,
            shutdown_coordinator,
        )
        .unwrap();

        let result = main_thread.parse_stations("88.9e6,invalid,107.3e6");
        assert!(result.is_err());
    }
}
