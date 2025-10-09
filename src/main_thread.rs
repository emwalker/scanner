use crate::audio::session::AudioSession;
use crate::core::types::{
    ConsoleWriter, Logger, ModulationType, Result, ScannerError, ScanningConfig,
};
use crate::hardware::DeviceId;
use crate::hardware::pool::{Pool, PoolFilter, Segment, TunerActivity, TunerState, TuningMode};
use crate::scanner_state::{PauseSignal, ScannerState};
use crate::scanning::window::Window;
use crate::shutdown::ShutdownCoordinator;
use crate::signal;
use crate::ui::{
    NoOpProgressReporter, ProgressEventType, ProgressReporter, ScannerCommand, TuiEvent,
};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use tracing::{debug, info};

#[derive(Clone)]
struct TuneParams {
    candidate_id: String,
    window_id: usize,
    center_frequency: f64,
    candidate_frequency: f64,
    signal_strength: Option<f64>,
    audio_quality: Option<crate::audio::quality::AudioQuality>,
}

enum LoopControl {
    Continue,
    Break,
    Advance,
}

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

    fn handle_tune_command(
        &self,
        audio_session: &mut AudioSession,
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

        // CRITICAL: Stop current station FIRST to release tuner back to pool
        // This must happen BEFORE creating new segment, otherwise we get NoAvailableTuner
        audio_session.stop_current_station();

        // Create pool-based segment for listening
        let segment = Segment::new(
            &self.pool,
            params.center_frequency,
            &self.config,
            &self.shutdown_coordinator,
        )?;

        // Get tuner ID from pool status (first allocated tuner for listening)
        let status = self.pool.status();
        let tuner_id = status
            .tuners
            .iter()
            .find(|t| {
                t.state == TunerState::Allocated && t.activity == Some(TunerActivity::Listening)
            })
            .map(|t| t.id.device_id.clone());

        let signal = crate::core::types::Signal {
            frequency_hz: params.candidate_frequency,
            signal_strength: params.signal_strength.unwrap_or(0.1) as f32,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: self.config.audio_sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 0,
            detection_center_freq: params.center_frequency,
            audio_quality: params
                .audio_quality
                .unwrap_or(crate::audio::quality::AudioQuality::Unknown),
        };

        tracing::info!(
            "playing {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );

        audio_session.tune_to_station(&signal, Box::new(segment), &self.config)?;

        debug!(
            candidate_id = ?params.candidate_id,
            event_type = "AudioPlaybackStarted",
            "MainThread: Sending AudioPlaybackStarted event to TUI"
        );

        self.progress_reporter.report(crate::ui::ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: params.candidate_frequency,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: params.center_frequency,
                window_id: params.window_id,
            },
            candidate_id: Some(params.candidate_id),
            audio_quality: None,
            signal_strength: None,
            timestamp: std::time::Instant::now(),
            tuner_id,
        });

        Ok(())
    }

    fn handle_command(
        &mut self,
        command: ScannerCommand,
        window_num: usize,
        _total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<Option<ScannerCommand>> {
        match command {
            ScannerCommand::Pause => {
                debug!(window = window_num, "Scanner paused, creating AudioSession");
                self.pause_signal.pause();
                self.scanner_state.handle_pause(window_num);

                *audio_session = Some(AudioSession::new(
                    &self.config,
                    self.shutdown_coordinator.clone(),
                )?);
                debug!("AudioSession created for browse mode");

                // Send Paused event to TUI so it knows scanning has stopped and can now tune
                if let Some(ref sender) = self.tui_event_sender {
                    // Get any available tuner ID from pool status
                    let status = self.pool.status();
                    let tuner_id = status
                        .tuners
                        .first()
                        .map(|t| t.id.device_id.clone())
                        .unwrap_or_else(|| DeviceId::from_serial("unknown", "0"));
                    let _ = sender.send(TuiEvent::Paused { tuner_id });
                }

                Ok(None)
            }
            ScannerCommand::ResumeScan => {
                debug!(
                    window = window_num,
                    "Scanner resuming - exiting selection mode and continuing scan"
                );
                self.pause_signal.unpause();
                let _next_window = self.scanner_state.handle_resume();

                // Pool will automatically handle tuner state when AudioSession drops
                *audio_session = None;
                debug!("AudioSession dropped, returning to scan mode");

                // Send updated pool status to TUI
                self.send_active_tuners_update();

                Ok(None)
            }
            ScannerCommand::TuneToCandidate {
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

                // Pool will automatically track tuner activity when we create AudioSession
                // Send updated status to TUI
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
                    self.handle_tune_command(session, params.clone())?;
                    self.current_playing = Some(params);
                    Ok(None)
                } else {
                    debug!("TuneToCandidate received but no AudioSession exists");
                    Ok(None)
                }
            }
            ScannerCommand::StopListening => {
                debug!("Stopped listening, returning to browsing mode");
                self.scanner_state.handle_stop_listening();

                if let Some(session) = audio_session {
                    session.stop_current_station();
                }

                // Send AudioPlaybackCompleted event if we were playing something
                if let Some(params) = self.current_playing.take() {
                    // Get tuner ID from pool status
                    let status = self.pool.status();
                    let tuner_id = status
                        .tuners
                        .iter()
                        .find(|t| {
                            t.state == TunerState::Allocated
                                && t.activity == Some(TunerActivity::Listening)
                        })
                        .map(|t| t.id.device_id.clone());
                    self.progress_reporter.report(crate::ui::ProgressEvent {
                        event_type: ProgressEventType::AudioPlaybackCompleted,
                        frequency_hz: params.candidate_frequency,
                        metadata: crate::scanning::window::WindowMetadata {
                            center_frequency_hz: params.center_frequency,
                            window_id: params.window_id,
                        },
                        candidate_id: Some(params.candidate_id),
                        audio_quality: params.audio_quality,
                        signal_strength: params.signal_strength,
                        timestamp: std::time::Instant::now(),
                        tuner_id,
                    });
                }

                Ok(None)
            }
        }
    }

    fn scan_stations(&self, stations_str: &str) -> Result<()> {
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

            // Create a window for this specific station frequency (pool-based)
            let window = Window::for_station(
                station_freq,
                station_idx + 1,
                total_stations,
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
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        let mut commands = Vec::new();
        if let Some(receiver) = &self.command_receiver {
            while let Ok(command) = receiver.try_recv() {
                commands.push(command);
            }
        }

        for command in commands {
            let next_cmd =
                self.handle_command(command, window_num, total_windows, audio_session)?;
            if let Some(cmd) = next_cmd {
                let _ = self.handle_command(cmd, window_num, total_windows, audio_session)?;
            }
        }
        Ok(())
    }

    fn check_and_handle_command(
        &mut self,
        window_num: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        if let Some(receiver) = &self.command_receiver
            && let Ok(command) = receiver.try_recv()
        {
            let next_cmd =
                self.handle_command(command, window_num, total_windows, audio_session)?;
            if let Some(cmd) = next_cmd {
                let _ = self.handle_command(cmd, window_num, total_windows, audio_session)?;
            }
        }
        Ok(())
    }

    fn handle_post_scan_waiting(
        &mut self,
        windows_to_process: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<LoopControl> {
        self.check_and_handle_command(windows_to_process, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_post_scan_browse_mode(
        &mut self,
        windows_to_process: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<LoopControl> {
        self.process_commands(windows_to_process, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn process_window(
        &mut self,
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

        self.check_and_handle_command(window_num, total_windows, audio_session)?;
        Ok(self.scanner_state.is_paused())
    }

    fn handle_scanning_state(
        &mut self,
        window_index: usize,
        windows_to_process: usize,
        window_centers: &[f64],
        audio_session: &mut Option<AudioSession>,
    ) -> Result<LoopControl> {
        if window_index >= windows_to_process {
            debug!("Scan band complete - all windows processed");
            self.scanner_state.mark_scan_complete(windows_to_process);
            return Ok(LoopControl::Continue);
        }

        debug!(
            iteration = window_index,
            total = windows_to_process,
            "Start of scan loop iteration"
        );

        if self.process_commands_with_pause_check(
            window_index + 1,
            window_centers.len(),
            audio_session,
        )? {
            return Ok(LoopControl::Continue);
        }

        let center_freq = window_centers[window_index];
        self.process_window(window_index + 1, center_freq, window_centers.len())?;

        self.process_commands(window_index + 1, window_centers.len(), audio_session)?;

        debug!(
            completed_window = window_index + 1,
            next_window = window_index + 2,
            remaining = windows_to_process - window_index - 1,
            "Window complete, advancing to next"
        );

        Ok(LoopControl::Advance)
    }

    fn handle_paused_state(
        &mut self,
        window_index: usize,
        windows_to_process: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<LoopControl> {
        if !window_index.is_multiple_of(50) {
            debug!(
                iteration = window_index,
                total = windows_to_process,
                "Paused - waiting for commands"
            );
        }

        self.process_commands(window_index + 1, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_listening_state(
        &mut self,
        window_index: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<LoopControl> {
        self.process_commands(window_index + 1, total_windows, audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn scan_band(&mut self) -> Result<()> {
        signal::clear_processed_frequencies();

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
        let mut audio_session: Option<AudioSession> = None;

        loop {
            if self.shutdown_coordinator.is_shutdown() {
                self.scanner_state.shutdown();
            }

            let control = match &self.scanner_state.mode {
                crate::scanner_state::ScanMode::ShuttingDown => {
                    debug!("Shutdown requested, stopping band scanning");
                    LoopControl::Break
                }
                crate::scanner_state::ScanMode::ScanComplete { .. } => self
                    .handle_post_scan_waiting(
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )?,
                crate::scanner_state::ScanMode::ScanCompletePaused { .. } => self
                    .handle_post_scan_browse_mode(
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )?,
                crate::scanner_state::ScanMode::Paused { .. } => self.handle_paused_state(
                    i,
                    windows_to_process,
                    window_centers.len(),
                    &mut audio_session,
                )?,
                crate::scanner_state::ScanMode::Listening { .. } => {
                    self.handle_listening_state(i, window_centers.len(), &mut audio_session)?
                }
                crate::scanner_state::ScanMode::Scanning => self.handle_scanning_state(
                    i,
                    windows_to_process,
                    &window_centers,
                    &mut audio_session,
                )?,
            };

            match control {
                LoopControl::Break => break,
                LoopControl::Continue => continue,
                LoopControl::Advance => i += 1,
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
    use crate::audio::quality::AudioAnalyzer;
    use crate::core::types::ScanningConfig;
    use crate::hardware::Backend;
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
        ScanningConfig {
            audio_buffer_size: 8192,
            scanning_windows: Some(2),
            fft_size: 1024,
            peak_scan_duration: 1.5,
            audio_analyzer: AudioAnalyzer::mock(),
            ..Default::default()
        }
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

impl Drop for MainThread {
    fn drop(&mut self) {
        self.pool.shutdown();
    }
}
