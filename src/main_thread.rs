use crate::scanner_state::{PauseSignal, ScannerState};
use crate::sdr::Device;
use crate::terminal::{NoOpProgressReporter, ProgressReporter};
use crate::types::{ConsoleWriter, Logger, Result, ScannerError, ScanningConfig};
use crate::window::Window;
use crate::{fm, soapy};
use std::sync::Arc;
use tracing::{debug, info};

pub struct MainThread {
    config: ScanningConfig,
    console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
    _logger: Arc<dyn Logger + Send + Sync>,
    devices: Vec<soapy::Device>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_listener: triggered::Listener,
    command_receiver: Option<std::sync::mpsc::Receiver<crate::terminal::ScannerCommand>>,
    scanner_state: ScannerState,
    pause_signal: PauseSignal,
}

impl MainThread {
    pub fn new(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        devices: Vec<soapy::Device>,
        shutdown_listener: triggered::Listener,
    ) -> Result<Self> {
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            devices,
            progress_reporter: Arc::new(NoOpProgressReporter),
            shutdown_listener,
            command_receiver: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
        })
    }

    pub fn new_with_progress(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        devices: Vec<soapy::Device>,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_listener: triggered::Listener,
    ) -> Result<Self> {
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            devices,
            progress_reporter,
            shutdown_listener,
            command_receiver: None,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
        })
    }

    pub fn with_command_receiver(
        mut self,
        receiver: std::sync::mpsc::Receiver<crate::terminal::ScannerCommand>,
    ) -> Self {
        self.command_receiver = Some(receiver);
        self
    }

    pub fn run(mut self, stations: Option<String>) -> Result<()> {
        // Logging is now initialized in main() before SDR operations

        // Discover available SDR devices
        if self.devices.is_empty() {
            return Err(crate::types::ScannerError::Custom(
                "No SDR devices found".to_string(),
            ));
        }

        // Create device from the first available device string
        let device = self.devices[0].clone();
        self.console_writer.write_info("Scanning for stations ...");

        if let Some(stations_str) = stations {
            self.scan_stations(&device, &stations_str)?;
        } else {
            self.scan_band(&device)?;
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
        device: &soapy::Device,
        audio_session: &mut crate::audio_session::AudioSession,
        center_frequency: f64,
        candidate_frequency: f64,
        signal_strength: Option<f64>,
        audio_quality: Option<crate::audio_quality::AudioQuality>,
    ) -> Result<()> {
        debug!(
            candidate_mhz = candidate_frequency / 1e6,
            center_mhz = center_frequency / 1e6,
            signal_strength = ?signal_strength,
            audio_quality = ?audio_quality,
            "Tuning to candidate"
        );

        let segment = device.tune(&self.config, center_frequency)?;

        let signal = crate::types::Signal {
            frequency_hz: candidate_frequency,
            signal_strength: signal_strength.unwrap_or(0.1) as f32,
            bandwidth_hz: 200_000.0,
            modulation: crate::types::ModulationType::WFM,
            audio_sample_rate: self.config.audio_sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 0,
            detection_center_freq: center_frequency,
            audio_quality: audio_quality.unwrap_or(crate::audio_quality::AudioQuality::Unknown),
        };

        tracing::info!(
            "playing {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );

        audio_session.tune_to_station(&signal, segment, &self.config)?;

        Ok(())
    }

    fn handle_command(
        &mut self,
        command: crate::terminal::ScannerCommand,
        device: &soapy::Device,
        window_num: usize,
        _total_windows: usize,
        _current_paused: bool,
        audio_session: &mut Option<crate::audio_session::AudioSession>,
    ) -> Result<(bool, Option<crate::terminal::ScannerCommand>)> {
        match command {
            crate::terminal::ScannerCommand::Pause => {
                debug!(window = window_num, "Scanner paused, creating AudioSession");
                self.pause_signal.pause();
                self.scanner_state.handle_pause(window_num);

                *audio_session = Some(crate::audio_session::AudioSession::new(&self.config)?);
                debug!("AudioSession created for browse mode");

                Ok((true, None))
            }
            crate::terminal::ScannerCommand::ResumeScan => {
                debug!(
                    window = window_num,
                    "Scanner resuming - exiting selection mode and continuing scan"
                );
                self.pause_signal.unpause();
                let _next_window = self.scanner_state.handle_resume();

                *audio_session = None;
                debug!("AudioSession dropped, returning to scan mode");

                Ok((false, None))
            }
            crate::terminal::ScannerCommand::TuneToCandidate {
                window_id: _,
                center_frequency,
                candidate_frequency,
                signal_strength,
                audio_quality,
            } => {
                self.scanner_state.handle_tune(window_num);

                if let Some(session) = audio_session {
                    self.handle_tune_command(
                        device,
                        session,
                        center_frequency,
                        candidate_frequency,
                        signal_strength,
                        audio_quality,
                    )?;
                    Ok((true, None))
                } else {
                    debug!("TuneToCandidate received but no AudioSession exists");
                    Ok((true, None))
                }
            }
            crate::terminal::ScannerCommand::StopListening => {
                debug!("Stopped listening, returning to browsing mode");
                self.scanner_state.handle_stop_listening();

                if let Some(session) = audio_session {
                    session.stop_current_station();
                }

                Ok((true, None))
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
            // Check for shutdown before processing each station
            if self.shutdown_listener.is_triggered() {
                debug!("Shutdown requested, stopping station scanning");
                break;
            }

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
                self.shutdown_listener.clone(),
            );

            // Process using the full band scanning pipeline (peak detection, candidates, etc.)
            window.process(&*segment)?;
        }

        Ok(())
    }

    fn scan_band(&mut self, device: &soapy::Device) -> Result<()> {
        // Clear any previously processed frequencies from earlier scans
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

        let mut paused = false;
        let mut i = 0;
        let mut audio_session: Option<crate::audio_session::AudioSession> = None;

        while i < windows_to_process {
            debug!(
                iteration = i,
                total = windows_to_process,
                paused = paused,
                "Start of scan loop iteration"
            );

            // Collect commands without holding borrow of self
            let mut commands = Vec::new();
            if let Some(receiver) = &self.command_receiver {
                while let Ok(command) = receiver.try_recv() {
                    commands.push(command);
                }
            }

            // Process commands
            for command in commands {
                let (new_paused, next_cmd) = self.handle_command(
                    command,
                    device,
                    i + 1,
                    window_centers.len(),
                    paused,
                    &mut audio_session,
                )?;
                paused = new_paused;
                if let Some(cmd) = next_cmd {
                    let (final_paused, _) = self.handle_command(
                        cmd,
                        device,
                        i + 1,
                        window_centers.len(),
                        paused,
                        &mut audio_session,
                    )?;
                    paused = final_paused;
                }
            }

            if paused {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }

            debug!(
                window = i + 1,
                total = windows_to_process,
                paused = paused,
                "Not paused, will process window"
            );

            // Check for shutdown before processing each window
            if self.shutdown_listener.is_triggered() {
                debug!("Shutdown requested, stopping band scanning");
                break;
            }

            if let Some(receiver) = &self.command_receiver
                && let Ok(command) = receiver.try_recv()
            {
                let (new_paused, next_cmd) = self.handle_command(
                    command,
                    device,
                    i + 1,
                    window_centers.len(),
                    paused,
                    &mut audio_session,
                )?;
                paused = new_paused;
                if let Some(cmd) = next_cmd {
                    let (final_paused, _) = self.handle_command(
                        cmd,
                        device,
                        i + 1,
                        window_centers.len(),
                        paused,
                        &mut audio_session,
                    )?;
                    paused = final_paused;
                }
                if paused {
                    continue;
                }
            }

            let center_freq = window_centers[i];
            let window = Window::new(crate::window::WindowConfig {
                center_freq,
                window_num: i + 1,
                total_windows: window_centers.len(),
                device: device.clone(),
                config: self.config.clone(),
                progress_reporter: self.progress_reporter.clone(),
                shutdown_listener: self.shutdown_listener.clone(),
                pause_signal: Some(self.pause_signal.clone()),
            });
            let segment = device.tune(&self.config, center_freq)?;

            if let Some(receiver) = &self.command_receiver
                && let Ok(command) = receiver.try_recv()
            {
                let (new_paused, next_cmd) = self.handle_command(
                    command,
                    device,
                    i + 1,
                    window_centers.len(),
                    paused,
                    &mut audio_session,
                )?;
                paused = new_paused;
                if let Some(cmd) = next_cmd {
                    let (final_paused, _) = self.handle_command(
                        cmd,
                        device,
                        i + 1,
                        window_centers.len(),
                        paused,
                        &mut audio_session,
                    )?;
                    paused = final_paused;
                }
                if paused {
                    continue;
                }
            }

            window.process(&*segment)?;

            // Collect commands without holding borrow of self
            let mut commands = Vec::new();
            if let Some(receiver) = &self.command_receiver {
                while let Ok(command) = receiver.try_recv() {
                    commands.push(command);
                }
            }

            // Process commands
            for command in commands {
                let (new_paused, next_cmd) = self.handle_command(
                    command,
                    device,
                    i + 1,
                    window_centers.len(),
                    paused,
                    &mut audio_session,
                )?;
                paused = new_paused;
                if let Some(cmd) = next_cmd {
                    let (final_paused, _) = self.handle_command(
                        cmd,
                        device,
                        i + 1,
                        window_centers.len(),
                        paused,
                        &mut audio_session,
                    )?;
                    paused = final_paused;
                }
            }

            debug!(
                completed_window = i + 1,
                next_window = i + 2,
                remaining = windows_to_process - i - 1,
                "Window complete, advancing to next"
            );
            i += 1;
        }
        debug!("Scan band complete - all windows processed");
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

    #[test]
    fn test_main_thread_creation() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let devices: Vec<soapy::Device> =
            vec![soapy::Device("driver=mock, label=Test Device".to_string())];
        let (_trigger, shutdown_listener) = triggered::trigger();

        let main_thread =
            MainThread::new(config, console_writer, logger, devices, shutdown_listener);
        assert!(main_thread.is_ok());
    }

    #[test]
    fn test_main_thread_run_no_devices() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let devices: Vec<soapy::Device> = vec![];
        let (_trigger, shutdown_listener) = triggered::trigger();

        let main_thread =
            MainThread::new(config, console_writer, logger, devices, shutdown_listener).unwrap();
        let result = main_thread.run(None);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("No SDR devices found"));
        }
    }

    #[test]
    fn test_console_output() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let console_clone = Arc::clone(&console_writer);
        let logger = Arc::new(MockLogger::new());
        let devices: Vec<soapy::Device> =
            vec![soapy::Device("driver=mock, label=Test Device".to_string())];
        let (_trigger, shutdown_listener) = triggered::trigger();

        let main_thread =
            MainThread::new(config, console_writer, logger, devices, shutdown_listener).unwrap();

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
        let devices: Vec<soapy::Device> =
            vec![soapy::Device("driver=mock, label=Test Device".to_string())];
        let (_trigger, shutdown_listener) = triggered::trigger();

        let main_thread =
            MainThread::new(config, console_writer, logger, devices, shutdown_listener).unwrap();

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
        let devices: Vec<soapy::Device> =
            vec![soapy::Device("driver=mock, label=Test Device".to_string())];
        let (_trigger, shutdown_listener) = triggered::trigger();

        let main_thread =
            MainThread::new(config, console_writer, logger, devices, shutdown_listener).unwrap();

        let result = main_thread.parse_stations("88.9e6,invalid,107.3e6");
        assert!(result.is_err());
    }
}
