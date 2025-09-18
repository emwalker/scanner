use crate::sdr::Device;
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
}

impl MainThread {
    pub fn new(
        config: ScanningConfig,
        console_writer: Arc<dyn ConsoleWriter + Send + Sync>,
        logger: Arc<dyn Logger + Send + Sync>,
        devices: Vec<soapy::Device>,
    ) -> Result<Self> {
        Ok(MainThread {
            config,
            console_writer,
            _logger: logger,
            devices,
        })
    }

    pub fn run(&self, stations: Option<String>) -> Result<()> {
        // Logging is now initialized in main() before SDR operations

        // Discover available SDR devices
        if self.devices.is_empty() {
            return Err(crate::types::ScannerError::Custom(
                "No SDR devices found".to_string(),
            ));
        }

        // Create device from the first available device string
        let device = &self.devices[0];
        self.console_writer.write_info("Scanning for stations ...");

        if let Some(stations_str) = stations {
            self.scan_stations(device, &stations_str)?;
        } else {
            self.scan_band(device)?;
        }

        self.console_writer.write_info("Scanning complete.");
        Ok(())
    }

    fn parse_stations(&self, stations_str: &str) -> Result<Vec<f64>> {
        stations_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
            .collect()
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
            );

            // Process using the full band scanning pipeline (peak detection, candidates, etc.)
            window.process(&*segment)?;
        }

        Ok(())
    }

    fn scan_band(&self, device: &soapy::Device) -> Result<()> {
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

        for (i, center_freq) in window_centers.iter().enumerate().take(windows_to_process) {
            let window = Window::new(
                *center_freq,
                i + 1,
                window_centers.len(),
                device.clone(),
                self.config.clone(),
            );
            let segment = device.tune(&self.config, *center_freq)?;
            window.process(&*segment)?;
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
    use crate::types::{Band, ScanningConfig};
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
            audio_sample_rate: 48000,
            band: Band::Fm,
            capture_audio_duration: 3.0,
            capture_audio: None,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            duration: 3,
            sdr_gain: 24.0,
            scanning_windows: Some(2),
            fft_size: 1024,
            peak_detection_threshold: 1.0,
            peak_scan_duration: None,
            print_candidates: false,
            samp_rate: 2_000_000.0,
            squelch_learning_duration: 1.0,
            frequency_tracking_method: "pll".to_string(),
            tracking_accuracy: 5000.0,
            disable_frequency_tracking: false,
            spectral_threshold: 0.2,
            agc_settling_time: 0.45,
            window_overlap: 0.75,
            disable_squelch: false,
            squelch_threshold: crate::audio_quality::AudioQuality::Moderate,
            disable_if_agc: false,
            audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),
        }
    }

    #[test]
    fn test_main_thread_creation() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let devices: Vec<soapy::Device> =
            vec![soapy::Device("driver=mock, label=Test Device".to_string())];

        let main_thread = MainThread::new(config, console_writer, logger, devices);
        assert!(main_thread.is_ok());
    }

    #[test]
    fn test_main_thread_run_no_devices() {
        let config = create_test_config();
        let console_writer = Arc::new(MockConsoleWriter::new());
        let logger = Arc::new(MockLogger::new());
        let devices: Vec<soapy::Device> = vec![];

        let main_thread = MainThread::new(config, console_writer, logger, devices).unwrap();
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

        let main_thread = MainThread::new(config, console_writer, logger, devices).unwrap();

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

        let main_thread = MainThread::new(config, console_writer, logger, devices).unwrap();

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

        let main_thread = MainThread::new(config, console_writer, logger, devices).unwrap();

        let result = main_thread.parse_stations("88.9e6,invalid,107.3e6");
        assert!(result.is_err());
    }
}
