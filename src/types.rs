use thiserror::Error;

use crate::fm;

#[derive(Error, Debug)]
pub enum ScannerError {
    #[error(transparent)]
    Sdr(#[from] soapysdr::Error),
    #[error("Error: {0}")]
    Custom(String),
    #[error(transparent)]
    Audio(#[from] cpal::SupportedStreamConfigsError),
    #[error(transparent)]
    AudioBuild(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    AudioPlay(#[from] cpal::PlayStreamError),
    #[error(transparent)]
    AudioDevice(#[from] cpal::DefaultStreamConfigError),
    #[error(transparent)]
    AudioDeviceName(#[from] cpal::DeviceNameError),
    #[error(transparent)]
    RustRadio(#[from] rustradio::Error),
    #[error(transparent)]
    Stderr(#[from] log::SetLoggerError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("I/Q capture error: {0}")]
    IqCapture(String),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ScannerError>;

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f32,
}

#[derive(Debug)]
pub enum Candidate {
    Fm(fm::Candidate),
}

/// Represents a successfully detected and demodulated signal
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Signal {
    /// Center frequency of the signal in Hz
    pub frequency_hz: f64,
    /// Signal strength/power measurement
    pub signal_strength: f32,
    /// Estimated bandwidth of the signal in Hz
    pub bandwidth_hz: f32,
    /// Type of modulation detected
    pub modulation: ModulationType,
    /// Audio sample rate for this signal
    pub audio_sample_rate: u32,
    /// Timestamp when signal was detected
    pub detected_at: std::time::SystemTime,
    /// Duration of analysis period that led to detection
    pub analysis_duration_ms: u32,
    /// Center frequency used by SDR during detection (needed for audio processing offset calculation)
    pub detection_center_freq: f64,
}

#[derive(Debug, Clone)]
pub enum ModulationType {
    Fm,
    // Future: Am, Digital, etc.
}

impl Signal {
    pub fn new_fm(
        frequency_hz: f64,
        signal_strength: f32,
        bandwidth_hz: f32,
        audio_sample_rate: u32,
        analysis_duration_ms: u32,
        detection_center_freq: f64,
    ) -> Self {
        Self {
            frequency_hz,
            signal_strength,
            bandwidth_hz,
            modulation: ModulationType::Fm,
            audio_sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms,
            detection_center_freq,
        }
    }
}

impl Candidate {
    pub fn frequency_hz(&self) -> f64 {
        match self {
            Candidate::Fm(candidate) => candidate.frequency_hz,
        }
    }

    pub fn analyze(
        &self,
        config: &ScanningConfig,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        center_freq: f64,
        signal_tx: std::sync::mpsc::SyncSender<Signal>,
    ) -> Result<()> {
        match self {
            Candidate::Fm(candidate) => candidate.analyze(config, sdr_rx, center_freq, signal_tx),
        }
    }
}

// Frequency bands for scanning
use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Band {
    /// FM broadcast band (88-108 MHz)
    Fm,
    /// VHF aircraft band (108-137 MHz)
    Aircraft,
    /// 2-meter amateur band (144-148 MHz)
    Ham2m,
    /// NOAA weather radio (162-163 MHz)
    Weather,
    /// Marine VHF band (156-162 MHz)
    Marine,
}

impl std::fmt::Display for Band {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Band::Fm => write!(f, "fm"),
            Band::Aircraft => write!(f, "aircraft"),
            Band::Ham2m => write!(f, "ham2m"),
            Band::Weather => write!(f, "weather"),
            Band::Marine => write!(f, "marine"),
        }
    }
}

impl Band {
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            Band::Fm => (88.0e6, 108.0e6),
            Band::Aircraft => (108.0e6, 137.0e6),
            Band::Ham2m => (144.0e6, 148.0e6),
            Band::Weather => (162.0e6, 163.0e6),
            Band::Marine => (156.0e6, 162.0e6),
        }
    }

    pub fn windows(&self, sample_rate: f64, overlap: f64) -> Vec<f64> {
        let (start_freq, end_freq) = self.frequency_range();
        let usable_bandwidth = sample_rate * 0.8; // Use 80% of bandwidth to avoid edge effects
        let step_size = usable_bandwidth * (1.0 - overlap); // Step size based on overlap percentage

        let mut windows = Vec::new();
        let mut center_freq = start_freq + (usable_bandwidth / 2.0);

        while center_freq - (usable_bandwidth / 2.0) < end_freq {
            windows.push(center_freq);
            center_freq += step_size;
        }

        windows
    }
}

/// Configuration for scanning operations
#[derive(Clone)]
pub struct ScanningConfig {
    pub audio_buffer_size: u32,
    pub audio_sample_rate: u32,
    pub band: Band,
    pub capture_audio_duration: f64,
    pub capture_audio: Option<String>,
    pub capture_duration: f64,
    pub capture_iq: Option<String>,
    pub debug_pipeline: bool,
    pub driver: String,
    pub duration: u64,
    pub scanning_windows: Option<usize>,
    pub fft_size: usize,
    pub peak_detection_threshold: f32,
    pub peak_scan_duration: Option<f64>,
    pub print_candidates: bool,
    pub samp_rate: f64,
    pub squelch_learning_duration: f32,

    // Frequency tracking configuration
    pub frequency_tracking_method: String,
    pub tracking_accuracy: f64,
    pub disable_frequency_tracking: bool,

    // Spectral analysis configuration
    pub spectral_threshold: f32,

    // AGC and window configuration
    pub agc_settling_time: f64,
    pub window_overlap: f64,
}

impl Default for ScanningConfig {
    fn default() -> Self {
        Self {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            band: Band::Fm,
            capture_audio: None,
            capture_audio_duration: 3.0,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            driver: "driver=sdrplay".to_string(),
            duration: 3,
            scanning_windows: None,
            fft_size: 1024,
            peak_detection_threshold: 1.0,
            peak_scan_duration: None,
            print_candidates: false,
            samp_rate: 2_000_000.0,
            squelch_learning_duration: 2.0,

            // Frequency tracking defaults
            frequency_tracking_method: "pll".to_string(),
            tracking_accuracy: 5000.0,
            disable_frequency_tracking: false,

            // Spectral analysis defaults
            spectral_threshold: 0.2,

            // AGC and window defaults
            agc_settling_time: 3.0,
            window_overlap: 0.75,
        }
    }
}
