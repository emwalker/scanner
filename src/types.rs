use rustradio::Complex;
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

/// Abstraction for sources of I/Q samples
pub trait SampleSource {
    /// Read samples into the provided buffer
    /// Returns the number of samples actually read
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize>;

    /// Get the configured sample rate
    fn sample_rate(&self) -> f64;

    /// Get the configured center frequency  
    fn center_frequency(&self) -> f64;

    /// Clean up resources when done
    fn deactivate(&mut self) -> Result<()>;

    fn peak_scan_duration(&self) -> f64;

    fn device_args(&self) -> &str;
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
        config: &crate::ScanningConfig,
        sdr_rx: std::sync::mpsc::Receiver<rustradio::Complex>,
        center_freq: f64,
        signal_tx: std::sync::mpsc::SyncSender<Signal>,
    ) -> Result<()> {
        match self {
            Candidate::Fm(candidate) => candidate.analyze(config, sdr_rx, center_freq, signal_tx),
        }
    }
}
