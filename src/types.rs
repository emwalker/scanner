use thiserror::Error;

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
}

pub type Result<T> = std::result::Result<T, ScannerError>;

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f32,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub frequency_hz: f64,
    pub peak_count: usize,
    pub max_magnitude: f32,
    pub avg_magnitude: f32,
    pub signal_strength: String,
}
