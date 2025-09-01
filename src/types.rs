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
}

pub type Result<T> = std::result::Result<T, ScannerError>;

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f32,
}

pub enum Candidate {
    Fm(fm::Candidate),
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
        audio_tx: std::sync::mpsc::SyncSender<f32>,
    ) -> Result<()> {
        match self {
            Candidate::Fm(candidate) => candidate.analyze(config, audio_tx),
        }
    }
}
