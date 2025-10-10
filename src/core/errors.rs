use thiserror::Error;

pub const TEST_FREQUENCY_HZ: f64 = 88_900_000.0; // 88.9 MHz - common test frequency

#[derive(Error, Debug)]
pub enum ScannerError {
    #[error(transparent)]
    Audio(#[from] cpal::SupportedStreamConfigsError),
    #[error(transparent)]
    AudioBuild(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    AudioDevice(#[from] cpal::DefaultStreamConfigError),
    #[error(transparent)]
    AudioDeviceName(#[from] cpal::DeviceNameError),
    #[error(transparent)]
    AudioPlay(#[from] cpal::PlayStreamError),
    #[error(transparent)]
    AudioPause(#[from] cpal::PauseStreamError),
    #[error(transparent)]
    Bincode(#[from] bincode::Error),
    #[error("Error: {0}")]
    Custom(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Hardware not available: {0}")]
    HardwareNotAvailable(String),
    #[error("Signal processing failed: {0}")]
    SignalProcessingFailed(String),
    #[error("Pool shutdown in progress")]
    PoolShutdown,
    #[error("Audio format not supported: {0}")]
    UnsupportedAudioFormat(String),
    #[error("ML model error: {0}")]
    ModelError(String),
    #[error("Initialization timeout: {0}")]
    InitializationTimeout(String),
    #[error("Thread panic: {0}")]
    ThreadPanic(String),
    #[error(transparent)]
    Hound(#[from] hound::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("I/Q capture error: {0}")]
    IqCapture(String),
    #[error(transparent)]
    Log(#[from] log::SetLoggerError),
    #[error(transparent)]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error(transparent)]
    RustRadio(#[from] rustradio::Error),
    #[error(transparent)]
    Sdr(#[from] soapysdr::Error),
    #[error(transparent)]
    SdrDevice(#[from] crate::hardware::DeviceError),
    #[error("Device in use: {0:?}")]
    DeviceInUse(crate::hardware::DeviceId),
    #[error("Device not found: {0:?}")]
    DeviceNotFound(crate::hardware::DeviceId),
    #[error("No available tuner matching requirements: {0:?}")]
    NoAvailableTuner(crate::hardware::pool::TaskRequirements),
    #[error("Pool lock timeout - operation would block")]
    PoolLockTimeout,
    #[error("Internal inconsistency: {message}")]
    InternalInconsistency { message: String },
    #[error("Tuner not found: {tuner_id:?}")]
    TunerNotFound {
        tuner_id: crate::hardware::pool::TunerId,
    },
    #[error("Mutex poisoned: {context}")]
    MutexPoisoned { context: String },
    #[error("Invalid device arguments")]
    InvalidDeviceArgs,
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    #[error(transparent)]
    SmartCore(#[from] smartcore::error::Failed),
    #[error("Thread join error")]
    ThreadJoin(Box<dyn std::any::Any + Send>),
    #[error("Failed to set tracing subscriber")]
    TracingSubscriber(#[from] tracing::subscriber::SetGlobalDefaultError),
}

pub type Result<T> = std::result::Result<T, ScannerError>;
