use clap::{Parser, Subcommand, ValueEnum};

use crate::core::types::Band;

#[derive(ValueEnum, Clone, Debug)]
pub enum AudioClassifier {
    Heuristic1,
    Heuristic2,
    Heuristic3,
    RandomForest,
}

#[derive(Parser, Debug)]
#[command(name = "scanner")]
#[command(about = "FM radio scanner with audio quality analysis")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Scan for FM radio stations
    Scan(Box<ScanArgs>),
    /// Train audio quality machine learning model
    Train(TrainArgs),
    /// Internal worker subprocesses (hidden from help)
    #[command(hide = true, subcommand)]
    Worker(WorkerCommand),
}

#[derive(Parser, Debug)]
pub struct ScanArgs {
    /// AGC settling time in seconds before peak scanning begins
    #[arg(long, default_value_t = 0.45)]
    pub agc_settling_time: f64,

    #[arg(long)]
    pub audio_capture_dir: Option<String>,

    #[arg(long, default_value_t = 2.0)]
    pub audio_capture_duration: f64,

    /// Audio quality classifier to use
    #[arg(long, default_value = "random-forest")]
    pub audio_classifier: AudioClassifier,

    #[arg(long, default_value_t = Band::Fm)]
    pub band: Band,

    #[arg(long, default_value_t = 2.0)]
    pub capture_duration: f64,

    #[arg(long)]
    pub capture_iq: Option<String>,

    #[arg(long)]
    pub debug_pipeline: bool,

    #[arg(long)]
    pub device_args: Option<String>,

    /// Disable CFAR detection (use fixed threshold instead of adaptive threshold)
    #[arg(long)]
    pub disable_cfar: bool,

    /// Disable frequency tracking (use FFT estimates directly)
    #[arg(long)]
    pub disable_frequency_tracking: bool,

    /// Disable IF AGC in both detection and audio pipelines (AGC enabled by default)
    #[arg(long)]
    pub disable_if_agc: bool,

    /// Disable signal averaging improvements (exponential smoothing, multi-frame averaging, etc.)
    #[arg(long)]
    pub disable_signal_averaging: bool,

    /// Disable spectral preprocessing (windowing, zero-padding)
    #[arg(long)]
    pub disable_spectral_preprocessing: bool,

    /// Disable squelch analysis and generate signals from all signals regardless of audio
    /// quality
    #[arg(long)]
    pub disable_squelch: bool,

    #[arg(long, default_value_t = 3)]
    pub duration: u64,

    /// Enable dynamic noise floor estimation (percentile-based adaptive thresholds) - experimental
    #[arg(long)]
    pub enable_dynamic_noise_floor: bool,

    /// Enable multi-frame integration (peak persistence tracking) - experimental
    #[arg(long)]
    pub enable_multi_frame_integration: bool,

    /// Frequency tracking method (pll, spectral, correlation)
    #[arg(long, default_value = "pll")]
    pub frequency_tracking: String,

    #[arg(long, help = "SDR gain in dB (0 to 48 for SDRplay, default 24)")]
    pub gain: Option<f64>,

    /// Write debug logs to file (useful with TUI to capture diagnostics)
    #[arg(long)]
    pub headless: bool,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,

    /// Duration in seconds for squelch to analyze audio vs noise
    #[arg(long, default_value_t = 1.0)]
    pub learning_duration: f32,

    /// Output in standard log format
    #[arg(long)]
    pub log: bool,

    /// Write debug logs to file (useful with TUI to capture diagnostics)
    #[arg(long, default_value = "/tmp/scanner.log")]
    pub log_file: Option<String>,

    /// Path to pre-trained model file (if not specified, auto-discovers latest)
    #[arg(long)]
    pub model_path: Option<String>,

    #[arg(long, default_value_t = 0.1)]
    pub peak_scan_duration: f64,

    #[arg(long)]
    pub print_signals: bool,

    /// Suppress debug and info logs (only show WARN and ERROR)
    #[arg(long)]
    pub quiet: bool,

    /// Maximum number of scanning windows to process (default: all windows)
    #[arg(long)]
    pub scanning_windows: Option<usize>,

    /// Minimum spectral score threshold for signal creation (0.0-1.0)
    #[arg(long, default_value_t = 0.2)]
    pub spectral_threshold: f32,

    /// Audio quality threshold for squelch ("static", "no-audio", "poor", "moderate", "good")
    /// Signals below this threshold will be filtered out. Default: "moderate"
    #[arg(long, default_value = "moderate")]
    pub squelch_threshold: String,

    #[arg(long)]
    pub stations: Option<String>,

    /// Output in simplified text format
    #[arg(long)]
    pub text: bool,

    /// TUI theme selection (basic-dark, basic-light, bladerunner-dark, bladerunner-light,
    /// interstellar-dark, interstellar-light, dune-dark, dune-light, transport-dark,
    /// transport-light, archive-dark, archive-light, minimal-dark, minimal-light)
    #[arg(long, default_value = "caladan-dark")]
    pub theme: String,

    /// Required accuracy for frequency tracking convergence (Hz)
    #[arg(long, default_value_t = 5000.0)]
    pub tracking_accuracy: f64,

    /// Reduce log output (show DEBUG and above)
    #[arg(long)]
    pub verbose: bool,

    /// Audio playback volume (0.0 to 1.0, default 0.3)
    #[arg(long, default_value_t = 0.3)]
    pub volume: f32,

    /// Window overlap percentage for band scanning (0.0-1.0, where 0.75 = 75% overlap)
    #[arg(long, default_value_t = 0.75)]
    pub window_overlap: f64,
}

#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Model version string
    #[arg(long, default_value = "0.1.0")]
    pub model_version: String,

    /// Output path for trained model (if not specified, auto-generates versioned filename)
    #[arg(long)]
    pub output_model: Option<String>,

    /// Sample rate for feature extraction
    #[arg(long, default_value_t = 48000.0)]
    pub sample_rate: f32,

    /// Directory containing training audio files
    #[arg(long, default_value = "tests/data/audio/quality")]
    pub training_data_dir: String,

    /// Reduce log output (show only INFO and above)
    #[arg(long)]
    pub verbose: bool,

    /// Suppress debug and info logs (only show WARN and ERROR)
    #[arg(long)]
    pub quiet: bool,
}

#[derive(Subcommand, Debug)]
pub enum WorkerCommand {
    /// Enumerate devices for a specific backend (internal use only)
    Enumerate {
        /// Backend to enumerate ("soapy", "seify", "rtlsdr")
        #[arg(long)]
        backend: String,
        /// Unix socket path for IPC communication
        #[arg(long)]
        socket_path: String,
        /// Optional log file path
        #[arg(long)]
        log_file: Option<String>,
    },
    /// Stream I/Q from device (internal use only)
    Device {
        /// Serialized DeviceId (will be parsed in worker)
        #[arg(long)]
        device_id_str: String,

        /// Unix socket path for control messages (bidirectional)
        #[arg(long)]
        control_socket_path: String,

        /// Unix socket path for data streaming (unidirectional)
        #[arg(long)]
        data_socket_path: String,

        /// Optional log file path
        #[arg(long)]
        log_file: Option<String>,
    },
}
