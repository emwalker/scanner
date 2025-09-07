use clap::Parser;
use scanner::logging::DefaultLogger;
use scanner::main_thread::{DefaultConsoleWriter, MainThread};
use scanner::soapy;
use scanner::types::{Band, Format, Result, ScanningConfig};
use std::sync::Arc;

pub struct Audio;

const DEFAULT_DRIVER: &str = "driver=sdrplay";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    device_args: Option<String>,

    #[arg(long)]
    stations: Option<String>,

    #[arg(long, help = "SDR gain in dB (0 to 48 for SDRplay, default 24)")]
    gain: Option<f64>,

    #[arg(long, default_value_t = Band::Fm)]
    band: Band,

    #[arg(long, default_value_t = 3)]
    duration: u64,

    #[arg(long)]
    peak_scan_duration: Option<f64>,

    /// Maximum number of scanning windows to process (default: all windows)
    #[arg(long)]
    scanning_windows: Option<usize>,

    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    debug_pipeline: bool,

    #[arg(long)]
    print_candidates: bool,

    /// Output format: plain text (default)
    #[arg(long, group = "output_format")]
    text: bool,

    /// Output format: JSON
    #[arg(long, group = "output_format")]
    json: bool,

    /// Output format: structured logging
    #[arg(long, group = "output_format")]
    log: bool,

    #[arg(long)]
    capture_iq: Option<String>,

    #[arg(long, default_value_t = 2.0)]
    capture_duration: f64,

    #[arg(long)]
    capture_audio: Option<String>,

    #[arg(long, default_value_t = 3.0)]
    capture_audio_duration: f64,

    /// Duration in seconds for squelch to analyze audio vs noise
    #[arg(long, default_value_t = 2.0)]
    learning_duration: f32,

    /// Frequency tracking method (pll, spectral, correlation)
    #[arg(long, default_value = "pll")]
    frequency_tracking: String,

    /// Required accuracy for frequency tracking convergence (Hz)
    #[arg(long, default_value_t = 5000.0)]
    tracking_accuracy: f64,

    /// Disable frequency tracking (use FFT estimates directly)
    #[arg(long)]
    disable_frequency_tracking: bool,

    /// Minimum spectral score threshold for candidate creation (0.0-1.0)
    #[arg(long, default_value_t = 0.2)]
    spectral_threshold: f32,

    /// AGC settling time in seconds before peak scanning begins
    #[arg(long, default_value_t = 3.0)]
    agc_settling_time: f64,

    /// Window overlap percentage for band scanning (0.0-1.0, where 0.75 = 75% overlap)
    #[arg(long, default_value_t = 0.75)]
    window_overlap: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Determine format from flags (default to Text if none specified)
    let format = if args.json {
        Format::Json
    } else if args.log {
        Format::Log
    } else {
        Format::Text // Default when no format specified or when --text is used
    };

    let config = ScanningConfig {
        audio_buffer_size: 8192, // Increased from 4K to 8K samples for better buffering
        audio_sample_rate: 48000,
        band: args.band,
        capture_audio_duration: args.capture_audio_duration,
        capture_audio: args.capture_audio,
        capture_duration: args.capture_duration,
        capture_iq: args.capture_iq,
        debug_pipeline: args.debug_pipeline,
        duration: args.duration,
        sdr_gain: args.gain.unwrap_or(24.0), // Default to middle of 0-48 dB range
        scanning_windows: args.scanning_windows,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: args.peak_scan_duration,
        print_candidates: args.print_candidates,
        samp_rate: 2_000_000.0f64,
        squelch_learning_duration: args.learning_duration,

        // Frequency tracking configuration
        frequency_tracking_method: args.frequency_tracking,
        tracking_accuracy: args.tracking_accuracy,
        disable_frequency_tracking: args.disable_frequency_tracking,

        // Spectral analysis configuration
        spectral_threshold: args.spectral_threshold,

        // AGC and window configuration
        agc_settling_time: args.agc_settling_time,
        window_overlap: args.window_overlap,
    };

    let driver = args.device_args.as_deref().unwrap_or(DEFAULT_DRIVER);
    let console_writer = Arc::new(DefaultConsoleWriter);
    let logger = Arc::new(DefaultLogger::new(args.verbose, format));

    // Enumerate devices and serialize them to strings for later re-instantiation
    let device_strings = soapysdr::enumerate(driver)?
        .into_iter()
        .map(|args| soapy::Device(format!("{}", args)))
        .collect::<Vec<soapy::Device>>();

    // Create and setup MainThread with device strings
    MainThread::new(config, console_writer, logger, device_strings)?.run(args.stations)
}
