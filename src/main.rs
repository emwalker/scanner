use clap::{Parser, ValueEnum};
use tracing::{debug, info};

mod broadcast;
mod file;
mod fm;
mod freq_xlating_fir;
mod frequency_tracking;
mod iq_capture;
mod logging;
mod mpsc;
mod sdr;
mod soapy;
mod testing;
mod types;
mod window;
use crate::types::{Band, Result, ScannerError, ScanningConfig};
use crate::window::Window;

fn parse_stations(stations_str: &str) -> Result<Vec<f64>> {
    stations_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| ScannerError::Custom(format!("Invalid station frequency: {}", s)))
        })
        .collect()
}

fn scan_stations(stations_str: &str, config: &ScanningConfig) -> Result<()> {
    let stations = parse_stations(stations_str)?;
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
        let window = Window::for_station(station_freq, station_idx + 1, total_stations);

        // Process using the full band scanning pipeline (peak detection, candidates, etc.)
        window.process(config)?;
    }

    Ok(())
}

fn scan_band(config: &ScanningConfig) -> Result<()> {
    // Clear any previously processed frequencies from earlier scans
    fm::clear_processed_frequencies();

    let window_centers = config.band.windows(config.samp_rate, config.window_overlap);
    debug!(
        "Scanning {} windows across {:?} band",
        window_centers.len(),
        config.band
    );

    let windows_to_process = match config.scanning_windows {
        Some(n) => n.min(window_centers.len()),
        None => window_centers.len(),
    };

    for (i, center_freq) in window_centers.iter().enumerate().take(windows_to_process) {
        let window = Window::new(*center_freq, i + 1, window_centers.len());
        window.process(config)?;
    }
    Ok(())
}

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Format {
    Json,
    Text,
    Log,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Json => write!(f, "json"),
            Format::Text => write!(f, "text"),
            Format::Log => write!(f, "log"),
        }
    }
}

pub struct Audio;

const DEFAULT_DRIVER: &str = "driver=sdrplay";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    device_args: Option<String>,

    #[arg(long)]
    stations: Option<String>,

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

    logging::init(args.verbose, format)?;

    let driver = args.device_args.unwrap_or(DEFAULT_DRIVER.into());
    let config = ScanningConfig {
        audio_buffer_size: 8192, // Increased from 4K to 8K samples for better buffering
        audio_sample_rate: 48000,
        band: args.band,
        capture_audio_duration: args.capture_audio_duration,
        capture_audio: args.capture_audio,
        capture_duration: args.capture_duration,
        capture_iq: args.capture_iq,
        debug_pipeline: args.debug_pipeline,
        driver: driver.clone(),
        duration: args.duration,
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

    soapysdr::configure_logging();
    info!("Scanning for stations ...");

    if let Some(stations_str) = args.stations {
        scan_stations(&stations_str, &config)?;
    } else {
        scan_band(&config)?;
    }

    info!("Scanning complete.");

    Ok(())
}
