use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::{Arc, Mutex, mpsc::SyncSender};
use std::thread;
use tracing::{debug, info};

mod file;
mod fm;
mod logging;
mod mpsc;
mod soapy;
#[cfg(test)]
mod testing;
mod types;
use crate::types::{Candidate, Result, ScannerError};

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

fn scan_stations(
    stations_str: &str,
    candidate_tx: &SyncSender<Candidate>,
    config: &ScanningConfig,
) -> Result<()> {
    let stations = parse_stations(stations_str)?;
    debug!(
        message = "Scanning stations",
        stations = format!("{:?}", stations)
    );

    for station_freq in stations {
        // If I/Q capture is requested, perform peak collection to trigger capture
        if config.capture_iq.is_some() {
            debug!(
                message = "Capturing I/Q samples for station",
                frequency_hz = station_freq
            );
            let _peaks = fm::collect_peaks(config, &config.driver, station_freq)?;
        }

        let candidate = Candidate::Fm(fm::Candidate {
            frequency_hz: station_freq,
            peak_count: 1,
            max_magnitude: 0.0,
            avg_magnitude: 0.0,
            signal_strength: "Station".to_string(),
        });
        candidate_tx.send(candidate).unwrap();
    }

    Ok(())
}

fn scan_band(
    config: &ScanningConfig,
    driver: &str,
    candidate_tx: &SyncSender<Candidate>,
) -> Result<()> {
    let windows = config.band.windows(config.samp_rate);
    debug!(
        "Scanning {} windows across {:?} band",
        windows.len(),
        config.band
    );

    for (i, center_freq) in windows.iter().enumerate() {
        debug!(
            "Scanning window {} of {} at {:.1} MHz",
            i + 1,
            windows.len(),
            center_freq / 1e6
        );

        let peaks = fm::collect_peaks(config, driver, *center_freq)?;

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());
            for candidate in fm::find_candidates(&peaks, config, *center_freq) {
                candidate_tx.send(candidate).unwrap();
            }
        } else {
            debug!("No peaks detected in this window");
        }

        if config.exit_early {
            break;
        }
    }
    Ok(())
}

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

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Format {
    /// JSON structured logging format
    Json,
    /// Simple text logging format
    Text,
    /// Standard log format with timestamps and levels
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

    pub fn windows(&self, sample_rate: f64) -> Vec<f64> {
        let (start_freq, end_freq) = self.frequency_range();
        let usable_bandwidth = sample_rate * 0.8; // Use 80% of bandwidth to avoid edge effects
        let step_size = usable_bandwidth * 0.9; // 10% overlap between windows

        let mut windows = Vec::new();
        let mut center_freq = start_freq + (usable_bandwidth / 2.0);

        while center_freq - (usable_bandwidth / 2.0) < end_freq {
            windows.push(center_freq);
            center_freq += step_size;
        }

        windows
    }
}

pub struct Audio;

#[derive(Clone)]
pub struct ScanningConfig {
    pub audio_buffer_size: u32,
    pub audio_sample_rate: u32,
    pub audo_mutex: Arc<Mutex<Audio>>,
    pub band: Band,
    pub capture_audio_duration: f64,
    pub capture_audio: Option<String>,
    pub capture_duration: f64,
    pub capture_iq: Option<String>,
    pub driver: String,
    pub duration: u64,
    pub exit_early: bool,
    pub fft_size: usize,
    pub peak_detection_threshold: f32,
    pub peak_scan_duration: Option<f64>,
    pub print_candidates: bool,
    pub samp_rate: f64,
    pub squelch_learning_duration: f32,
}

impl Default for ScanningConfig {
    fn default() -> Self {
        Self {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            audo_mutex: Arc::new(Mutex::new(Audio)),
            band: Band::Fm,
            driver: "driver=sdrplay".to_string(),
            duration: 3,
            exit_early: false,
            fft_size: 1024,
            peak_detection_threshold: 1.0,
            peak_scan_duration: None,
            print_candidates: false,
            samp_rate: 1_000_000.0,
            capture_iq: None,
            capture_duration: 2.0,
            capture_audio: None,
            capture_audio_duration: 3.0,
            squelch_learning_duration: 2.0,
        }
    }
}

const DEFAULT_DRIVER: &str = "driver=sdrplay";

/// Configure audio device and find compatible configuration
fn setup_audio_device(
    audio_sample_rate: u32,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();
    let audio_device = host
        .default_output_device()
        .expect("no output device available");

    let supported_configs_range = audio_device
        .supported_output_configs()
        .expect("error while querying configs");

    let supported_config = supported_configs_range
        .filter(|d| d.sample_format() == SampleFormat::F32)
        .find(|d| {
            d.min_sample_rate().0 <= audio_sample_rate && d.max_sample_rate().0 >= audio_sample_rate
        })
        .expect("no supported config found")
        .with_sample_rate(cpal::SampleRate(audio_sample_rate));

    Ok((audio_device, supported_config))
}

/// Create audio stream with callback for processing samples
fn create_audio_stream(
    device: &cpal::Device,
    stream_config: &StreamConfig,
    audio_rx: std::sync::mpsc::Receiver<f32>,
) -> Result<cpal::Stream> {
    let err_fn = |err| debug!("Audio error: {}", err);

    let stream = device.build_output_stream(
        stream_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // Simple: try to fill the entire buffer from the shared channel
            for sample in data.iter_mut() {
                match audio_rx.try_recv() {
                    Ok(audio_sample) => *sample = audio_sample.clamp(-1.0, 1.0),
                    Err(_) => *sample = 0.0, // Silence when no audio available
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn spawn_audio_thread(
    audio_rx: std::sync::mpsc::Receiver<f32>,
    config: &ScanningConfig,
) -> Result<thread::JoinHandle<Result<()>>> {
    let audio_sample_rate = config.audio_sample_rate;
    let audio_buffer_size = config.audio_buffer_size;

    let handle = thread::spawn(move || -> Result<()> {
        // Set up audio device and configuration
        let (audio_device, supported_config) = setup_audio_device(audio_sample_rate)?;

        let sample_format = supported_config.sample_format();
        let mut stream_config: StreamConfig = supported_config.into();
        stream_config.buffer_size = cpal::BufferSize::Fixed(audio_buffer_size);

        // Create and start audio stream
        let stream = match sample_format {
            SampleFormat::F32 => create_audio_stream(&audio_device, &stream_config, audio_rx)?,
            _ => return Err(ScannerError::Custom("Unsupported audio format".to_string())),
        };

        stream.play()?;

        // Keep audio thread alive
        debug!("Audio system ready");
        thread::park();
        Ok(())
    });
    Ok(handle)
}

fn spawn_scanning_thread(
    candidate_rx: std::sync::mpsc::Receiver<Candidate>,
    config: ScanningConfig,
    shared_audio_tx: std::sync::mpsc::SyncSender<f32>,
    audio_handle: thread::JoinHandle<Result<()>>,
) -> Result<thread::JoinHandle<Result<()>>> {
    let handle = thread::spawn(move || -> Result<()> {
        debug!("Scanning for transmissions ...");

        for candidate in candidate_rx {
            if config.print_candidates {
                info!(
                    "candidate found at {:.1} MHz",
                    candidate.frequency_hz() / 1e6
                );
            } else {
                candidate.analyze(&config, shared_audio_tx.clone())?;
            }

            if config.exit_early {
                debug!("Early exit requested - stopping after first candidate.");
                break;
            }
        }

        debug!("All frequencies scanned.");
        audio_handle.thread().unpark();
        let _ = audio_handle.join();
        Ok(())
    });
    Ok(handle)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SDR device arguments (e.g., "driver=sdrplay")
    #[arg(long)]
    device_args: Option<String>,

    /// Comma-delimited list of stations to scan (e.g., "88.9e6,92.3e6")
    #[arg(long)]
    stations: Option<String>,

    /// Frequency band to scan
    #[arg(long, default_value_t = Band::Fm)]
    band: Band,

    /// Duration (for testing)
    #[arg(long, default_value_t = 3)]
    duration: u64,

    /// Duration for peak detection scan (seconds)
    #[arg(long)]
    peak_scan_duration: Option<f64>,

    /// Exit after analyzing the first candidate station
    #[arg(long)]
    exit_early: bool,

    /// Enable verbose output from libraries
    #[arg(long)]
    verbose: bool,

    /// Print candidate stations instead of analyzing them
    #[arg(long)]
    print_candidates: bool,

    /// Output format for logs
    #[arg(long, default_value_t = Format::Text)]
    format: Format,

    /// Capture I/Q samples to file for testing
    #[arg(long)]
    capture_iq: Option<String>,

    /// Duration to capture I/Q samples (seconds)
    #[arg(long, default_value_t = 2.0)]
    capture_duration: f64,

    /// Capture demodulated audio to file for squelch testing
    #[arg(long)]
    capture_audio: Option<String>,

    /// Duration to capture audio samples (seconds)
    #[arg(long, default_value_t = 3.0)]
    capture_audio_duration: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    logging::init(args.verbose, args.format)?;

    let driver = args.device_args.unwrap_or(DEFAULT_DRIVER.into());
    let config = ScanningConfig {
        audio_buffer_size: 4096,
        audio_sample_rate: 48000,
        audo_mutex: Arc::new(Mutex::new(Audio)),
        band: args.band,
        capture_audio_duration: args.capture_audio_duration,
        capture_audio: args.capture_audio,
        capture_duration: args.capture_duration,
        capture_iq: args.capture_iq,
        driver: driver.clone(),
        duration: args.duration,
        exit_early: args.exit_early,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: args.peak_scan_duration,
        print_candidates: args.print_candidates,
        samp_rate: 1_000_000.0f64,
        squelch_learning_duration: 2.0,
    };

    // Configure SoapySDR logging (required by rustradio)
    soapysdr::configure_logging();
    info!("Scanning for stations ...");

    // Single shared audio channel to which all candidates write
    let (audio_tx, audio_rx) = std::sync::mpsc::sync_channel::<f32>(config.samp_rate as _);
    // MPSC channel for candidate stations
    let (candidate_tx, candidate_rx) = std::sync::mpsc::sync_channel::<Candidate>(100);
    // Audio playback thread with simple blocking receive
    let audio_handle = spawn_audio_thread(audio_rx, &config)?;

    let scanning_thread =
        spawn_scanning_thread(candidate_rx, config.clone(), audio_tx.clone(), audio_handle)?;

    if let Some(stations_str) = args.stations {
        scan_stations(&stations_str, &candidate_tx, &config)?;
    } else {
        scan_band(&config, &driver, &candidate_tx)?;
    }

    // Wait for scanning to complete
    drop(candidate_tx);
    let _ = scanning_thread.join();

    info!("Scanning complete.");
    logging::flush();

    Ok(())
}
