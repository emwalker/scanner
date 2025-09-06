use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::{Arc, Mutex, mpsc::SyncSender};
use std::thread;
use tracing::{debug, info};

mod broadcast;
mod file;
mod fm;
mod freq_xlating_fir;
mod logging;
mod mpsc;
mod sdr;
mod soapy;
mod testing;
mod types;
use crate::broadcast::BroadcastSource;
use crate::types::{Candidate, Result, ScannerError};
use rustradio::graph::GraphRunner;

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
    candidate_tx: &SyncSender<(Candidate, f64)>,
    config: &ScanningConfig,
    sdr_manager: &Arc<Mutex<sdr::SdrManager>>,
) -> Result<()> {
    let stations = parse_stations(stations_str)?;
    debug!(
        message = "Scanning stations",
        stations = format!("{:?}", stations)
    );

    for station_freq in stations {
        sdr_manager.lock().unwrap().set_frequency(station_freq)?;

        let candidate = Candidate::Fm(fm::Candidate {
            frequency_hz: station_freq,
            peak_count: 1,
            max_magnitude: 0.0,
            avg_magnitude: 0.0,
            signal_strength: "Station".to_string(),
        });

        if config.debug_pipeline {
            debug!(
                message = "Station mode candidate created",
                station_frequency_mhz = station_freq / 1e6,
                sdr_center_frequency_mhz = station_freq / 1e6,
                frequency_offset_khz = 0.0
            );
        }

        candidate_tx.send((candidate, station_freq)).unwrap();
    }

    Ok(())
}

fn scan_band(
    config: &ScanningConfig,
    candidate_tx: &SyncSender<(Candidate, f64)>,
    sdr_manager: &Arc<Mutex<sdr::SdrManager>>,
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

        sdr_manager.lock().unwrap().set_frequency(*center_freq)?;
        let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();

        let peaks = fm::collect_peaks(config, sdr_rx, *center_freq)?;

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());

            if config.debug_pipeline {
                debug!(
                    message = "Band scanning window analysis",
                    window_number = i + 1,
                    window_center_mhz = center_freq / 1e6,
                    peaks_found = peaks.len()
                );

                for (peak_idx, peak) in peaks.iter().enumerate() {
                    debug!(
                        message = "Peak detected",
                        window_number = i + 1,
                        peak_index = peak_idx,
                        frequency_mhz = peak.frequency_hz / 1e6,
                        magnitude = peak.magnitude
                    );
                }
            }

            for candidate in fm::find_candidates(&peaks, config, *center_freq) {
                if config.debug_pipeline {
                    let frequency_offset = candidate.frequency_hz() - center_freq;
                    debug!(
                        message = "Candidate created",
                        candidate_frequency_mhz = candidate.frequency_hz() / 1e6,
                        window_center_mhz = center_freq / 1e6,
                        frequency_offset_khz = frequency_offset / 1e3,
                        signal_strength = match &candidate {
                            crate::types::Candidate::Fm(fm_candidate) =>
                                &fm_candidate.signal_strength,
                        }
                    );
                }

                candidate_tx.send((candidate, *center_freq)).unwrap();
            }
        } else {
            debug!("No peaks detected in this window");
            if config.debug_pipeline {
                debug!(
                    message = "Band scanning window analysis",
                    window_number = i + 1,
                    window_center_mhz = center_freq / 1e6,
                    peaks_found = 0
                );
            }
        }

        if config.exit_early {
            break;
        }
    }
    Ok(())
}

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Band {
    Fm,
    Aircraft,
    Ham2m,
    Weather,
    Marine,
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
        let usable_bandwidth = sample_rate * 0.8;
        let step_size = usable_bandwidth * 0.9;

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
    pub band: Band,
    pub capture_audio_duration: f64,
    pub capture_audio: Option<String>,
    pub capture_duration: f64,
    pub capture_iq: Option<String>,
    pub debug_pipeline: bool,
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
            audio_buffer_size: 8192, // Increased from 4K to 8K samples for better buffering
            audio_sample_rate: 48000,
            band: Band::Fm,
            capture_audio: None,
            capture_audio_duration: 3.0,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            driver: "driver=sdrplay".to_string(),
            duration: 3,
            exit_early: false,
            fft_size: 1024,
            peak_detection_threshold: 1.0,
            peak_scan_duration: None,
            print_candidates: false,
            samp_rate: 2_000_000.0,
            squelch_learning_duration: 2.0,
        }
    }
}

const DEFAULT_DRIVER: &str = "driver=sdrplay";

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

fn create_audio_stream(
    device: &cpal::Device,
    stream_config: &StreamConfig,
    audio_rx: std::sync::mpsc::Receiver<f32>,
) -> Result<cpal::Stream> {
    let err_fn = |err| debug!("Audio error: {}", err);

    let stream = device.build_output_stream(
        stream_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // More efficient batch processing - try to fill buffer in chunks
            let mut filled = 0;
            while filled < data.len() {
                match audio_rx.try_recv() {
                    Ok(audio_sample) => {
                        data[filled] = audio_sample.clamp(-1.0, 1.0);
                        filled += 1;
                    }
                    Err(_) => {
                        // Fill remaining with silence to avoid underruns
                        for sample in &mut data[filled..] {
                            *sample = 0.0;
                        }
                        break;
                    }
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn create_audio_fm_graph(
    signal: &types::Signal,
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    audio_tx: std::sync::mpsc::SyncSender<f32>,
    config: &ScanningConfig,
    center_freq: f64,
) -> Result<rustradio::graph::Graph> {
    use rustradio::{
        Float, blockchain,
        blocks::{QuadratureDemod, RationalResampler},
        fir,
        fir::FirFilter,
        window::WindowType,
    };

    let mut graph = rustradio::graph::Graph::new();
    let station_name = format!("{:.1}FM_Audio", signal.frequency_hz / 1e6);

    let (source_block, prev) = BroadcastSource::new(sdr_rx);
    graph.add(Box::new(source_block));

    let frequency_offset = signal.frequency_hz - center_freq;
    debug!(
        "Audio graph: signal {:.1} MHz, center {:.1} MHz, offset {:.1} kHz",
        signal.frequency_hz / 1e6,
        center_freq / 1e6,
        frequency_offset / 1e3
    );

    // Get optimized filter configuration for audio stage
    let filter_config = crate::fm::filter_config::FmFilterConfig::for_purpose(
        crate::fm::filter_config::FilterPurpose::Audio,
        config.samp_rate,
    );

    debug!(
        message = "Audio stage filter configuration",
        passband_khz = filter_config.channel_bandwidth / 1000.0,
        transition_khz = filter_config.transition_width / 1000.0,
        decimation = filter_config.decimation,
        estimated_taps = filter_config.estimated_taps,
        estimated_mflops = filter_config.estimated_mflops
    );

    let taps = fir::low_pass(
        config.samp_rate as f32,
        filter_config.cutoff_frequency(),
        filter_config.transition_width,
        &WindowType::Hamming,
    );

    let decimation = filter_config.decimation;
    let (freq_xlating_block, prev) = crate::freq_xlating_fir::FreqXlatingFir::with_real_taps(
        prev,
        &taps,
        frequency_offset as f32,
        config.samp_rate as f32,
        decimation,
    );
    graph.add(Box::new(freq_xlating_block));

    let decimated_samp_rate = config.samp_rate / decimation as f64;
    let quad_rate = decimated_samp_rate as f32;

    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8;
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    let (deemphasis_block, deemphasis_output_stream) =
        fm::deemph::Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    let prev = deemphasis_output_stream;

    let taps = fir::low_pass(quad_rate, 12_000.0f32, 8_000.0f32, &WindowType::Hamming);
    let prev = blockchain![graph, prev, FirFilter::new(prev, &taps)];

    let audio_rate = 48_000.0;
    let prev = blockchain![
        graph,
        prev,
        RationalResampler::<Float>::builder()
            .deci(quad_rate as usize)
            .interp(audio_rate as usize)
            .build(prev)?
    ];

    graph.add(Box::new(crate::mpsc::MpscSink::new(
        prev,
        audio_tx,
        station_name,
    )));

    Ok(graph)
}

fn process_signal_for_audio(
    signal: &types::Signal,
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    audio_tx: std::sync::mpsc::SyncSender<f32>,
    config: &ScanningConfig,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    debug!(
        "Creating audio processing pipeline for {:.1} MHz",
        signal.frequency_hz / 1e6
    );

    let mut audio_graph = create_audio_fm_graph(
        signal,
        sdr_rx,
        audio_tx,
        config,
        signal.detection_center_freq,
    )?;

    let duration = std::time::Duration::from_secs(config.duration);
    debug!("Playing audio for {:?}", duration);
    debug!("Starting audio graph thread...");

    let cancel_token = audio_graph.cancel_token();
    let graph_handle = std::thread::spawn(move || {
        debug!("Audio graph thread started, running graph...");
        if let Err(e) = audio_graph.run() {
            debug!("Audio graph error: {}", e);
        } else {
            debug!("Audio graph completed successfully");
        }
    });

    // Sleep with periodic cancellation checks instead of blocking sleep
    let check_interval = std::time::Duration::from_millis(100);
    let mut remaining = duration;

    while !remaining.is_zero() {
        let sleep_duration = std::cmp::min(remaining, check_interval);
        std::thread::sleep(sleep_duration);
        remaining = remaining.saturating_sub(sleep_duration);

        // Check if shutdown requested
        if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
            debug!("Shutdown requested during audio processing, stopping early");
            break;
        }
    }

    debug!("Cancelling audio graph...");
    cancel_token.cancel();
    debug!("Waiting for audio graph thread to finish...");
    let _ = graph_handle.join();
    debug!("Audio graph thread finished");

    debug!(
        "Finished playing audio for {:.1} MHz",
        signal.frequency_hz / 1e6
    );
    Ok(())
}

fn spawn_audio_thread(
    signal_rx: std::sync::mpsc::Receiver<types::Signal>,
    sdr_manager: Arc<Mutex<sdr::SdrManager>>,
    config: ScanningConfig,
) -> Result<thread::JoinHandle<Result<()>>> {
    let audio_sample_rate = config.audio_sample_rate;
    let audio_buffer_size = config.audio_buffer_size;

    let (audio_tx, audio_processing_rx) = std::sync::mpsc::sync_channel::<f32>(16384); // Increased from 4K to 16K samples (~340ms at 48kHz)

    let handle = thread::spawn(move || -> Result<()> {
        let (audio_device, supported_config) = setup_audio_device(audio_sample_rate)?;

        let sample_format = supported_config.sample_format();
        let mut stream_config: StreamConfig = supported_config.into();
        stream_config.buffer_size = cpal::BufferSize::Fixed(audio_buffer_size);

        let stream = match sample_format {
            SampleFormat::F32 => {
                create_audio_stream(&audio_device, &stream_config, audio_processing_rx)?
            }
            _ => return Err(ScannerError::Custom("Unsupported audio format".to_string())),
        };

        stream.play()?;
        debug!("Audio system ready - waiting for signals");

        // Shutdown flag for interruptible audio processing
        let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

        loop {
            // Use timeout to allow periodic checking for shutdown
            match signal_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(signal) => {
                    debug!(
                        "Processing signal for audio: {:.1} MHz, strength: {:.3}, modulation: {:?}",
                        signal.frequency_hz / 1e6,
                        signal.signal_strength,
                        signal.modulation
                    );
                    let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();

                    if let Err(e) = process_signal_for_audio(
                        &signal,
                        sdr_rx,
                        audio_tx.clone(),
                        &config,
                        shutdown_flag.clone(),
                    ) {
                        debug!("Error processing signal for audio: {}", e);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Timeout - continue loop to check for more signals
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    debug!("Signal channel disconnected, audio thread exiting");
                    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }

        Ok(())
    });
    Ok(handle)
}

fn spawn_scanning_thread(
    candidate_rx: std::sync::mpsc::Receiver<(Candidate, f64)>,
    config: ScanningConfig,
    signal_tx: std::sync::mpsc::SyncSender<types::Signal>,
    sdr_manager: Arc<Mutex<sdr::SdrManager>>,
) -> Result<thread::JoinHandle<Result<()>>> {
    let handle = thread::spawn(move || -> Result<()> {
        debug!("Scanning for transmissions ...");

        for (candidate, center_freq) in candidate_rx {
            if config.print_candidates {
                info!(
                    "candidate found at {:.1} MHz",
                    candidate.frequency_hz() / 1e6
                );
            } else {
                let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();
                candidate.analyze(&config, sdr_rx, center_freq, signal_tx.clone())?;
            }

            if config.exit_early {
                debug!("Early exit requested - stopping after first candidate.");
                break;
            }
        }

        debug!("All frequencies scanned.");
        drop(signal_tx);
        Ok(())
    });
    Ok(handle)
}

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

    #[arg(long)]
    exit_early: bool,

    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    debug_pipeline: bool,

    #[arg(long)]
    print_candidates: bool,

    #[arg(long, default_value_t = Format::Text)]
    format: Format,

    #[arg(long)]
    capture_iq: Option<String>,

    #[arg(long, default_value_t = 2.0)]
    capture_duration: f64,

    #[arg(long)]
    capture_audio: Option<String>,

    #[arg(long, default_value_t = 3.0)]
    capture_audio_duration: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    logging::init(args.verbose, args.format)?;

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
        exit_early: args.exit_early,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: args.peak_scan_duration,
        print_candidates: args.print_candidates,
        samp_rate: 2_000_000.0f64,
        squelch_learning_duration: 2.0,
    };

    soapysdr::configure_logging();
    info!("Scanning for stations ...");

    let (candidate_tx, candidate_rx) = std::sync::mpsc::sync_channel::<(Candidate, f64)>(100);
    let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<types::Signal>(100);

    let sdr_manager = sdr::SdrManager::new(config.driver.clone(), config.samp_rate)?;
    let sdr_manager_arc = Arc::new(Mutex::new(sdr_manager));

    let audio_handle = spawn_audio_thread(signal_rx, Arc::clone(&sdr_manager_arc), config.clone())?;

    let scanning_thread = spawn_scanning_thread(
        candidate_rx,
        config.clone(),
        signal_tx.clone(),
        Arc::clone(&sdr_manager_arc),
    )?;

    if let Some(stations_str) = args.stations {
        scan_stations(&stations_str, &candidate_tx, &config, &sdr_manager_arc)?;
    } else {
        scan_band(&config, &candidate_tx, &sdr_manager_arc)?;
    }

    drop(candidate_tx);
    scanning_thread.join().unwrap()?;

    // Drop signal_tx to disconnect the signal channel and allow audio thread to exit
    drop(signal_tx);
    audio_handle.join().unwrap()?;

    sdr_manager_arc.lock().unwrap().stop()?;

    info!("Scanning complete.");
    logging::flush();

    Ok(())
}
