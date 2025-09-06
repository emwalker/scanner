use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
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
use crate::broadcast::BroadcastSource;
use crate::types::{Band, Candidate, Result, ScannerError, ScanningConfig};
use rustradio::graph::GraphRunner;

/// Represents a frequency window for band scanning with complete lifecycle management
struct Window {
    center_freq: f64,
    window_num: usize,
    total_windows: usize,
}

impl Window {
    fn new(center_freq: f64, window_num: usize, total_windows: usize) -> Self {
        Self {
            center_freq,
            window_num,
            total_windows,
        }
    }

    /// Process this window completely: tune SDR, find candidates, run detection/audio, wait for completion
    fn process(
        &self,
        config: &ScanningConfig,
        signal_tx: &std::sync::mpsc::SyncSender<types::Signal>,
        sdr_manager: &Arc<Mutex<sdr::SdrManager>>,
    ) -> Result<()> {
        debug!(
            "Scanning window {} of {} at {:.1} MHz",
            self.window_num,
            self.total_windows,
            self.center_freq / 1e6
        );

        // Tune SDR to this window's center frequency
        sdr_manager
            .lock()
            .unwrap()
            .set_frequency(self.center_freq)?;
        let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();

        // Find candidates in this window
        let peaks = crate::fm::collect_peaks(config, sdr_rx, self.center_freq)?;
        let mut candidates = Vec::new();

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());

            if config.debug_pipeline {
                debug!(
                    message = "Band scanning window analysis",
                    window_number = self.window_num,
                    window_center_mhz = self.center_freq / 1e6,
                    peaks_found = peaks.len()
                );

                for (peak_idx, peak) in peaks.iter().enumerate() {
                    debug!(
                        message = "Peak detected",
                        window_number = self.window_num,
                        peak_index = peak_idx,
                        frequency_mhz = peak.frequency_hz / 1e6,
                        magnitude = peak.magnitude
                    );
                }
            }

            for candidate in crate::fm::find_candidates(&peaks, config, self.center_freq) {
                if config.debug_pipeline {
                    let frequency_offset = candidate.frequency_hz() - self.center_freq;
                    debug!(
                        message = "Candidate created",
                        candidate_frequency_mhz = candidate.frequency_hz() / 1e6,
                        window_center_mhz = self.center_freq / 1e6,
                        frequency_offset_khz = frequency_offset / 1e3,
                        signal_strength = match &candidate {
                            crate::types::Candidate::Fm(fm_candidate) =>
                                &fm_candidate.signal_strength,
                        }
                    );
                }

                candidates.push(candidate);
            }
        } else {
            debug!("No peaks detected in this window");
            if config.debug_pipeline {
                debug!(
                    message = "Band scanning window analysis",
                    window_number = self.window_num,
                    window_center_mhz = self.center_freq / 1e6,
                    peaks_found = 0
                );
            }
        }

        // Process all candidates in this window and wait for completion
        if !candidates.is_empty() {
            let candidate_count = candidates.len();
            let mut candidate_threads = Vec::new();

            for candidate in candidates {
                if config.print_candidates {
                    info!(
                        "candidate found at {:.1} MHz",
                        candidate.frequency_hz() / 1e6
                    );
                    continue; // Skip analysis in print-only mode
                }

                // Get fresh SDR receiver for each candidate
                let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();
                let signal_tx_clone = signal_tx.clone();
                let config_clone = config.clone();
                let center_freq = self.center_freq;

                // Spawn thread for this candidate's detection and audio processing
                let handle = thread::spawn(move || -> Result<()> {
                    candidate.analyze(&config_clone, sdr_rx, center_freq, signal_tx_clone)
                });

                candidate_threads.push(handle);
            }

            // Wait for all candidate threads to complete with timeout
            let window_timeout = Duration::from_secs(60); // Generous timeout per window
            let threads_completed =
                self.wait_for_threads_with_timeout(candidate_threads, window_timeout);

            debug!(
                "Window {} at {:.1} MHz: {}/{} candidates completed processing",
                self.window_num,
                self.center_freq / 1e6,
                threads_completed,
                candidate_count
            );
        }

        Ok(())
    }

    /// Wait for threads to complete with a timeout, returns number of threads that completed
    fn wait_for_threads_with_timeout(
        &self,
        threads: Vec<thread::JoinHandle<Result<()>>>,
        timeout: Duration,
    ) -> usize {
        use std::time::Instant;

        let start_time = Instant::now();
        let mut completed = 0;

        for (i, handle) in threads.into_iter().enumerate() {
            let remaining_time = timeout.saturating_sub(start_time.elapsed());

            if remaining_time.is_zero() {
                debug!("Thread {} timed out, not waiting for completion", i + 1);
                break;
            }

            // Try to join the thread
            match handle.join() {
                Ok(Ok(())) => {
                    completed += 1;
                    debug!("Thread {} completed successfully", i + 1);
                }
                Ok(Err(e)) => {
                    completed += 1;
                    debug!("Thread {} completed with error: {}", i + 1, e);
                }
                Err(_) => {
                    debug!("Thread {} panicked", i + 1);
                }
            }
        }

        completed
    }
}

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
    config: &ScanningConfig,
    signal_tx: &std::sync::mpsc::SyncSender<types::Signal>,
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

        if config.print_candidates {
            info!(
                "candidate found at {:.1} MHz",
                candidate.frequency_hz() / 1e6
            );
        } else {
            let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();
            candidate.analyze(config, sdr_rx, station_freq, signal_tx.clone())?;
        }
    }

    Ok(())
}

fn scan_band(
    config: &ScanningConfig,
    signal_tx: &std::sync::mpsc::SyncSender<types::Signal>,
    sdr_manager: &Arc<Mutex<sdr::SdrManager>>,
) -> Result<()> {
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
        window.process(config, signal_tx, sdr_manager)?;
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

        // Priority queue to collect and sort signals by frequency
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Debug)]
        struct OrderedSignal {
            signal: types::Signal,
            priority: std::cmp::Reverse<i64>, // Reverse for min-heap (lowest frequency first)
        }

        impl PartialEq for OrderedSignal {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }

        impl Eq for OrderedSignal {}

        impl PartialOrd for OrderedSignal {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for OrderedSignal {
            fn cmp(&self, other: &Self) -> Ordering {
                self.priority.cmp(&other.priority)
            }
        }

        let mut signal_queue: BinaryHeap<OrderedSignal> = BinaryHeap::new();
        let mut collection_timeout = std::time::Instant::now();
        let collection_window = std::time::Duration::from_millis(2000); // Collect signals for 2 seconds

        loop {
            // Use timeout to allow periodic checking for shutdown
            match signal_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(signal) => {
                    // Add signal to priority queue
                    let ordered_signal = OrderedSignal {
                        priority: std::cmp::Reverse(signal.frequency_hz as i64),
                        signal,
                    };
                    signal_queue.push(ordered_signal);

                    // Reset collection timeout when we get a new signal
                    collection_timeout = std::time::Instant::now();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Check if collection window has expired and we have signals to play
                    if collection_timeout.elapsed() >= collection_window && !signal_queue.is_empty()
                    {
                        // Play all collected signals in frequency order
                        while let Some(ordered_signal) = signal_queue.pop() {
                            let signal = ordered_signal.signal;
                            info!("playing {:.1} MHz", signal.frequency_hz / 1e6);
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
                        // Reset timeout for next collection window
                        collection_timeout = std::time::Instant::now();
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    debug!(
                        "Signal channel disconnected, playing remaining {} queued signals",
                        signal_queue.len()
                    );

                    // Play any remaining signals in the queue before exiting
                    while let Some(ordered_signal) = signal_queue.pop() {
                        let signal = ordered_signal.signal;
                        info!("playing {:.1} MHz", signal.frequency_hz / 1e6);
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

                    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }

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

    let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<types::Signal>(100);

    let sdr_manager = sdr::SdrManager::new(&config)?;
    let sdr_manager_arc = Arc::new(Mutex::new(sdr_manager));

    let audio_handle = spawn_audio_thread(signal_rx, Arc::clone(&sdr_manager_arc), config.clone())?;

    if let Some(stations_str) = args.stations {
        scan_stations(&stations_str, &config, &signal_tx, &sdr_manager_arc)?;
    } else {
        scan_band(&config, &signal_tx, &sdr_manager_arc)?;
    }

    // Drop signal_tx to disconnect the signal channel and allow audio thread to exit
    drop(signal_tx);
    audio_handle.join().unwrap()?;

    sdr_manager_arc.lock().unwrap().stop()?;

    info!("Scanning complete.");

    Ok(())
}
