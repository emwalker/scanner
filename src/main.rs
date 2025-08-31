use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rustradio::blocks::*;
use rustradio::fir;
use rustradio::graph::{CancellationToken, Graph, GraphRunner};
use rustradio::stream::ReadStream;
use rustradio::window::WindowType;
use rustradio::{Complex, Float, blockchain};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use thiserror::Error;

mod deemphasis;
mod mpsc_receiver_source;
mod mpsc_sender_sink;
use crate::deemphasis::Deemphasis;
use crate::mpsc_receiver_source::MpscReceiverSource;
use crate::mpsc_sender_sink::MpscSenderSink;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SDR device arguments (e.g., "driver=sdrplay")
    #[arg(long)]
    device_args: Option<String>,

    /// Center frequency to tune to in Hz (e.g., 88.9 MHz)
    #[arg(long, default_value_t = 88.9e6)]
    center_freq: f64,

    /// Duration (for testing)
    #[arg(long, default_value_t = 3)]
    duration: u64,

    /// Duration for peak detection scan (seconds)
    #[arg(long, default_value_t = 1)]
    peak_scan_duration: u64,
}

#[derive(Error, Debug)]
enum ScannerError {
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

type Result<T> = std::result::Result<T, ScannerError>;

#[derive(Debug, Clone)]
struct Peak {
    frequency_hz: f64,
    magnitude: f32,
}

#[derive(Debug, Clone)]
struct Candidate {
    frequency_mhz: f64,
    peak_count: usize,
    max_magnitude: f32,
    avg_magnitude: f32,
    signal_strength: String,
}

/// Collect RF peaks synchronously by directly interfacing with the SDR device
/// This performs FFT analysis to detect spectral peaks above the threshold
fn collect_peaks(
    device_args: &str,
    center_freq: f64,
    sample_rate: f64,
    duration_secs: u64,
    fft_size: usize,
    threshold: f32,
) -> Result<Vec<Peak>> {
    println!(
        "Starting peak detection scan for {} seconds...",
        duration_secs
    );

    // Initialize SDR device and stream
    let dev = soapysdr::Device::new(device_args)?;
    let mut rxstream = dev.rx_stream::<Complex>(&[0])?;
    rxstream.activate(None)?;

    // Prepare FFT processing
    let mut peaks = Vec::new();
    let mut fft_buffer = vec![rustfft::num_complex::Complex32::default(); fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Calculate sampling parameters
    let samples_per_second = sample_rate as usize;
    let total_samples_needed = samples_per_second * duration_secs as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); fft_size];

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        match rxstream.read(&mut [&mut read_buffer], 1000000) {
            Ok(samples_read) => {
                if samples_read == 0 {
                    continue;
                }

                // Copy samples to FFT buffer and process
                for (i, sample) in read_buffer
                    .iter()
                    .take(samples_read.min(fft_size))
                    .enumerate()
                {
                    fft_buffer[i] = rustfft::num_complex::Complex32::new(sample.re, sample.im);
                }

                fft.process(&mut fft_buffer);
                let magnitudes: Vec<f32> = fft_buffer.iter().map(|c| c.norm_sqr()).collect();

                // Detect peaks: local maxima above threshold
                for i in 1..magnitudes.len() - 1 {
                    if magnitudes[i] > threshold
                        && magnitudes[i] > magnitudes[i - 1]
                        && magnitudes[i] > magnitudes[i + 1]
                    {
                        // Convert FFT bin to frequency
                        let freq_offset = (i as f64 / fft_size as f64) * sample_rate;
                        let freq_hz = center_freq - (sample_rate / 2.0) + freq_offset;

                        peaks.push(Peak {
                            frequency_hz: freq_hz,
                            magnitude: magnitudes[i],
                        });
                    }
                }

                samples_collected += samples_read;
            }
            Err(e) => {
                eprintln!("Error reading from SDR: {}", e);
                break;
            }
        }
    }

    rxstream.deactivate(None)?;
    println!("Peak detection scan complete. Found {} peaks.", peaks.len());
    Ok(peaks)
}

/// Analyze spectral characteristics around a frequency to determine if it's a main lobe or sidelobe
/// Main lobes are wider and have characteristic spectral patterns compared to sidelobes
fn analyze_spectral_characteristics(
    peaks: &[Peak],
    target_freq_mhz: f64,
    _sample_rate: f64,
    center_freq: f64,
) -> (f32, String) {
    let target_freq_hz = target_freq_mhz * 1e6;

    // Find peaks within ±200 kHz of target frequency (wider than FM channel spacing)
    let analysis_range_hz = 200000.0;
    let nearby_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|peak| (peak.frequency_hz - target_freq_hz).abs() <= analysis_range_hz)
        .collect();

    if nearby_peaks.is_empty() {
        return (0.0, "No signal".to_string());
    }

    // Sort peaks by frequency for width analysis
    let mut sorted_peaks = nearby_peaks.clone();
    sorted_peaks.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());

    // Calculate spectral width characteristics
    let peak_count = sorted_peaks.len();
    let freq_span_khz = if peak_count > 1 {
        (sorted_peaks.last().unwrap().frequency_hz - sorted_peaks.first().unwrap().frequency_hz)
            / 1000.0
    } else {
        0.0
    };

    // Find the strongest peak in the group (should be the main signal)
    let max_magnitude = sorted_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Calculate average magnitude
    let avg_magnitude = sorted_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32;

    // Main lobe characteristics analysis
    let mut score: f64 = 0.0;
    let mut analysis_notes = Vec::new();

    // 1. Peak density analysis (main lobes have consistent energy distribution)
    let peak_density = peak_count as f64 / freq_span_khz.max(1.0);
    if peak_density > 20.0 && peak_density < 200.0 {
        score += 0.3;
        analysis_notes.push("Good peak density");
    } else if peak_density > 200.0 {
        score -= 0.2; // Too many peaks suggests broadband interference
        analysis_notes.push("High peak density (interference?)");
    }

    // 2. Frequency span analysis (main lobes have characteristic widths)
    // FM broadcast stations typically show energy across 150-200 kHz
    if freq_span_khz > 80.0 && freq_span_khz < 250.0 {
        score += 0.3;
        analysis_notes.push("Appropriate spectral width");
    } else if freq_span_khz < 30.0 {
        score -= 0.3; // Too narrow suggests sidelobe
        analysis_notes.push("Narrow spectral width (sidelobe?)");
    }

    // 3. Signal strength and consistency
    let magnitude_ratio = max_magnitude / avg_magnitude.max(1.0);
    if magnitude_ratio < 3.0 {
        score += 0.2; // Consistent energy suggests main lobe
        analysis_notes.push("Consistent energy");
    } else if magnitude_ratio > 10.0 {
        score -= 0.1; // Single spike suggests sidelobe
        analysis_notes.push("Sharp peak (possible sidelobe)");
    }

    // 4. Distance from center frequency (closer = more likely to be legitimate)
    let center_freq_mhz = center_freq / 1e6;
    let dist_from_center_mhz = (target_freq_mhz - center_freq_mhz).abs();
    if dist_from_center_mhz <= 0.1 {
        score += 0.4; // Strong bonus for center frequency
        analysis_notes.push("Near center freq");
    } else if dist_from_center_mhz <= 0.3 {
        score += 0.1; // Moderate bonus for nearby frequencies
    } else if dist_from_center_mhz > 0.4 {
        score -= 0.2; // Penalty for distant frequencies
        analysis_notes.push("Far from center");
    }

    // 5. Absolute signal strength
    if max_magnitude > 500.0 {
        score += 0.2;
        analysis_notes.push("Strong signal");
    } else if max_magnitude < 100.0 {
        score -= 0.1;
        analysis_notes.push("Weak signal");
    }

    let analysis_summary = analysis_notes.join(", ");
    (score.clamp(0.0, 1.0) as f32, analysis_summary)
}

/// Detect FM radio stations using spectral analysis with main lobe vs sidelobe discrimination
/// This approach analyzes spectral characteristics like peak width, density, and shape
fn find_candidates(
    peaks: &[Peak],
    center_freq: f64,
    sample_rate: f64,
    candidate_tx: &mpsc::SyncSender<Candidate>,
) {
    println!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    // Calculate the frequency range we scanned based on center freq and sample rate
    let scan_range_mhz = sample_rate / 2e6; // Half sample rate in MHz (Nyquist)
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    println!(
        "Analyzing spectral patterns in range: {:.1} - {:.1} MHz",
        freq_start_mhz, freq_end_mhz
    );

    // Generate possible FM frequencies within our scan range
    // FM stations are at odd tenth frequencies: 88.1, 88.3, 88.5, etc.
    let mut fm_freq = (freq_start_mhz * 10.0).ceil() / 10.0;
    if (fm_freq * 10.0) as i32 % 2 == 0 {
        fm_freq += 0.1; // Make it an odd tenth
    }

    while fm_freq <= freq_end_mhz {
        print!("Analyzing {:.1} MHz... ", fm_freq);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, sample_rate, center_freq);

        println!("score: {:.3} ({})", spectral_score, analysis_summary);

        // Only consider frequencies with significant spectral score
        if spectral_score > 0.3 {
            // Map spectral score to traditional signal strength categories
            let signal_strength = if spectral_score > 0.8 {
                "Strong"
            } else if spectral_score > 0.6 {
                "Medium"
            } else {
                "Weak"
            };

            // Find relevant peaks for this frequency
            let tolerance_mhz = 0.1;
            let nearby_peaks: Vec<&Peak> = peaks
                .iter()
                .filter(|peak| {
                    let peak_freq_mhz = peak.frequency_hz / 1e6;
                    (peak_freq_mhz - fm_freq).abs() <= tolerance_mhz
                })
                .collect();

            let peak_count = nearby_peaks.len();
            let max_magnitude = nearby_peaks
                .iter()
                .map(|p| p.magnitude)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let avg_magnitude = if peak_count > 0 {
                nearby_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32
            } else {
                0.0
            };

            candidate_tx
                .send(Candidate {
                    frequency_mhz: fm_freq,
                    peak_count,
                    max_magnitude,
                    avg_magnitude,
                    signal_strength: signal_strength.to_string(),
                })
                .expect("Failed to send candidate");
        }

        fm_freq = (fm_freq * 10.0 + 2.0) / 10.0; // Next odd tenth (add 0.2)
    }
}

// New architecture: Station-based audio mixing
struct Channel {
    receiver: mpsc::Receiver<f32>,
    volume: f32,
    name: String,
    frequency: f64,
}

struct AudioMixer {
    channels: Vec<Channel>,
}

impl AudioMixer {
    fn new() -> Self {
        AudioMixer {
            channels: Vec::new(),
        }
    }

    fn add_channel(
        &mut self,
        name: String,
        frequency: f64,
        receiver: mpsc::Receiver<f32>,
        volume: f32,
    ) {
        self.channels.push(Channel {
            receiver,
            volume,
            name,
            frequency,
        });
    }

    fn mix_samples(&mut self, output_buffer: &mut [f32]) {
        for sample_out in output_buffer.iter_mut() {
            let mut mixed_sample = 0.0f32;

            for channel in &self.channels {
                match channel.receiver.try_recv() {
                    Ok(sample) => {
                        mixed_sample += sample * channel.volume;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // Channel not producing audio right now - add silence
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        println!(
                            "Channel '{}' ({:.1} MHz) disconnected",
                            channel.name,
                            channel.frequency / 1e6
                        );
                    }
                }
            }
            *sample_out = mixed_sample.clamp(-1.0, 1.0);
        }
    }
}

/// Rust Radio sink that pushes samples to an MPSC channel
struct MpscSink {
    src: ReadStream<Float>,
    sender: mpsc::SyncSender<f32>,
    channel_name: String,
}

impl MpscSink {
    fn new(src: ReadStream<Float>, sender: mpsc::SyncSender<f32>, channel_name: String) -> Self {
        MpscSink {
            src,
            sender,
            channel_name,
        }
    }
}

impl rustradio::block::BlockName for MpscSink {
    fn block_name(&self) -> &str {
        "MpscSink"
    }
}

impl rustradio::block::BlockEOF for MpscSink {
    fn eof(&mut self) -> bool {
        self.src.eof()
    }
}

impl rustradio::block::Block for MpscSink {
    fn work(&mut self) -> rustradio::Result<rustradio::block::BlockRet<'_>> {
        let (input_buf, _) = self.src.read_buf()?;
        let samples = input_buf.slice();

        // Send samples to MPSC channel
        let mut consumed = 0;
        for &sample in samples {
            if self.sender.send(sample).is_err() {
                // Channel is full - this provides backpressure to the entire graph
                println!(
                    "MPSC channel full for {}, backpressuring graph",
                    self.channel_name
                );
                break;
            }
            consumed += 1;
        }

        input_buf.consume(consumed);

        if consumed > 0 {
            Ok(rustradio::block::BlockRet::Again)
        } else {
            Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1))
        }
    }
}

// Create rustradio graph for a station (matches the scanner example exactly)
fn create_station_graph(
    source_receiver: mpsc::Receiver<rustradio::Complex>,
    samp_rate: f64,
    audio_sender: mpsc::SyncSender<f32>,
    channel_name: String,
) -> rustradio::Result<Graph> {
    let mut graph = Graph::new();

    // MPSC Source
    let (mpsc_source_block, prev) = MpscReceiverSource::new(source_receiver);
    graph.add(Box::new(mpsc_source_block));

    // RF Filter
    let taps = fir::low_pass_complex(
        samp_rate as _,
        120_000.0f32,
        25_000.0f32,
        &WindowType::Hamming,
    );
    let prev = blockchain![graph, prev, FftFilter::new(prev, taps)];

    // Resample to 100kHz
    let quad_rate = 240_000.0;
    let prev = blockchain![
        graph,
        prev,
        RationalResampler::<rustradio::Complex>::builder()
            .deci(samp_rate as usize)
            .interp(quad_rate as usize)
            .build(prev)?
    ];

    // Quadrature demodulation
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, 1.0)];

    // FM Deemphasis
    let (deemphasis_block, deemphasis_output_stream) = Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    let prev = deemphasis_output_stream;

    // Audio filter
    let taps = fir::low_pass(
        quad_rate,
        15_000.0f32,
        5_000.0f32,
        &WindowType::BlackmanHarris,
    );
    let prev = blockchain![graph, prev, FirFilter::new(prev, &taps)];

    // Resample to 48kHz audio
    let audio_rate = 48_000.0;
    let prev = blockchain![
        graph,
        prev,
        RationalResampler::<Float>::builder()
            .deci(quad_rate as usize)
            .interp(audio_rate as usize)
            .build(prev)?
    ];

    // Our custom MPSC sink
    graph.add(Box::new(MpscSink::new(prev, audio_sender, channel_name)));

    Ok(graph)
}

fn spawn_sdr_producer_thread(
    device_args: Option<String>,
    center_freq: f64,
    samp_rate: f64,
    sdr_tx: mpsc::SyncSender<rustradio::Complex>,
) -> Result<(thread::JoinHandle<Result<()>>, CancellationToken)> {
    let mut sdr_graph = Graph::new();
    let sdr_graph_cancel_token = sdr_graph.cancel_token();

    let sdr_handle = thread::spawn(move || -> Result<()> {
        let driver = device_args.unwrap_or_else(|| "driver=sdrplay".to_string());
        eprintln!("Frequency {}, sample rate {}", center_freq, samp_rate);

        let dev = soapysdr::Device::new(&*driver)?;
        let (sdr_source_block, sdr_output_stream) =
            SoapySdrSource::builder(&dev, center_freq, samp_rate)
                .igain(1 as _)
                .build()?;

        sdr_graph.add(Box::new(sdr_source_block));
        let prev = sdr_output_stream;

        sdr_graph.add(Box::new(MpscSenderSink::new(prev, sdr_tx)));

        sdr_graph
            .run()
            .map_err(|e| ScannerError::Custom(format!("SDR Graph error: {}", e)))?;
        Ok(())
    });

    Ok((sdr_handle, sdr_graph_cancel_token))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Configure SoapySDR logging (required by rustradio)
    soapysdr::configure_logging();
    stderrlog::new()
        .module(module_path!())
        .module("rustradio")
        .quiet(false)
        .verbosity(11)
        .timestamp(stderrlog::Timestamp::Second)
        .init()?;

    println!("Starting RustRadio-based scanner...");

    // Create audio mixer system
    let mut audio_mixer = AudioMixer::new();

    // Create MPSC channel for this station
    let (audio_tx, audio_rx) = mpsc::sync_channel::<f32>(16384);
    let station_name = format!("{:.1}FM", args.center_freq / 1e6);

    // Create MPSC channel for candidate stations
    let (candidate_tx, candidate_rx) = mpsc::sync_channel::<Candidate>(100);

    // Collect peaks synchronously before starting rustradio graph
    let driver = args
        .device_args
        .clone()
        .unwrap_or_else(|| "driver=sdrplay".to_string());
    let samp_rate = 1_000_000.0f64;
    let fft_size = 1024;
    let threshold = 1.0;

    let peaks = collect_peaks(
        &driver,
        args.center_freq,
        samp_rate,
        args.peak_scan_duration,
        fft_size,
        threshold,
    )?;

    // Analyze peaks to infer likely FM stations
    if peaks.is_empty() {
        println!("No peaks detected above threshold.");
    } else {
        find_candidates(&peaks, args.center_freq, samp_rate, &candidate_tx.clone());
        println!();
    }

    // Create MPSC channel for SDR samples
    let (sdr_tx, sdr_rx) = mpsc::sync_channel::<rustradio::Complex>(16384);

    // Spawn thread to process candidate stations
    let _candidate_processor_handle = thread::spawn(move || {
        println!("Candidate processor thread started.");
        for candidate in candidate_rx {
            println!(
                "[Candidate Processor] Received: {:.1} MHz - {} signal ({} peaks, max: {:.1}, avg: {:.1})",
                candidate.frequency_mhz,
                candidate.signal_strength,
                candidate.peak_count,
                candidate.max_magnitude,
                candidate.avg_magnitude
            );
            // In the future, this is where we'll spawn rustradio threads
        }
        println!("Candidate processor thread finished.");
    });

    // Spawn SDR thread
    let (sdr_handle, sdr_graph_cancel_token) = spawn_sdr_producer_thread(
        args.device_args.clone(),
        args.center_freq,
        samp_rate,
        sdr_tx,
    )?;

    // Add station to mixer
    audio_mixer.add_channel(station_name.clone(), args.center_freq, audio_rx, 1.0);

    // Create rustradio graph for the station
    let mut graph = create_station_graph(sdr_rx, samp_rate, audio_tx, station_name.clone())?;
    let main_graph_cancel_token = graph.cancel_token(); // Get main graph's cancel token

    // Set up audio playback thread
    let audio_handle = thread::spawn(move || -> Result<()> {
        // Set up CPAL audio
        let host = cpal::default_host();
        let audio_device = host
            .default_output_device()
            .expect("no output device available");
        let supported_configs_range = audio_device
            .supported_output_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .filter(|d| d.sample_format() == SampleFormat::F32)
            .find(|d| d.min_sample_rate().0 <= 48000 && d.max_sample_rate().0 >= 48000)
            .expect("no supported config found")
            .with_sample_rate(cpal::SampleRate(48000));

        let sample_format = supported_config.sample_format();
        let mut config: StreamConfig = supported_config.into();
        config.buffer_size = cpal::BufferSize::Fixed(4096);

        let err_fn = |err| eprintln!("Audio error: {}", err);
        let audio_mixer_arc = Arc::new(Mutex::new(audio_mixer));
        let mixer_clone = audio_mixer_arc.clone();

        let stream = match sample_format {
            SampleFormat::F32 => audio_device.build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if let Ok(mut mixer) = mixer_clone.lock() {
                        mixer.mix_samples(data);
                    }
                },
                err_fn,
                None,
            )?,
            _ => return Err(ScannerError::Custom("Unsupported audio format".to_string())),
        };

        stream.play()?;

        // Keep audio thread alive
        println!("Audio system ready - playing {}", station_name);
        thread::park();
        Ok(())
    });

    // Set up timer thread
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(args.duration));
        println!("{} second timer expired, stopping graph.", args.duration);
        main_graph_cancel_token.cancel();
        sdr_graph_cancel_token.cancel();
    });

    // Run the rustradio graph in the main thread
    println!("Starting RustRadio graph...");
    graph
        .run()
        .map_err(|e| ScannerError::Custom(format!("Graph error: {}", e)))?;

    // Unpark and join the audio thread to allow graceful shutdown
    audio_handle.thread().unpark();
    let _ = audio_handle.join();

    // Join candidate processor thread
    // candidate_processor_handle.join().unwrap();

    // Join SDR thread with timeout to prevent hanging
    let sdr_join_handle = std::thread::spawn(move || {
        sdr_handle.join().unwrap_or_else(|_| {
            eprintln!("SDR thread panicked during shutdown");
            Ok(())
        })
    });

    // Wait for SDR thread to finish with timeout
    let timeout_duration = Duration::from_secs(2);
    let start_time = std::time::Instant::now();

    loop {
        if sdr_join_handle.is_finished() {
            sdr_join_handle.join().unwrap()?;
            break;
        }
        if start_time.elapsed() > timeout_duration {
            eprintln!("SDR thread failed to shut down within timeout, forcing exit");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    println!("Scanner finished.");
    Ok(())
}
