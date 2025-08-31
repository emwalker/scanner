use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rustradio::blocks::*;
use rustradio::fir;
use rustradio::graph::{Graph, GraphRunner};
use rustradio::stream::ReadStream;
use rustradio::window::WindowType;
use rustradio::{Float, blockchain};
mod deemphasis;
mod mpsc_receiver_source;
mod mpsc_sender_sink;
use crate::deemphasis::Deemphasis;
use crate::mpsc_receiver_source::MpscReceiverSource;
use crate::mpsc_sender_sink::MpscSenderSink;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Start frequency in Hz (e.g., 88.1M or 88100000)
    #[arg(long)]
    start_freq: Option<f64>,

    /// Stop frequency in Hz (e.g., 108M or 108000000)
    #[arg(long)]
    stop_freq: Option<f64>,

    /// SDR device arguments (e.g., "driver=sdrplay")
    #[arg(long)]
    device_args: Option<String>,

    /// Specific frequency to tune to in Hz (e.g., 88.9 MHz)
    #[arg(long, default_value_t = 88.9e6)]
    tune_freq: f64,

    /// Duration (for testing)
    #[arg(long, default_value_t = 3)]
    duration: u64,
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
    samp_rate: f32,
    audio_sender: mpsc::SyncSender<f32>,
    channel_name: String,
) -> rustradio::Result<Graph> {
    let mut graph = Graph::new();

    // MPSC Source
    let (mpsc_source_block, prev) = MpscReceiverSource::new(source_receiver);
    graph.add(Box::new(mpsc_source_block));

    // RF Filter
    let taps = fir::low_pass_complex(samp_rate, 120_000.0f32, 25_000.0f32, &WindowType::Hamming);
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

    // Create MPSC channel for SDR samples
    let (sdr_tx, sdr_rx) = mpsc::sync_channel::<rustradio::Complex>(16384);

    let samp_rate = 1_000_000.0f32;

    // Create SDR graph and get its cancel token
    let mut sdr_graph = Graph::new();
    let sdr_graph_cancel_token = sdr_graph.cancel_token();

    // Spawn SDR thread
    let sdr_handle = thread::spawn(move || -> Result<()> {
        let driver = args
            .device_args
            .unwrap_or_else(|| "driver=sdrplay".to_string());
        eprintln!("Frequency {}, sample rate {}", args.tune_freq, samp_rate);

        // Create device first, then pass to builder (like working example)
        let dev = soapysdr::Device::new(&*driver)?;
        let (sdr_source_block, sdr_output_stream) =
            SoapySdrSource::builder(&dev, args.tune_freq, samp_rate as f64)
                .igain(1 as _)
                .build()?;

        sdr_graph.add(Box::new(sdr_source_block));
        let prev = sdr_output_stream;

        // Add MpscSenderSink to send SDR samples to the channel
        sdr_graph.add(Box::new(MpscSenderSink::new(prev, sdr_tx)));

        // Run the SDR graph
        sdr_graph
            .run()
            .map_err(|e| ScannerError::Custom(format!("SDR Graph error: {}", e)))?;
        Ok(())
    });

    // Create audio mixer system
    let mut audio_mixer = AudioMixer::new();

    // Create MPSC channel for this station
    let (audio_tx, audio_rx) = mpsc::sync_channel::<f32>(16384);
    let station_name = format!("{:.1}FM", args.tune_freq / 1e6);

    // Add station to mixer
    audio_mixer.add_channel(station_name.clone(), args.tune_freq, audio_rx, 1.0);

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

    // Join SDR thread
    sdr_handle.join().unwrap()?;

    println!("Scanner finished.");
    Ok(())
}
