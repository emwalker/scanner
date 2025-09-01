use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::thread;

mod fm;
mod mpsc;
mod types;
use crate::types::{Candidate, Result, ScannerError};

const DEFAULT_DRIVER: &str = "driver=sdrplay";

struct ScanningConfig {
    duration: u64,
    center_freq: f64,
    samp_rate: f64,
    driver: String,
    exit_early: bool,
}

fn spawn_audio_thread(
    audio_rx: std::sync::mpsc::Receiver<f32>,
) -> Result<thread::JoinHandle<Result<()>>> {
    let handle = thread::spawn(move || -> Result<()> {
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

        let stream = match sample_format {
            SampleFormat::F32 => audio_device.build_output_stream(
                &config,
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
            )?,
            _ => return Err(ScannerError::Custom("Unsupported audio format".to_string())),
        };

        stream.play()?;

        // Keep audio thread alive
        println!("Audio system ready");
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
        println!("Scanning for transmissions ...");

        for candidate in candidate_rx {
            // All candidates write to the same shared audio channel
            fm::analyze_channel(
                candidate,
                config.duration,
                config.center_freq,
                config.samp_rate,
                &config.driver,
                shared_audio_tx.clone(),
            )?;
            if config.exit_early {
                println!("Early exit requested - stopping after first candidate.");
                break;
            }
        }

        println!("All frequencies scanned.");
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

    /// Center frequency to tune to in Hz (e.g., 88.9 MHz)
    #[arg(long, default_value_t = 88.9e6)]
    center_freq: f64,

    /// Duration (for testing)
    #[arg(long, default_value_t = 3)]
    duration: u64,

    /// Duration for peak detection scan (seconds)
    #[arg(long, default_value_t = 1)]
    peak_scan_duration: u64,

    /// Exit after analyzing the first candidate station
    #[arg(long)]
    exit_early: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let driver = args.device_args.unwrap_or(DEFAULT_DRIVER.into());

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

    // Create single shared audio channel - all candidates write to this
    let (audio_tx, audio_rx) = std::sync::mpsc::sync_channel::<f32>(48000); // 1 second buffer

    // Create MPSC channel for candidate stations
    let (candidate_tx, candidate_rx) = std::sync::mpsc::sync_channel::<Candidate>(100);

    // Collect peaks synchronously before starting rustradio graph
    let samp_rate = 1_000_000.0f64;
    let fft_size = 1024;
    let threshold = 1.0;

    let peaks = fm::collect_peaks(
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
        fm::find_candidates(&peaks, args.center_freq, samp_rate, &candidate_tx.clone());
        println!();
    }

    // Set up audio playback thread with simple blocking receive
    let audio_handle = spawn_audio_thread(audio_rx)?;

    let scanning_thread = spawn_scanning_thread(
        candidate_rx,
        ScanningConfig {
            duration: args.duration,
            center_freq: args.center_freq,
            samp_rate,
            driver,
            exit_early: args.exit_early,
        },
        audio_tx.clone(),
        audio_handle,
    )?;

    // Wait for scanning to complete
    drop(candidate_tx);
    let _ = scanning_thread.join();

    println!("Scanner finished.");
    Ok(())
}
