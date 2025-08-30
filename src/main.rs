use clap::{Parser, ValueEnum};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use soapysdr::{Device, Direction, RxStream};
use std::cmp::Ordering;
use std::f32::consts::PI;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use thiserror::Error;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device as AudioDevice, SampleFormat, Stream, StreamConfig};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Start frequency in Hz (e.g., 88.1M or 88100000)
    #[arg(long)]
    start_freq: Option<f64>,

    /// Stop frequency in Hz (e.g., 108M or 108000000)
    #[arg(long)]
    stop_freq: Option<f64>,

    /// Pre-defined frequency band to scan
    #[arg(long)]
    band: Option<Band>,

    /// SDR device arguments (e.g., "driver=sdrplay")
    #[arg(long)]
    device_args: Option<String>,

    /// Number of processing loops to run (defaults to 1)
    #[arg(long, default_value_t = 1)]
    loops: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum Band {
    /// FM Broadcast Band (87.5 - 108 MHz)
    FmBroadcast,
    /// Weather Band (162.4 - 162.55 MHz)
    Weather,
    /// Air Band (108 - 137 MHz)
    Air,
}

impl Band {
    fn frequencies(&self) -> (f64, f64) {
        match self {
            Band::FmBroadcast => (87.5e6, 108.0e6),
            Band::Weather => (162.4e6, 162.55e6),
            Band::Air => (108.0e6, 137.0e6),
        }
    }
}

#[derive(Error, Debug)]
enum ScannerError {
    #[error("Invalid frequency arguments: must specify start/stop frequencies or a single band")]
    InvalidFrequencyArgs,
    #[error(transparent)]
    Sdr(#[from] soapysdr::Error),
    #[error("No SDR devices found matching args: {0}")]
    NoDevicesFound(String),
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
}

type Result<T> = std::result::Result<T, ScannerError>;

fn main() -> Result<()> {
    let args = Args::parse();

    let (device, device_label) = open_device(&args.device_args)?;
    println!("Opened device: {}", device_label);

    let (start_freq, stop_freq) = determine_frequency_range(&args, &device)?;

    println!("Scanner starting...");
    println!(
        "Scanning from {:.3} MHz to {:.3} MHz",
        start_freq / 1e6,
        stop_freq / 1e6
    );

    let (mut scanner_app, audio_rx) = MainLoop::new(device, start_freq, stop_freq, args.loops)?;
    scanner_app.run(audio_rx)?; // Pass audio_rx to run method

    println!("Scanner finished.");
    Ok(())
}

fn open_device(args: &Option<String>) -> Result<(Device, String)> {
    let device_args = args.as_deref().unwrap_or("driver=sdrplay");
    let devices = soapysdr::enumerate(device_args)?;

    if devices.is_empty() {
        return Err(ScannerError::NoDevicesFound(device_args.to_string()));
    }

    let device_label = devices[0]
        .get("label")
        .unwrap_or("Unnamed device")
        .to_string();

    let device = Device::new(device_args)?;
    Ok((device, device_label))
}

fn determine_frequency_range(args: &Args, device: &Device) -> Result<(f64, f64)> {
    match (args.start_freq, args.stop_freq, args.band) {
        (Some(start), Some(stop), None) => {
            if start >= stop {
                return Err(ScannerError::InvalidFrequencyArgs);
            }
            Ok((start, stop))
        }
        (None, None, Some(band)) => Ok(band.frequencies()),
        (None, None, None) => {
            let ranges = device.frequency_range(Direction::Rx, 0)?;
            if let Some(range) = ranges.first() {
                println!(
                    "Device reports frequency range: {:.3} MHz to {:.3} MHz",
                    range.minimum / 1e6,
                    range.maximum / 1e6
                );
                Ok((range.minimum, range.maximum))
            } else {
                Err(ScannerError::Custom(
                    "Device does not report a frequency range".to_string(),
                ))
            }
        }
        _ => Err(ScannerError::InvalidFrequencyArgs),
    }
}

struct SdrManager {
    device: Device,
    rx_channel: usize,
    sample_rate: f64,
    bandwidth: f64,
    tuned_freq: f64,
}

impl SdrManager {
    fn new(
        device: Device,
        rx_channel: usize,
        sample_rate: f64,
        bandwidth: f64,
        tuned_freq: f64,
    ) -> Self {
        SdrManager {
            device,
            rx_channel,
            sample_rate,
            bandwidth,
            tuned_freq,
        }
    }

    fn configure_device(&self) -> Result<()> {
        self.device
            .set_sample_rate(Direction::Rx, self.rx_channel, self.sample_rate)?;
        println!(
            "Configured Sample Rate: {}",
            self.device.sample_rate(Direction::Rx, self.rx_channel)?
        );
        self.device
            .set_bandwidth(Direction::Rx, self.rx_channel, self.bandwidth)?;
        println!(
            "Configured Bandwidth: {}",
            self.device.bandwidth(Direction::Rx, self.rx_channel)?
        );
        self.device
            .set_frequency(Direction::Rx, self.rx_channel, self.tuned_freq, "")?;
        println!(
            "Configured Frequency: {}",
            self.device.frequency(Direction::Rx, self.rx_channel)?
        );
        self.device
            .set_gain_mode(Direction::Rx, self.rx_channel, true)?;
        println!(
            "Configured Gain Mode: {}",
            self.device.gain_mode(Direction::Rx, self.rx_channel)?
        );
        self.device.set_gain(Direction::Rx, self.rx_channel, 40.0)?;
        println!(
            "Configured Gain: {}",
            self.device.gain(Direction::Rx, self.rx_channel)?
        );

        Ok(())
    }

    fn start_stream(&self) -> Result<RxStream<Complex<f32>>> {
        let rx_stream = self.device.rx_stream::<Complex<f32>>(&[self.rx_channel])?;
        Ok(rx_stream)
    }
}

struct DspProcessor {
    sample_rate: f64,
    center_freq: f64,
    decimation_factor: usize,
    fft_planner: Arc<Mutex<FftPlanner<f32>>>,
}

impl DspProcessor {
    fn new(sample_rate: f64, center_freq: f64, decimation_factor: usize) -> Self {
        let planner = FftPlanner::new();

        DspProcessor {
            sample_rate,
            center_freq,
            decimation_factor,
            fft_planner: Arc::new(Mutex::new(planner)),
        }
    }

    fn process_and_demodulate_chunk(&self, chunk: &[Complex<f32>], audio_tx: mpsc::Sender<f32>) {
        let fft_size = chunk.len();

        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
            .collect();

        let mut buffer: Vec<Complex<f32>> = chunk
            .iter()
            .zip(window.iter())
            .map(|(sample, w)| Complex::new(sample.re * w, sample.im * w))
            .collect();

        // Perform FFT (for signal detection)
        let fft = self.fft_planner.lock().unwrap().plan_fft_forward(fft_size);
        fft.process(&mut buffer);
        let fft_output = &buffer;

        let power_spectrum: Vec<f32> = fft_output.iter().map(|c| c.norm_sqr()).collect();

        let (max_idx, _max_power) = power_spectrum
            .iter()
            .enumerate()
            .max_by(|&(_, &a), &(_, &b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
            .unwrap_or((0, &0.0));

        let freq_resolution = self.sample_rate / fft_size as f64;
        let peak_freq_offset = if max_idx < fft_size / 2 {
            max_idx as f64 * freq_resolution
        } else {
            (max_idx as f64 - fft_size as f64) * freq_resolution
        };
        let peak_freq = self.center_freq + peak_freq_offset;
        // Log the actual frequency of the strongest signal for debugging/information.
        println!("Strongest signal detected at: {:.3} MHz", peak_freq / 1e6);

        // --- WBFM Demodulation ---
        let mut prev_phase = 0.0f32;
        let mut current_freq_offset = 0.0f32; // For frequency shifting

        for (i, sample) in buffer.iter().enumerate() {
            // Frequency shift to baseband (for the strongest signal found)
            // This is a simplified shift, a proper one would track the actual signal center
            let phase_increment = -2.0 * PI * (peak_freq_offset as f32 / self.sample_rate as f32);
            current_freq_offset += phase_increment;
            let shifter = Complex::new(current_freq_offset.cos(), current_freq_offset.sin());
            let shifted_sample = sample * shifter;

            // WBFM Demodulation (phase differentiation)
            let phase = shifted_sample.arg(); // atan2(im, re)
            let mut diff = phase - prev_phase;
            prev_phase = phase;

            // Handle phase wrapping
            if diff > PI {
                diff -= 2.0 * PI;
            }
            if diff < -PI {
                diff += 2.0 * PI;
            }

            // Decimate and send to audio
            if i % self.decimation_factor == 0 {
                // Scale for audio output (arbitrary scaling for now)
                let audio_sample = diff * 50.0;
                if audio_tx.send(audio_sample).is_err() {
                    // Audio thread disconnected
                    break;
                }
            }
        }
    }
}

#[allow(dead_code)]
struct AudioPlayer {
    host: cpal::Host,
    device: AudioDevice,
    config: StreamConfig,
    sample_format: SampleFormat,
    audio_rx: Arc<Mutex<mpsc::Receiver<f32>>>,
    stream: Option<Stream>,
}

impl AudioPlayer {
    fn new(audio_rx: mpsc::Receiver<f32>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");
        let supported_config = device
            .default_output_config()
            .expect("Failed to get default output config");
        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();

        Ok(AudioPlayer {
            host,
            device,
            config,
            sample_format,
            audio_rx: Arc::new(Mutex::new(audio_rx)),
            stream: None,
        })
    }

    fn build_and_play_stream(&mut self) -> Result<()> {
        // Change signature
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
        let audio_rx_moved = self.audio_rx.clone(); // Clone the Arc

        let stream = match self.sample_format {
            SampleFormat::F32 => self.device.build_output_stream(
                &self.config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    for sample in data.iter_mut() {
                        *sample = match audio_rx_moved.lock().unwrap().try_recv() {
                            // Lock and receive
                            Ok(s) => s,
                            Err(mpsc::TryRecvError::Empty) => 0.0f32,
                            Err(mpsc::TryRecvError::Disconnected) => 0.0f32,
                        };
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::I16 => self.device.build_output_stream(
                &self.config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    for sample in data.iter_mut() {
                        *sample = match audio_rx_moved.lock().unwrap().try_recv() {
                            // Lock and receive
                            Ok(s) => cpal::Sample::from_sample(s),
                            Err(mpsc::TryRecvError::Empty) => cpal::Sample::from_sample(0.0f32),
                            Err(mpsc::TryRecvError::Disconnected) => {
                                cpal::Sample::from_sample(0.0f32)
                            }
                        };
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::U16 => self.device.build_output_stream(
                &self.config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    for sample in data.iter_mut() {
                        *sample = match audio_rx_moved.lock().unwrap().try_recv() {
                            // Lock and receive
                            Ok(s) => cpal::Sample::from_sample(s),
                            Err(mpsc::TryRecvError::Empty) => cpal::Sample::from_sample(0.0f32),
                            Err(mpsc::TryRecvError::Disconnected) => {
                                cpal::Sample::from_sample(0.0f32)
                            }
                        };
                    }
                },
                err_fn,
                None,
            )?,
            _ => {
                return Err(ScannerError::Custom(
                    "Unsupported sample format".to_string(),
                ));
            }
        };

        stream.play()?;
        self.stream = Some(stream);
        Ok(()) // Return Ok(())
    }
}

struct MainLoop {
    sdr_manager: SdrManager,
    dsp_processor: DspProcessor,
    iq_tx: mpsc::Sender<Vec<Complex<f32>>>,
    iq_rx: mpsc::Receiver<Vec<Complex<f32>>>,
    audio_tx_main: mpsc::Sender<f32>,
    stop_tx: mpsc::Sender<()>, // This can remain a Sender
    stop_rx: Arc<Mutex<mpsc::Receiver<()>>>,
    loops: usize,
}

impl MainLoop {
    fn new(
        device: Device,
        _start_freq: f64,
        _stop_freq: f64,
        loops: usize,
    ) -> Result<(Self, mpsc::Receiver<f32>)> {
        let rx_channel = 0;
        let sample_rate = 5.0e6; // 5.0 Msps
        let bandwidth = 5.0e6;
        let test_freq = 88.9e6; // A test frequency in the FM band
        let audio_sample_rate = 48_000.0; // 48 kHz
        let decimation_factor = (sample_rate / audio_sample_rate) as usize;

        let sdr_manager = SdrManager::new(device, rx_channel, sample_rate, bandwidth, test_freq);
        let dsp_processor = DspProcessor::new(sample_rate, test_freq, decimation_factor);

        let (iq_tx, iq_rx) = mpsc::channel::<Vec<Complex<f32>>>();
        let (audio_tx_main, audio_rx) = mpsc::channel::<f32>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        Ok((
            MainLoop {
                sdr_manager,
                dsp_processor,
                iq_tx,
                iq_rx,
                audio_tx_main,
                stop_tx,
                stop_rx: Arc::new(Mutex::new(stop_rx)),
                loops,
            },
            audio_rx,
        ))
    }

    #[allow(clippy::unnecessary_mut_passed)]
    fn run(&mut self, audio_rx: mpsc::Receiver<f32>) -> Result<()> {
        self.sdr_manager.configure_device()?;

        println!(
            "Tuned to {:.3} MHz. Starting stream...",
            self.sdr_manager.tuned_freq / 1e6
        );

        let mut rx_stream = self.sdr_manager.start_stream()?;

        // Spawn a dedicated I/O thread for reading from SDR
        let iq_tx_reader = self.iq_tx.clone(); // Clone for reader thread
        let stop_rx_reader = self.stop_rx.clone(); // Clone the Arc
        let reader_handle = thread::spawn(move || {
            rx_stream.activate(None).unwrap();
            loop {
                // Check for stop signal
                if stop_rx_reader.lock().unwrap().try_recv().is_ok() {
                    // Lock and receive
                    break;
                }

                let mut buffer = vec![Complex::new(0.0, 0.0); 8192];
                match rx_stream.read(&mut [&mut buffer], 1_000_000) {
                    Ok(len) => {
                        buffer.truncate(len);

                        if iq_tx_reader.send(buffer).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Reader thread: Error reading from stream: {}. Breaking loop.",
                            e
                        );
                        break;
                    }
                }
            }
            rx_stream.deactivate(None).ok();
            println!("Reader thread finished.");
        });

        // Spawn an audio playback thread
        let _audio_handle = thread::spawn(move || {
            let mut audio_player = AudioPlayer::new(audio_rx)?; // Create AudioPlayer inside the thread
            audio_player.build_and_play_stream()?; // Build and play the stream

            // The audio thread will park itself indefinitely to keep the stream alive.
            // It will be unparked or terminated when the main process exits.
            thread::park();
            Ok::<(), ScannerError>(())
        });

        // Main thread acts as the DSP processor
        println!("Receiving samples for processing...");
        for chunk in self.iq_rx.iter().take(self.loops) {
            if !chunk.is_empty() {
                self.dsp_processor
                    .process_and_demodulate_chunk(&chunk, self.audio_tx_main.clone());
            }
        }

        // Signal reader thread to stop explicitly
        self.stop_tx.send(()).ok();

        // The audio thread will eventually exit when its receiver disconnects.
        // For now, we don't explicitly join it here as it might block indefinitely
        // if the audio stream is still active and waiting for data.
        // In a real app, you'd have a more robust shutdown for the audio thread.
        reader_handle.join().expect("Reader thread panicked");

        Ok(())
    }
}
