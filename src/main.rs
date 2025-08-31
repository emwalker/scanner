use clap::{Parser, ValueEnum};
use rustfft::num_complex::Complex;
use soapysdr::{Device, Direction, RxStream};
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
    #[arg(long, default_value_t = 1000)]
    loops: usize,

    /// Specific frequency to tune to in Hz (e.g., 98.1M or 98100000)
    #[arg(long, default_value_t = 88.9e6)]
    tune_freq: f64,
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

    let (mut scanner_app, audio_rx) = MainLoop::new(device, args.tune_freq, args.loops)?;
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
        // self.device.set_gain(Direction::Rx, self.rx_channel, 40.0)?;
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

// FM broadcast constants
const FM_MAX_DEVIATION_HZ: f32 = 75_000.0; // 75 kHz maximum frequency deviation for FM broadcast
const FM_CHANNEL_BANDWIDTH_HZ: f32 = 200_000.0; // 200 kHz FM channel bandwidth
const AUDIO_BANDWIDTH_HZ: f32 = 15_000.0; // 15 kHz audio bandwidth for FM broadcast

struct DspProcessor {
    sample_rate: f64,
    decimation_factor: usize,
    // Pre-demodulation low-pass filter state (IIR filter for RF signal)
    filter_prev_i: f32,
    filter_prev_q: f32,
    filter_alpha: f32, // Pre-demod filter coefficient
    intermediate_decimation: usize,
    intermediate_counter: usize,
    // FM demodulation state
    prev_phase: f32,
    decimation_counter: usize,
    // Post-demodulation audio low-pass filter state
    audio_filter_prev: f32,
    audio_filter_alpha: f32, // Audio filter coefficient
    // Debugging counters
    total_samples: usize,
    audio_samples_sent: usize,
    // FM demodulation debugging
    min_signal_strength: f32,
    max_signal_strength: f32,
    min_phase_diff: f32,
    max_phase_diff: f32,
    min_freq_deviation: f32,
    max_freq_deviation: f32,
    samples_at_full_scale: usize,
    k_fm: f32, // Scaling factor for FM demodulation
    debug_print_interval: usize,
    debug_counter: usize,
}

impl DspProcessor {
    fn new(sample_rate: f64, decimation_factor: usize) -> Self {
        // Calculate pre-demodulation low-pass filter coefficient for FM channel bandwidth
        // Simple IIR: alpha = 2*pi*fc / (2*pi*fc + fs)
        let cutoff_freq = FM_CHANNEL_BANDWIDTH_HZ / 2.0; // ~100 kHz cutoff
        let filter_alpha = 2.0 * PI * cutoff_freq / (2.0 * PI * cutoff_freq + sample_rate as f32);

        // Intermediate decimation: reduce from 4.8M to ~480k (10x decimation)
        // This gives us 480k sample rate, plenty for 200kHz FM bandwidth
        let intermediate_decimation = 1; // No intermediate decimation, process at full sample rate

        // Calculate post-demodulation audio filter coefficient
        // Use the final audio sample rate (48kHz) for the audio filter
        let audio_sample_rate = sample_rate / decimation_factor as f64;
        let audio_filter_alpha = 2.0 * PI * AUDIO_BANDWIDTH_HZ
            / (2.0 * PI * AUDIO_BANDWIDTH_HZ + audio_sample_rate as f32);

        let k_fm = sample_rate as f32 / (2.0 * PI * FM_MAX_DEVIATION_HZ);

        DspProcessor {
            sample_rate,
            decimation_factor,
            filter_prev_i: 0.0,
            filter_prev_q: 0.0,
            filter_alpha,
            intermediate_decimation,
            intermediate_counter: 0,
            prev_phase: 0.0,
            decimation_counter: 0,
            audio_filter_prev: 0.0,
            audio_filter_alpha,
            total_samples: 0,
            audio_samples_sent: 0,
            min_signal_strength: f32::INFINITY,
            max_signal_strength: 0.0,
            min_phase_diff: f32::INFINITY,
            max_phase_diff: f32::NEG_INFINITY,
            min_freq_deviation: f32::INFINITY,
            max_freq_deviation: f32::NEG_INFINITY,
            samples_at_full_scale: 0,
            k_fm,
            debug_print_interval: 1000,
            debug_counter: 0,
        }
    }

    fn process_and_demodulate_chunk(
        &mut self,
        chunk: &[Complex<f32>],
        audio_tx: std::sync::mpsc::SyncSender<f32>,
    ) {
        for sample in chunk {
            self.total_samples += 1;

            // 1. Pre-demodulation Low-Pass Filter (IIR)
            // y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            let filtered_i =
                self.filter_alpha * sample.re + (1.0 - self.filter_alpha) * self.filter_prev_i;
            let filtered_q =
                self.filter_alpha * sample.im + (1.0 - self.filter_alpha) * self.filter_prev_q;
            self.filter_prev_i = filtered_i;
            self.filter_prev_q = filtered_q;

            let filtered_sample = Complex::new(filtered_i, filtered_q);

            // Update signal strength stats
            let signal_strength = filtered_sample.norm();
            self.min_signal_strength = self.min_signal_strength.min(signal_strength);
            self.max_signal_strength = self.max_signal_strength.max(signal_strength);

            // 2. Intermediate Decimation
            self.intermediate_counter += 1;
            if self.intermediate_counter % self.intermediate_decimation != 0 {
                continue; // Skip this sample if not time to decimate
            }
            self.intermediate_counter = 0; // Reset counter

            // 3. FM Quadrature Demodulation
            // Calculate instantaneous phase
            let current_phase = filtered_sample.arg(); // atan2(im, re)

            // Calculate phase difference
            let mut phase_diff = current_phase - self.prev_phase;

            // Handle phase wrapping (unwrap phase)
            if phase_diff > PI {
                phase_diff -= 2.0 * PI;
            } else if phase_diff < -PI {
                phase_diff += 2.0 * PI;
            }

            self.prev_phase = current_phase;

            // Update phase diff stats
            self.min_phase_diff = self.min_phase_diff.min(phase_diff);
            self.max_phase_diff = self.max_phase_diff.max(phase_diff);

            // Scale phase difference to get frequency deviation
            let demodulated_sample = phase_diff * self.k_fm;

            // Update frequency deviation stats
            self.min_freq_deviation = self.min_freq_deviation.min(demodulated_sample);
            self.max_freq_deviation = self.max_freq_deviation.max(demodulated_sample);

            // Check for samples at full scale (clipping)
            if demodulated_sample.abs() >= 0.99 {
                self.samples_at_full_scale += 1;
            }

            // 4. Post-demodulation Audio Low-Pass Filter (IIR)
            // y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            let audio_filtered_sample = self.audio_filter_alpha * demodulated_sample
                + (1.0 - self.audio_filter_alpha) * self.audio_filter_prev;
            self.audio_filter_prev = audio_filtered_sample;

            // 5. Overall Decimation and Send to audio_tx
            self.decimation_counter += 1;
            if self.decimation_counter % self.decimation_factor != 0 {
                continue; // Skip this sample if not time to decimate
            }
            self.decimation_counter = 0; // Reset counter

            // Send the final audio sample
            if audio_tx.try_send(audio_filtered_sample).is_ok() {
                self.audio_samples_sent += 1;
            } else {
                // Handle error if audio_tx is full or disconnected
                // For now, just ignore, but in a real app, you might want to log or handle
            }
        }

        // Debugging output
        self.debug_counter += 1;
        if self.debug_counter >= self.debug_print_interval {
            println!("\n--- DSP Debug Info ---");
            println!("Total samples processed: {}", self.total_samples);
            println!("Audio samples sent: {}", self.audio_samples_sent);
            println!(
                "Signal Strength: Min={:.4}, Max={:.4}",
                self.min_signal_strength, self.max_signal_strength
            );
            println!(
                "Phase Diff: Min={:.4}, Max={:.4}",
                self.min_phase_diff, self.max_phase_diff
            );
            println!(
                "Freq Deviation: Min={:.4}, Max={:.4}",
                self.min_freq_deviation, self.max_freq_deviation
            );
            println!("Samples at full scale: {}", self.samples_at_full_scale);
            println!("----------------------");

            // Reset for next interval
            self.total_samples = 0;
            self.audio_samples_sent = 0;
            self.min_signal_strength = f32::INFINITY;
            self.max_signal_strength = 0.0;
            self.min_phase_diff = f32::INFINITY;
            self.max_phase_diff = f32::NEG_INFINITY;
            self.min_freq_deviation = f32::INFINITY;
            self.max_freq_deviation = f32::NEG_INFINITY;
            self.samples_at_full_scale = 0;
            self.debug_counter = 0;
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
        let mut config: StreamConfig = supported_config.into();
        // Reduce buffer size to decrease callback pressure
        config.buffer_size = cpal::BufferSize::Fixed(4096); // Increased buffer size to reduce underruns

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
                    let mut samples_received = 0;
                    let mut samples_empty = 0;

                    // Try to get as many samples as possible
                    for sample in data.iter_mut() {
                        *sample = match audio_rx_moved.lock().unwrap().try_recv() {
                            Ok(s) => {
                                samples_received += 1;
                                s
                            }
                            Err(mpsc::TryRecvError::Empty) => {
                                samples_empty += 1;
                                0.0f32
                            }
                            Err(mpsc::TryRecvError::Disconnected) => 0.0f32,
                        };
                    }

                    // Debug audio callback underruns when they occur
                    if samples_empty > data.len() / 2 {
                        // More than 50% empty samples
                        println!(
                            "Audio underrun: requested {}, got {}, empty {}",
                            data.len(),
                            samples_received,
                            samples_empty
                        );
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
    audio_tx_main: std::sync::mpsc::SyncSender<f32>,
    stop_tx: mpsc::Sender<()>, // This can remain a Sender
    stop_rx: Arc<Mutex<mpsc::Receiver<()>>>,
    loops: usize,
}

impl MainLoop {
    fn new(device: Device, tuned_freq: f64, loops: usize) -> Result<(Self, mpsc::Receiver<f32>)> {
        let rx_channel = 0;
        // Use 4.8 MHz for perfect integer decimation to 48 kHz audio (4.8M / 48k = 100)
        let sample_rate = 960e3; // 4.8 Msps - divides evenly into 48 kHz
        let bandwidth = 960e3;

        // Configure audio system for a specific sample rate instead of using default (likely 44.1kHz)
        let host = cpal::default_host();
        let audio_device = host
            .default_output_device()
            .expect("no output device available");

        let supported_config = audio_device
            .supported_output_configs()
            .expect("Failed to get supported configs")
            .find(|c| c.channels() == 1 && c.sample_format() == cpal::SampleFormat::F32)
            .expect("No suitable audio config found for 1 channel F32");

        let config: StreamConfig = supported_config
            .with_sample_rate(cpal::SampleRate(48000))
            .into();

        let audio_sample_rate = config.sample_rate.0 as f64;
        println!("cpal reported audio sample rate: {} Hz", audio_sample_rate);
        // Calculate decimation factor to match audio system exactly
        let ideal_decimation = sample_rate / audio_sample_rate;
        let decimation_factor = (ideal_decimation as f64).round() as usize;

        let sdr_manager = SdrManager::new(device, rx_channel, sample_rate, bandwidth, tuned_freq);
        let dsp_processor = DspProcessor::new(sample_rate, decimation_factor);

        println!("Audio system sample rate: {:.1} Hz", audio_sample_rate);
        println!("Ideal decimation: {:.3}", ideal_decimation);
        println!("Actual decimation factor: {}", decimation_factor);
        println!(
            "Actual audio production rate: {:.1} Hz",
            sample_rate / decimation_factor as f64
        );
        println!(
            "Rate error: {:.1}%",
            ((sample_rate / decimation_factor as f64) - audio_sample_rate) / audio_sample_rate
                * 100.0
        );

        let (iq_tx, iq_rx) = mpsc::channel::<Vec<Complex<f32>>>();

        // Create a larger bounded channel for audio to provide buffering (1 second buffer)
        let audio_buffer_size = audio_sample_rate as usize;
        let (audio_tx_main, audio_rx) = std::sync::mpsc::sync_channel::<f32>(audio_buffer_size);
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
