use clap::{Parser, ValueEnum};
use num_complex::Complex;
use soapysdr::{Device, Direction};
use std::f32::consts::PI;
use std::sync::mpsc;
use std::thread;
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

    /// Pre-defined frequency band to scan
    #[arg(long)]
    band: Option<Band>,

    /// SDR device arguments (e.g., "driver=sdrplay")
    #[arg(long)]
    device_args: Option<String>,
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

    run_scanner(device, start_freq, stop_freq)?;

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

#[allow(clippy::unnecessary_mut_passed)]
fn run_scanner(device: Device, _start_freq: f64, _stop_freq: f64) -> Result<()> {
    let rx_channel = 0;
    let sample_rate = 2.4e6;
    let bandwidth = 2.4e6;
    let test_freq = 88.9e6;

    device.set_sample_rate(Direction::Rx, rx_channel, sample_rate)?;
    device.set_bandwidth(Direction::Rx, rx_channel, bandwidth)?;
    device.set_frequency(Direction::Rx, rx_channel, test_freq, "")?;
    device.set_gain_mode(Direction::Rx, rx_channel, false)?;
    device.set_gain(Direction::Rx, rx_channel, 20.0)?;

    println!("Tuned to {:.3} MHz. Starting stream...", test_freq / 1e6);

    let mut rx_stream = device.rx_stream::<Complex<i16>>(&[rx_channel])?;
    let (tx, rx) = mpsc::channel::<Vec<Complex<i16>>>();

    let reader_handle = thread::spawn(move || {
        rx_stream.activate(None).unwrap();
        loop {
            let mut buffer = vec![Complex::new(0, 0); 8192];
            match rx_stream.read(&mut [&mut buffer], 1_000_000) {
                Ok(len) => {
                    buffer.truncate(len);
                    if tx.send(buffer).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error reading from stream: {}", e);
                    break;
                }
            }
        }
        rx_stream.deactivate(None).ok();
        println!("Reader thread finished.");
    });

    println!("Receiving samples for processing...");
    for chunk in rx.iter().take(10) {
        if !chunk.is_empty() {
            process_chunk(&chunk, sample_rate, test_freq);
        }
    }

    drop(rx);
    println!("Shutting down reader thread...");
    reader_handle.join().expect("Reader thread panicked");

    Ok(())
}

/// A basic, unoptimized Discrete Fourier Transform.
fn dft(samples: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = samples.len();
    let mut output = Vec::with_capacity(n);

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (i, sample) in samples.iter().enumerate() {
            let angle = -2.0 * PI * (k as f32 * i as f32) / n as f32;
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum += sample * twiddle;
        }
        output.push(sum);
    }

    output
}

fn process_chunk(chunk: &[Complex<i16>], sample_rate: f64, center_freq: f64) {
    let fft_size = chunk.len();

    let window: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
        .collect();

    let buffer: Vec<Complex<f32>> = chunk
        .iter()
        .zip(window.iter())
        .map(|(sample, w)| {
            Complex::new(
                (sample.re as f32 / 32768.0) * w,
                (sample.im as f32 / 32768.0) * w,
            )
        })
        .collect();

    // Perform DFT
    let fft_output = dft(&buffer);

    let power_spectrum: Vec<f32> = fft_output.iter().map(|c| c.norm_sqr()).collect();

    let (max_idx, _max_power) = power_spectrum
        .iter()
        .enumerate()
        .max_by_key(|&(_, &power)| power as u32)
        .unwrap_or((0, &0.0));

    let freq_resolution = sample_rate / fft_size as f64;
    let peak_freq_offset = (max_idx as f64 - (fft_size as f64 / 2.0)) * freq_resolution;
    let peak_freq = center_freq + peak_freq_offset;

    println!("Strongest signal found at: {:.3} MHz", peak_freq / 1e6);
}
