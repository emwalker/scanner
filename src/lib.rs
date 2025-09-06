pub mod broadcast;
pub mod file;
pub mod fm;
pub mod freq_xlating_fir;
pub mod logging;
pub mod mpsc;
pub mod soapy;
pub mod testing;
pub mod types;

pub use crate::logging::LogBuffer;
pub use crate::types::{Candidate, Peak, Result, ScannerError};

// Re-export main types for testing
use clap::ValueEnum;

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
            audio_buffer_size: 4096,
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
