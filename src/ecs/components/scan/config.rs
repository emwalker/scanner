//! Scan configuration component

use super::ScanType;

/// Component holding scan configuration parameters
#[derive(Debug, Clone)]
pub struct ScanConfigComponent {
    /// Type of scan (band or stations)
    pub scan_type: ScanType,

    /// Minimum frequency in Hz
    pub freq_min: f64,

    /// Maximum frequency in Hz
    pub freq_max: f64,

    /// Window size in Hz
    pub window_size: f64,

    /// Sample rate in Hz
    pub sample_rate: f64,

    /// Gain in dB
    pub gain_db: f64,

    /// Duration per window in seconds
    pub duration_per_window: f64,

    /// Number of scanning windows (parallel processing)
    pub scanning_windows: usize,

    /// Specific stations to scan (for ScanType::Stations)
    pub stations: Vec<f64>,
}

impl ScanConfigComponent {
    /// Create a new scan configuration
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scan_type: ScanType,
        freq_min: f64,
        freq_max: f64,
        window_size: f64,
        sample_rate: f64,
        gain_db: f64,
        duration_per_window: f64,
        scanning_windows: usize,
    ) -> Self {
        Self {
            scan_type,
            freq_min,
            freq_max,
            window_size,
            sample_rate,
            gain_db,
            duration_per_window,
            scanning_windows,
            stations: Vec::new(),
        }
    }

    pub fn with_stations(mut self, stations: Vec<f64>) -> Self {
        self.stations = stations;
        self
    }

    /// Calculate total bandwidth being scanned
    pub fn bandwidth(&self) -> f64 {
        self.freq_max - self.freq_min
    }

    /// Calculate total number of windows
    pub fn total_windows(&self) -> usize {
        match self.scan_type {
            ScanType::Stations => self.stations.len().max(1),
            ScanType::Band => ((self.freq_max - self.freq_min) / self.window_size).ceil() as usize,
        }
    }
}
