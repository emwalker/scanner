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

    /// Step size in Hz (distance between window centers)
    pub step_size: f64,

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
        step_size: f64,
        sample_rate: f64,
        gain_db: f64,
        duration_per_window: f64,
        scanning_windows: usize,
    ) -> Self {
        Self {
            scan_type,
            freq_min,
            freq_max,
            step_size,
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
    ///
    /// For band scans, calculates the number of windows needed to cover the range [freq_min,
    /// freq_max] inclusively. The formula ensures the last window center is <= freq_max.
    ///
    /// Formula: floor((freq_max - freq_min) / step_size) + 1
    ///
    /// Examples:
    /// - Range 88-108 MHz, step 1 MHz: floor(20/1) + 1 = 21 windows (88, 89, ..., 108)
    /// - Range 88-108 MHz, step 0.5 MHz: floor(20/0.5) + 1 = 41 windows (88.0, 88.5, ..., 108.0)
    /// - Range 100-101 MHz, step 1 MHz: floor(1/1) + 1 = 2 windows (100, 101)
    pub fn total_windows(&self) -> usize {
        match self.scan_type {
            ScanType::Stations => self.stations.len().max(1),
            ScanType::Band => {
                // Calculate number of steps from freq_min to freq_max, then add 1 for the starting
                // window
                let steps = ((self.freq_max - self.freq_min) / self.step_size).floor() as usize;
                steps + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that all window center frequencies stay within the configured band range.
    ///
    /// This is a regression test for the bug where scanning continued past 108 MHz.
    /// The issue was that total_windows() calculated based on (max-min)/step but didn't
    /// ensure the last window center stayed <= freq_max.
    ///
    /// Example that should FAIL without the fix:
    /// - FM band: 88 MHz - 108 MHz
    /// - Step size: 1 MHz
    /// - Current formula: (108-88)/1 = 20 windows
    /// - Window centers: 88, 89, ..., 107 (only 20 windows, missing 108)
    /// - But if configured with 40 windows (due to overlap config): Window centers: 88, 89, ...,
    ///   127 (goes way past 108!)
    #[test]
    fn test_all_window_centers_within_freq_max() {
        // FM band: 88-108 MHz, 1 MHz steps
        let config = ScanConfigComponent::new(
            ScanType::Band,
            88.0e6,  // freq_min
            108.0e6, // freq_max
            1.0e6,   // step_size (1 MHz)
            2.0e6,   // sample_rate
            24.0,    // gain_db
            3.0,     // duration_per_window
            2,       // scanning_windows (parallel processing, not related to total count)
        );

        let total_windows = config.total_windows();

        // Calculate all window centers
        for window_index in 0..total_windows {
            let center_freq = config.freq_min + (window_index as f64 * config.step_size);

            assert!(
                center_freq <= config.freq_max,
                "Window {} has center frequency {:.1} MHz which exceeds freq_max {:.1} MHz. \
                 total_windows={}, step_size={:.1} MHz",
                window_index,
                center_freq / 1e6,
                config.freq_max / 1e6,
                total_windows,
                config.step_size / 1e6
            );
        }
    }

    /// Test with fractional step sizes (0.5 MHz, typical with 75% window overlap)
    #[test]
    fn test_window_centers_with_overlap() {
        // FM band with 75% overlap (0.5 MHz steps)
        let config = ScanConfigComponent::new(
            ScanType::Band,
            88.0e6,  // freq_min
            108.0e6, // freq_max
            0.5e6,   // step_size (0.5 MHz with overlap)
            2.0e6,   // sample_rate
            24.0,    // gain_db
            3.0,     // duration_per_window
            2,       // scanning_windows
        );

        let total_windows = config.total_windows();

        // All window centers must be <= freq_max
        for window_index in 0..total_windows {
            let center_freq = config.freq_min + (window_index as f64 * config.step_size);

            assert!(
                center_freq <= config.freq_max,
                "Window {} center {:.1} MHz > freq_max {:.1} MHz",
                window_index,
                center_freq / 1e6,
                config.freq_max / 1e6
            );
        }
    }

    /// Test edge case: step_size equals bandwidth
    #[test]
    fn test_single_window_scan() {
        let config = ScanConfigComponent::new(
            ScanType::Band,
            100.0e6, // freq_min
            101.0e6, // freq_max
            1.0e6,   // step_size (equals bandwidth)
            2.0e6,   // sample_rate
            24.0,    // gain_db
            3.0,     // duration_per_window
            1,       // scanning_windows
        );

        // Should have exactly 2 windows: one at 100 MHz, one at 101 MHz
        assert_eq!(
            config.total_windows(),
            2,
            "Should have 2 windows for inclusive range"
        );

        let center_0 = config.freq_min;
        let center_1 = config.freq_min + config.step_size;

        assert_eq!(center_0, 100.0e6);
        assert_eq!(center_1, 101.0e6);
        assert!(center_1 <= config.freq_max);
    }
}
