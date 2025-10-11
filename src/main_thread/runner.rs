use super::MainThread;
use super::state_manager;
use crate::core::types::{Result, ScannerError};
use crate::hardware::pool::TunerProvider;
use crate::scanning::window::Window;
use crate::signal;
use std::sync::Arc;
use tracing::debug;

impl MainThread {
    pub(super) fn parse_stations(&self, stations_str: &str) -> Result<Vec<f64>> {
        stations_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
            .collect()
    }

    pub(super) fn scan_stations(&self, stations_str: &str) -> Result<()> {
        let stations = self.parse_stations(stations_str)?;
        debug!(
            message = "Scanning stations",
            stations = format!("{:?}", stations)
        );
        let _total_stations = stations.len();

        // Create a separate window for each station, using the station frequency as center frequency
        for (station_idx, station_freq) in stations.into_iter().enumerate() {
            debug!(
                "Processing station {} of {} at {:.1} MHz",
                station_idx + 1,
                _total_stations,
                station_freq / 1e6
            );

            // Create a window for this specific station frequency (pool-based)
            let window = Window::for_station(
                station_freq,
                station_idx + 1,
                _total_stations,
                Arc::clone(&self.pool) as Arc<dyn TunerProvider>,
                self.config.clone(),
                self.progress_reporter.clone(),
                self.shutdown_coordinator.clone(),
            );

            // Process using pool-based flow
            window.process_with_pool()?;
        }

        Ok(())
    }

    pub(super) fn scan_band(&mut self) -> Result<()> {
        signal::clear_processed_frequencies();

        let window_centers = self.config.band.windows(
            self.config.samp_rate,
            self.config.signal_processing.window_overlap,
        );
        debug!(
            "Scanning {} windows across {:?} band",
            window_centers.len(),
            self.config.band
        );

        let windows_to_process = match self.config.scanning_windows {
            Some(n) => n.min(window_centers.len()),
            None => window_centers.len(),
        };

        let mut context = state_manager::ScanContext::new(self, window_centers, windows_to_process);

        loop {
            let control = context.determine_next_action()?;

            match control {
                state_manager::LoopControl::Break => break,
                state_manager::LoopControl::Continue => continue,
                state_manager::LoopControl::Advance => context.advance(),
            }
        }

        Ok(())
    }
}
