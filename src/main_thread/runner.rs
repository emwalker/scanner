use super::MainThread;
use super::state_manager;
use crate::audio::session::AudioSession;
use crate::core::types::{Result, ScannerError};
use crate::scanning::window::Window;
use crate::signal;
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
                self.pool.clone(),
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

        let mut i: usize = 0;
        let mut audio_session: Option<AudioSession> = None;

        loop {
            if self.shutdown_coordinator.is_shutdown() {
                self.scanner_state.shutdown();
            }

            let control = match &self.scanner_state.mode {
                crate::scanner_state::ScanMode::ShuttingDown => {
                    debug!("Shutdown requested, stopping band scanning");
                    state_manager::LoopControl::Break
                }
                crate::scanner_state::ScanMode::ScanComplete { .. } => {
                    self.check_and_handle_command(windows_to_process, &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::ScanCompletePaused { .. } => {
                    self.process_commands(
                        windows_to_process,
                        window_centers.len(),
                        &mut audio_session,
                    )?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Paused { .. } => {
                    if !i.is_multiple_of(50) {
                        debug!(
                            iteration = i,
                            total = windows_to_process,
                            "Paused - waiting for commands"
                        );
                    }
                    self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Listening { .. } => {
                    self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    state_manager::LoopControl::Continue
                }
                crate::scanner_state::ScanMode::Scanning => {
                    if i >= windows_to_process {
                        debug!("Scan band complete - all windows processed");
                        self.scanner_state.mark_scan_complete(windows_to_process);
                        state_manager::LoopControl::Continue
                    } else {
                        debug!(
                            iteration = i,
                            total = windows_to_process,
                            "Start of scan loop iteration"
                        );

                        if self.process_commands_with_pause_check(
                            i + 1,
                            window_centers.len(),
                            &mut audio_session,
                        )? {
                            state_manager::LoopControl::Continue
                        } else {
                            let center_freq = window_centers[i];
                            self.process_window(i + 1, center_freq, window_centers.len())?;
                            self.process_commands(i + 1, window_centers.len(), &mut audio_session)?;

                            debug!(
                                completed_window = i + 1,
                                next_window = i + 2,
                                remaining = windows_to_process - i - 1,
                                "Window complete, advancing to next"
                            );

                            state_manager::LoopControl::Advance
                        }
                    }
                }
            };

            match control {
                state_manager::LoopControl::Break => break,
                state_manager::LoopControl::Continue => continue,
                state_manager::LoopControl::Advance => i += 1,
            }
        }

        Ok(())
    }
}
