use super::MainThread;
use crate::core::types::{Result, ScannerError};
use crate::task::{ScanBandTask, ScanStationsTask, Task};
use std::time::Duration;
use tracing::debug;

impl MainThread {
    pub(super) fn parse_stations(&self, stations_str: &str) -> Result<Vec<f64>> {
        stations_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
            .collect()
    }

    pub(super) fn scan_stations(&mut self, stations_str: &str) -> Result<()> {
        let stations = self.parse_stations(stations_str)?;
        debug!(
            station_count = stations.len(),
            "Creating ScanStationsTask for station scanning"
        );

        let scan_task = ScanStationsTask::new_full(
            (*self.config).clone(),
            stations,
            self.progress_reporter.clone(),
            self.pool.clone(),
            self.shutdown_coordinator.clone(),
            self.command_receiver.take(),
            self.tui_event_sender.clone(),
        )
        .with_scan_entities(self.scan_entities.clone())
        .with_station_entities(self.station_entities.clone())
        .with_audio_entities(self.audio_entities.clone());

        let handle = self.scheduler.submit(Task::ScanStations(scan_task))?;

        while !handle.is_cancelled() && !self.shutdown_coordinator.is_shutdown() {
            std::thread::sleep(Duration::from_millis(100));
        }

        debug!("ScanStationsTask completed");
        Ok(())
    }

    pub(super) fn scan_band(&mut self) -> Result<()> {
        debug!(
            band = ?self.config.band,
            "Creating ScanBandTask for band scanning"
        );

        let scan_task = ScanBandTask::new_full(
            (*self.config).clone(),
            self.config.band,
            self.progress_reporter.clone(),
            self.pool.clone(),
            self.shutdown_coordinator.clone(),
            self.command_receiver.take(),
            self.tui_event_sender.clone(),
        )
        .with_scan_entities(self.scan_entities.clone())
        .with_station_entities(self.station_entities.clone())
        .with_audio_entities(self.audio_entities.clone());

        let handle = self.scheduler.submit(Task::ScanBand(Box::new(scan_task)))?;

        while !handle.is_cancelled() && !self.shutdown_coordinator.is_shutdown() {
            std::thread::sleep(Duration::from_millis(100));
        }

        debug!("ScanBandTask completed");
        Ok(())
    }
}
