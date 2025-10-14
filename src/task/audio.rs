//! Audio streaming task

use crate::audio::quality::AudioQuality;
use crate::audio::session::AudioSession;
use crate::core::types::{ModulationType, Result, ScannerError, ScanningConfig, Signal};
use crate::hardware::pool::{Pool, Segment};
use crate::hardware::types::Backend;
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio_util::sync::CancellationToken;

/// Audio streaming task (tuner-holder pattern)
#[allow(dead_code)]
pub struct AudioTask {
    station_freq: f64,
    config: ScanningConfig,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

impl AudioTask {
    #[allow(dead_code)]
    pub fn new(
        station_freq: f64,
        config: ScanningConfig,
        pool: Arc<Pool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            station_freq,
            config,
            pool,
            shutdown_coordinator,
        }
    }

    #[allow(dead_code)]
    pub fn backend(&self) -> Backend {
        Backend::Soapy
    }

    #[allow(dead_code)]
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<()> {
        if shutdown.is_cancelled() {
            return Ok(());
        }

        let mut audio_session = AudioSession::new(&self.config, self.shutdown_coordinator.clone())?;

        if shutdown.is_cancelled() {
            return Ok(());
        }

        let segment = Segment::new(
            &self.pool,
            self.station_freq,
            &self.config,
            &self.shutdown_coordinator,
        )?;

        let signal = Signal {
            frequency_hz: self.station_freq,
            signal_strength: 0.5,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: self.config.audio.sample_rate,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 0,
            detection_center_freq: self.station_freq,
            audio_quality: AudioQuality::Unknown,
        };

        audio_session.tune_to_station(&signal, Box::new(segment), &self.config)?;

        while !shutdown.is_cancelled() {
            std::thread::sleep(Duration::from_millis(100));
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn description(&self) -> String {
        format!("Audio: {:.1} MHz FM", self.station_freq / 1e6)
    }

    #[allow(dead_code)]
    pub fn on_start(&mut self) {
        // TODO: Notify progress_reporter
    }

    #[allow(dead_code)]
    pub fn on_complete(&mut self) {
        // TODO: Notify progress_reporter
    }

    #[allow(dead_code)]
    pub fn on_error(&mut self, _error: &ScannerError) {
        // TODO: Report error
    }
}
