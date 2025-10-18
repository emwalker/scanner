use crate::audio::session::AudioSession;
use crate::core::types::ScanningConfig;
use crate::core::types::{ModulationType, Result, Signal};
use crate::hardware::pool::{Pool, Segment, TunerActivity, TunerState};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::{ProgressEvent, ProgressEventType, ProgressReporter};
use std::sync::Arc;
use tracing::debug;

#[derive(Clone)]
pub struct TuneParams {
    pub candidate_id: String,
    pub window_id: usize,
    pub center_frequency: f64,
    pub candidate_frequency: f64,
    pub signal_strength: Option<f64>,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
}

pub struct AudioCoordinator<'a> {
    pool: &'a Arc<Pool>,
    config: &'a ScanningConfig,
    shutdown_coordinator: &'a Arc<ShutdownCoordinator>,
    progress_reporter: &'a Arc<dyn ProgressReporter>,
}

impl<'a> AudioCoordinator<'a> {
    pub fn new(
        pool: &'a Arc<Pool>,
        config: &'a ScanningConfig,
        shutdown_coordinator: &'a Arc<ShutdownCoordinator>,
        progress_reporter: &'a Arc<dyn ProgressReporter>,
    ) -> Self {
        Self {
            pool,
            config,
            shutdown_coordinator,
            progress_reporter,
        }
    }

    pub fn tune_to_station(
        &self,
        audio_session: &mut AudioSession,
        params: TuneParams,
    ) -> Result<()> {
        debug!(
            candidate_id = ?params.candidate_id,
            candidate_mhz = params.candidate_frequency / 1e6,
            center_mhz = params.center_frequency / 1e6,
            signal_strength = ?params.signal_strength,
            audio_quality = ?params.audio_quality,
            "Tuning to candidate"
        );

        // CRITICAL: Stop current station FIRST to release tuner back to pool
        audio_session.stop_current_station();

        // Create pool-based segment for listening
        let segment = Segment::new(
            self.pool,
            params.center_frequency,
            self.config,
            self.shutdown_coordinator,
        )?;

        // Get tuner ID from pool status (first allocated tuner for listening)
        let status = self.pool.status();
        let tuner_id = status
            .tuners
            .iter()
            .find(|t| {
                t.state == TunerState::Allocated && t.activity == Some(TunerActivity::Listening)
            })
            .map(|t| t.id.device_id.clone());

        let signal = Signal {
            frequency_hz: params.candidate_frequency,
            signal_strength: params.signal_strength.unwrap_or(0.1) as f32,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: self.config.audio.sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms: 0,
            detection_center_freq: params.center_frequency,
            audio_quality: params
                .audio_quality
                .unwrap_or(crate::audio::quality::AudioQuality::Unknown),
        };

        tracing::info!(
            "playing {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );

        audio_session.tune_to_station(&signal, Box::new(segment), self.config)?;

        debug!(
            candidate_id = ?params.candidate_id,
            event_type = "AudioPlaybackStarted",
            "AudioCoordinator: Sending AudioPlaybackStarted event"
        );

        self.progress_reporter.report(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: params.candidate_frequency,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: params.center_frequency,
                window_id: params.window_id,
            },
            candidate_id: Some(params.candidate_id),
            audio_quality: None,
            signal_strength: None,
            timestamp: std::time::Instant::now(),
            tuner_id,
        });

        Ok(())
    }

    pub fn stop_listening(
        &self,
        audio_session: &mut Option<AudioSession>,
        station: Option<&crate::ecs::StationEntity>,
        audio: Option<&crate::ecs::AudioEntity>,
    ) {
        debug!("Stopped listening, returning to browsing mode");

        if let Some(session) = audio_session {
            session.stop_current_station();
        }

        // Send AudioPlaybackCompleted event if we were playing something
        if let (Some(station_entity), Some(audio_entity)) = (station, audio) {
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: station_entity.frequency(),
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: audio_entity.tuning.center_frequency_hz,
                    window_id: station_entity.discovery.window_id,
                },
                candidate_id: None,
                audio_quality: station_entity.info.audio_quality,
                signal_strength: Some(station_entity.signal_strength() as f64),
                timestamp: std::time::Instant::now(),
                tuner_id: audio_entity.tuner_id().cloned(),
            });
        }
    }
}
