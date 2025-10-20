use crate::core::types::ScanningConfig;
use crate::ecs::{CandidateEntity, Entities, ScanId, StationEntity};
use crate::hardware::pool::TunerProvider;
use crate::pause_signal::PauseSignal;
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy)]
pub struct WindowMetadata {
    pub center_frequency_hz: f64,
    pub window_id: usize,
}

pub struct WindowConfig {
    pub center_freq: f64,
    pub window_num: usize,
    pub total_windows: usize,
    pub tuner_provider: Arc<dyn TunerProvider>,
    pub config: Arc<ScanningConfig>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub window_cancellation: Option<CancellationToken>,
    pub pause_signal: Option<PauseSignal>,
    pub station_entities: Option<Entities<StationEntity>>,
    pub candidate_entities: Option<Entities<CandidateEntity>>,
    pub scan_id: ScanId,
}
