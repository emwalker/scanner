use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::{
    core::types::ScanningConfig,
    ecs::{Entities, ScanId, StationEntity},
    hardware::pool::TunerProvider,
    pause_signal::PauseSignal,
    shutdown::ShutdownCoordinator,
};

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
    pub scan_id: ScanId,
}
