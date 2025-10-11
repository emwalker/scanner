use crate::core::types::ScanningConfig;
use crate::hardware::pool::TunerProvider;
use crate::scanner_state::PauseSignal;
use crate::shutdown::ShutdownCoordinator;
use crate::ui::ProgressReporter;
use std::sync::Arc;

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
    pub progress_reporter: Arc<dyn ProgressReporter>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pause_signal: Option<PauseSignal>,
}
