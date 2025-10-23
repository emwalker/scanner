//! Component storing inputs needed to spawn signal analysis thread

use std::sync::Arc;

use crate::{
    core::config::ScanningConfig, ecs::components::window::WindowId, pause_signal::PauseSignal,
};

/// Component holding all inputs needed to spawn an analysis thread
///
/// This component is consumed when the analysis thread is spawned,
/// transitioning the entity from NotStarted to InProgress state.
///
/// Requires two receivers created at the same buffer position by the caller.
/// Both receivers must be created via resubscribe() at the same moment to avoid
/// race conditions where one receiver lags behind the other.
pub struct AnalysisInputComponent {
    pub sdr_rx_refining: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    pub sdr_rx_detection: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    pub config: Arc<ScanningConfig>,
    pub window_id: WindowId,
    pub center_freq: f64,
    pub pause_signal: Option<PauseSignal>,
}

impl AnalysisInputComponent {
    pub fn new(
        sdr_rx_refining: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        sdr_rx_detection: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        config: Arc<ScanningConfig>,
        window_id: WindowId,
        center_freq: f64,
        pause_signal: Option<PauseSignal>,
    ) -> Self {
        Self {
            sdr_rx_refining,
            sdr_rx_detection,
            config,
            window_id,
            center_freq,
            pause_signal,
        }
    }
}
