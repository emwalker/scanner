use std::{thread::JoinHandle, time::Instant};

use tokio_util::sync::CancellationToken;

use crate::core::types::Result;

pub struct WindowWorkerComponent {
    pub window_index: usize,
    pub task_handle: JoinHandle<Result<WindowWorkerResult>>,
    pub cancellation_token: CancellationToken,
    pub started_at: Instant,
    pub cancelling: bool,
}

impl std::fmt::Debug for WindowWorkerComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowWorkerComponent")
            .field("window_index", &self.window_index)
            .field("task_handle", &"<JoinHandle>")
            .field("cancellation_token", &self.cancellation_token)
            .field("started_at", &self.started_at)
            .field("cancelling", &self.cancelling)
            .finish()
    }
}

pub enum WindowWorkerOutcome {
    Success {
        signals: Vec<SignalData>,
        segment: std::sync::Arc<crate::hardware::pool::Segment>,
        center_freq: f64,
    },
    NoSignals {
        center_freq: f64,
        reason: String,
    },
}

impl std::fmt::Debug for WindowWorkerOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowWorkerOutcome::Success {
                signals,
                center_freq,
                ..
            } => f
                .debug_struct("Success")
                .field("signals", signals)
                .field("segment", &"<Segment>")
                .field("center_freq", center_freq)
                .finish(),
            WindowWorkerOutcome::NoSignals {
                center_freq,
                reason,
            } => f
                .debug_struct("NoSignals")
                .field("center_freq", center_freq)
                .field("reason", reason)
                .finish(),
        }
    }
}

pub struct WindowWorkerResult {
    pub window_index: usize,
    pub outcome: WindowWorkerOutcome,
    pub completed_at: Instant,
}

impl std::fmt::Debug for WindowWorkerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowWorkerResult")
            .field("window_index", &self.window_index)
            .field("outcome", &self.outcome)
            .field("completed_at", &self.completed_at)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct SignalData {
    pub frequency_hz: f64,
    pub signal_strength: f64,
    pub audio_quality: crate::audio::quality::AudioQuality,
}
