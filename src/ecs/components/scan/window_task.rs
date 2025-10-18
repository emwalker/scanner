use crate::core::types::Result;
use std::thread::JoinHandle;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

pub struct WindowTaskComponent {
    pub window_index: usize,
    pub task_handle: JoinHandle<Result<WindowTaskResult>>,
    pub cancellation_token: CancellationToken,
    pub started_at: Instant,
}

impl std::fmt::Debug for WindowTaskComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowTaskComponent")
            .field("window_index", &self.window_index)
            .field("task_handle", &"<JoinHandle>")
            .field("cancellation_token", &self.cancellation_token)
            .field("started_at", &self.started_at)
            .finish()
    }
}

#[derive(Debug)]
pub struct WindowTaskResult {
    pub window_index: usize,
    pub candidates: Vec<CandidateData>,
    pub completed_at: Instant,
}

#[derive(Debug, Clone)]
pub struct CandidateData {
    pub frequency_hz: f64,
    pub signal_strength: f64,
    pub audio_quality: crate::audio::quality::AudioQuality,
}
