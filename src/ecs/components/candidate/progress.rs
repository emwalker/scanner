//! Candidate progress component

use crate::scanning::window::WindowMetadata;

/// Component tracking candidate discovery context
#[derive(Debug, Clone, Copy)]
pub struct CandidateProgressComponent {
    pub metadata: WindowMetadata,
}

impl CandidateProgressComponent {
    pub fn new(metadata: WindowMetadata) -> Self {
        Self { metadata }
    }

    pub fn window_id(&self) -> usize {
        self.metadata.window_id
    }

    pub fn center_frequency_hz(&self) -> f64 {
        self.metadata.center_frequency_hz
    }
}
