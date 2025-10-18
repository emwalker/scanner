//! Candidate lifecycle component

use std::time::Instant;

/// States a candidate can be in during its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateState {
    /// Peak detected in FFT
    Detected,
    /// Audio analysis in progress
    Analyzing,
    /// Passed squelch, has valid signal
    Signal,
    /// Failed quality threshold
    Rejected,
    /// Currently playing audio
    Playing,
    /// Finished playing
    Completed,
}

impl CandidateState {
    /// Get completion percentage for this state
    pub fn completion(&self) -> f64 {
        match self {
            CandidateState::Detected => 0.3,
            CandidateState::Analyzing => 0.5,
            CandidateState::Signal => 0.6,
            CandidateState::Rejected => 1.0,
            CandidateState::Playing => 0.8,
            CandidateState::Completed => 1.0,
        }
    }
}

/// Component tracking candidate lifecycle state
#[derive(Debug, Clone)]
pub struct CandidateLifecycleComponent {
    state: CandidateState,
    detected_at: Instant,
    last_transition: Instant,
}

impl CandidateLifecycleComponent {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            state: CandidateState::Detected,
            detected_at: now,
            last_transition: now,
        }
    }

    pub fn state(&self) -> CandidateState {
        self.state
    }

    pub fn detected_at(&self) -> Instant {
        self.detected_at
    }

    pub fn last_transition(&self) -> Instant {
        self.last_transition
    }

    pub fn transition_to(&mut self, new_state: CandidateState) {
        self.state = new_state;
        self.last_transition = Instant::now();
    }

    pub fn is_detected(&self) -> bool {
        matches!(self.state, CandidateState::Detected)
    }

    pub fn is_analyzing(&self) -> bool {
        matches!(self.state, CandidateState::Analyzing)
    }

    pub fn is_signal(&self) -> bool {
        matches!(self.state, CandidateState::Signal)
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self.state, CandidateState::Rejected)
    }

    pub fn is_playing(&self) -> bool {
        matches!(self.state, CandidateState::Playing)
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.state, CandidateState::Completed)
    }
}

impl Default for CandidateLifecycleComponent {
    fn default() -> Self {
        Self::new()
    }
}
