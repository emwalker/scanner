//! Candidate entity combining candidate components

use crate::audio::quality::AudioQuality;
use crate::ecs::Entity;
use crate::ecs::components::{
    CandidateId, CandidateInfoComponent, CandidateLifecycleComponent, CandidateProgressComponent,
    CandidateState,
};
use crate::scanning::window::WindowMetadata;

/// Entity representing a signal candidate during scanning
#[derive(Debug, Clone)]
pub struct CandidateEntity {
    id: CandidateId,
    pub info: CandidateInfoComponent,
    pub lifecycle: CandidateLifecycleComponent,
    pub progress: CandidateProgressComponent,
}

impl CandidateEntity {
    /// Create a new candidate from peak detection
    pub fn new(frequency_hz: f64, metadata: WindowMetadata) -> Self {
        Self {
            id: CandidateId::new(frequency_hz, metadata.window_id),
            info: CandidateInfoComponent::new(frequency_hz),
            lifecycle: CandidateLifecycleComponent::new(),
            progress: CandidateProgressComponent::new(metadata),
        }
    }

    /// Get candidate frequency
    pub fn frequency(&self) -> f64 {
        self.info.frequency_hz
    }

    /// Get window ID
    pub fn window_id(&self) -> usize {
        self.progress.window_id()
    }

    /// Get current state
    pub fn state(&self) -> CandidateState {
        self.lifecycle.state()
    }

    /// Get completion percentage based on state
    pub fn completion(&self) -> f64 {
        self.lifecycle.state().completion()
    }

    /// Get audio quality if available
    pub fn audio_quality(&self) -> Option<AudioQuality> {
        self.info.audio_quality
    }

    /// Get signal strength if available
    pub fn signal_strength(&self) -> Option<f64> {
        self.info.signal_strength
    }

    /// Transition to analyzing state
    pub fn start_analysis(&mut self) {
        self.lifecycle.transition_to(CandidateState::Analyzing);
    }

    /// Mark as valid signal with quality
    pub fn mark_as_signal(&mut self, quality: AudioQuality, strength: Option<f64>) {
        self.info.set_audio_quality(quality);
        if let Some(s) = strength {
            self.info.set_signal_strength(s);
        }
        self.lifecycle.transition_to(CandidateState::Signal);
    }

    /// Mark as rejected
    pub fn reject(&mut self) {
        self.lifecycle.transition_to(CandidateState::Rejected);
    }

    /// Start playing audio
    pub fn start_playback(&mut self) {
        self.lifecycle.transition_to(CandidateState::Playing);
    }

    /// Complete playback
    pub fn complete_playback(&mut self) {
        self.lifecycle.transition_to(CandidateState::Completed);
    }

    /// Check if candidate is in specific states
    pub fn is_detected(&self) -> bool {
        self.lifecycle.is_detected()
    }

    pub fn is_analyzing(&self) -> bool {
        self.lifecycle.is_analyzing()
    }

    pub fn is_signal(&self) -> bool {
        self.lifecycle.is_signal()
    }

    pub fn is_rejected(&self) -> bool {
        self.lifecycle.is_rejected()
    }

    pub fn is_playing(&self) -> bool {
        self.lifecycle.is_playing()
    }

    pub fn is_completed(&self) -> bool {
        self.lifecycle.is_completed()
    }
}

impl Entity for CandidateEntity {
    type Id = CandidateId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metadata() -> WindowMetadata {
        WindowMetadata {
            window_id: 0,
            center_frequency_hz: 88.9e6,
        }
    }

    #[test]
    fn test_candidate_creation() {
        let metadata = create_test_metadata();
        let candidate = CandidateEntity::new(88.9e6, metadata);

        assert_eq!(candidate.frequency(), 88.9e6);
        assert_eq!(candidate.window_id(), 0);
        assert!(candidate.is_detected());
        assert_eq!(candidate.completion(), 0.3);
    }

    #[test]
    fn test_candidate_lifecycle() {
        let metadata = create_test_metadata();
        let mut candidate = CandidateEntity::new(88.9e6, metadata);

        candidate.start_analysis();
        assert!(candidate.is_analyzing());
        assert_eq!(candidate.completion(), 0.5);

        candidate.mark_as_signal(AudioQuality::Good, Some(0.8));
        assert!(candidate.is_signal());
        assert_eq!(candidate.audio_quality(), Some(AudioQuality::Good));
        assert_eq!(candidate.signal_strength(), Some(0.8));
        assert_eq!(candidate.completion(), 0.6);

        candidate.start_playback();
        assert!(candidate.is_playing());
        assert_eq!(candidate.completion(), 0.8);

        candidate.complete_playback();
        assert!(candidate.is_completed());
        assert_eq!(candidate.completion(), 1.0);
    }

    #[test]
    fn test_candidate_rejection() {
        let metadata = create_test_metadata();
        let mut candidate = CandidateEntity::new(88.9e6, metadata);

        candidate.start_analysis();
        candidate.reject();

        assert!(candidate.is_rejected());
        assert_eq!(candidate.completion(), 1.0);
    }
}
