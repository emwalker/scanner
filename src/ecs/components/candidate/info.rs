//! Candidate information component

use crate::audio::quality::AudioQuality;

/// Component holding candidate signal information
#[derive(Debug, Clone)]
pub struct CandidateInfoComponent {
    pub frequency_hz: f64,
    pub signal_strength: Option<f64>,
    pub audio_quality: Option<AudioQuality>,
}

impl CandidateInfoComponent {
    pub fn new(frequency_hz: f64) -> Self {
        Self {
            frequency_hz,
            signal_strength: None,
            audio_quality: None,
        }
    }

    pub fn with_signal_strength(mut self, strength: f64) -> Self {
        self.signal_strength = Some(strength);
        self
    }

    pub fn with_audio_quality(mut self, quality: AudioQuality) -> Self {
        self.audio_quality = Some(quality);
        self
    }

    pub fn set_audio_quality(&mut self, quality: AudioQuality) {
        self.audio_quality = Some(quality);
    }

    pub fn set_signal_strength(&mut self, strength: f64) {
        self.signal_strength = Some(strength);
    }
}
