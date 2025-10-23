use crate::audio::quality::{AudioAnalyzer, AudioQuality};

/// Audio-related configuration
#[derive(Clone)]
pub struct AudioConfig {
    pub buffer_size: u32,
    pub sample_rate: u32,
    pub analyzer: AudioAnalyzer,
    pub squelch: SquelchConfig,
    pub volume: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            sample_rate: 48000,
            analyzer: AudioAnalyzer::pass_through(),
            squelch: SquelchConfig::default(),
            volume: 0.5,
        }
    }
}

/// Squelch configuration
#[derive(Clone)]
pub struct SquelchConfig {
    pub disabled: bool,
    pub threshold: AudioQuality,
    pub learning_duration: f32,
}

impl Default for SquelchConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            threshold: AudioQuality::Moderate,
            learning_duration: 1.0,
        }
    }
}
