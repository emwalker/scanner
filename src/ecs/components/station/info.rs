//! Station information component

use crate::audio::quality::AudioQuality;

/// Component holding station information
#[derive(Debug, Clone)]
pub struct StationInfoComponent {
    /// Center frequency in Hz
    pub frequency: f64,

    /// Signal strength (0.0 to 1.0)
    pub signal_strength: f32,

    /// Audio quality assessment
    pub audio_quality: Option<AudioQuality>,

    /// Optional station name (e.g., from RDS or user-provided)
    pub name: Option<String>,
}

impl StationInfoComponent {
    /// Create a new station info component
    pub fn new(frequency: f64, signal_strength: f32, audio_quality: Option<AudioQuality>) -> Self {
        Self {
            frequency,
            signal_strength,
            audio_quality,
            name: None,
        }
    }

    /// Update the signal strength
    pub fn update_signal_strength(&mut self, signal_strength: f32) {
        self.signal_strength = signal_strength;
    }

    /// Update the audio quality
    pub fn update_audio_quality(&mut self, audio_quality: AudioQuality) {
        self.audio_quality = Some(audio_quality);
    }

    /// Set the station name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_station_info() {
        let info = StationInfoComponent::new(88.9e6, 0.8, Some(AudioQuality::Good));

        assert_eq!(info.frequency, 88.9e6);
        assert_eq!(info.signal_strength, 0.8);
        assert_eq!(info.audio_quality, Some(AudioQuality::Good));
        assert_eq!(info.name, None);
    }

    #[test]
    fn test_update_signal_strength() {
        let mut info = StationInfoComponent::new(88.9e6, 0.8, None);
        info.update_signal_strength(0.9);
        assert_eq!(info.signal_strength, 0.9);
    }

    #[test]
    fn test_update_audio_quality() {
        let mut info = StationInfoComponent::new(88.9e6, 0.8, None);
        info.update_audio_quality(AudioQuality::Moderate);
        assert_eq!(info.audio_quality, Some(AudioQuality::Moderate));
    }

    #[test]
    fn test_set_name() {
        let mut info = StationInfoComponent::new(88.9e6, 0.8, None);
        info.set_name("KEXP".to_string());
        assert_eq!(info.name, Some("KEXP".to_string()));
    }
}
