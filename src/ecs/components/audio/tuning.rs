//! Audio tuning component

use crate::core::types::Signal;

/// Component tracking what frequency is being tuned and the signal characteristics
#[derive(Debug, Clone)]
pub struct AudioTuningComponent {
    /// The signal being played
    pub signal: Signal,

    /// Center frequency the tuner is tuned to
    pub center_frequency_hz: f64,
}

impl AudioTuningComponent {
    pub fn new(signal: Signal, center_frequency_hz: f64) -> Self {
        Self {
            signal,
            center_frequency_hz,
        }
    }

    pub fn frequency(&self) -> f64 {
        self.signal.frequency_hz
    }

    pub fn signal_strength(&self) -> f32 {
        self.signal.signal_strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioQuality;
    use crate::core::types::ModulationType;
    use std::time::SystemTime;

    fn create_test_signal() -> Signal {
        Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_create_tuning() {
        let signal = create_test_signal();
        let tuning = AudioTuningComponent::new(signal.clone(), 88.9e6);

        assert_eq!(tuning.frequency(), 88.9e6);
        assert_eq!(tuning.signal_strength(), 0.8);
        assert_eq!(tuning.center_frequency_hz, 88.9e6);
    }

    #[test]
    fn test_convenience_methods() {
        let signal = create_test_signal();
        let tuning = AudioTuningComponent::new(signal.clone(), 88.9e6);

        assert_eq!(tuning.frequency(), signal.frequency_hz);
        assert_eq!(tuning.signal_strength(), signal.signal_strength);
    }
}
