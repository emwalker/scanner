//! Station entity combining station components

use crate::core::types::Signal;
use crate::ecs::Entity;
use crate::ecs::components::scan::ScanId;
use crate::ecs::components::station::{
    StationDiscoveryComponent, StationHistoryComponent, StationId, StationInfoComponent,
};
use crate::scanning::window::WindowMetadata;
use std::time::{Duration, Instant};

/// Entity representing a discovered or known station
#[derive(Debug, Clone)]
pub struct StationEntity {
    pub id: StationId,
    pub info: StationInfoComponent,
    pub discovery: StationDiscoveryComponent,
    pub history: StationHistoryComponent,
}

impl StationEntity {
    /// Create a new station entity from a discovered signal
    pub fn from_signal(signal: &Signal, scan_id: ScanId, window_metadata: WindowMetadata) -> Self {
        Self {
            id: StationId::new(),
            info: StationInfoComponent::new(
                signal.frequency_hz,
                signal.signal_strength,
                Some(signal.audio_quality),
            ),
            discovery: StationDiscoveryComponent::new(
                scan_id,
                window_metadata.window_id,
                window_metadata,
            ),
            history: StationHistoryComponent::new(),
        }
    }

    /// Get the station frequency
    pub fn frequency(&self) -> f64 {
        self.info.frequency
    }

    /// Get the signal strength
    pub fn signal_strength(&self) -> f32 {
        self.info.signal_strength
    }

    /// Get when the station was last played
    pub fn last_played_at(&self) -> Option<Instant> {
        self.history.last_heard
    }

    /// Get the number of times this station was played
    pub fn play_count(&self) -> usize {
        self.history.play_count
    }

    /// Check if this station is currently playing
    pub fn is_playing(&self) -> bool {
        self.history.is_playing()
    }

    /// Get total play duration
    pub fn total_play_duration(&self) -> Duration {
        self.history.total_play_duration
    }
}

impl Entity for StationEntity {
    type Id = StationId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioQuality;
    use std::time::SystemTime;

    fn create_test_signal() -> Signal {
        Signal {
            frequency_hz: 88.9e6,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88.9e6,
            audio_quality: AudioQuality::Good,
        }
    }

    fn create_test_metadata() -> WindowMetadata {
        WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        }
    }

    #[test]
    fn test_from_signal() {
        let signal = create_test_signal();
        let scan_id = ScanId::new();
        let metadata = create_test_metadata();

        let station = StationEntity::from_signal(&signal, scan_id, metadata);

        assert_eq!(station.frequency(), 88.9e6);
        assert_eq!(station.signal_strength(), 0.8);
        assert_eq!(station.info.audio_quality, Some(AudioQuality::Good));
        assert_eq!(station.play_count(), 0);
        assert!(!station.is_playing());
    }

    #[test]
    fn test_play_tracking() {
        let signal = create_test_signal();
        let scan_id = ScanId::new();
        let metadata = create_test_metadata();

        let mut station = StationEntity::from_signal(&signal, scan_id, metadata);

        station.history.record_play_start();
        assert!(station.is_playing());
        assert_eq!(station.play_count(), 1);

        std::thread::sleep(Duration::from_millis(10));
        station.history.record_play_end();

        assert!(!station.is_playing());
        assert!(station.total_play_duration().as_millis() >= 10);
    }

    #[test]
    fn test_entity_trait() {
        let signal = create_test_signal();
        let scan_id = ScanId::new();
        let metadata = create_test_metadata();

        let station = StationEntity::from_signal(&signal, scan_id, metadata);
        let id = station.id();

        assert_eq!(id, &station.id);
    }

    #[test]
    fn test_convenience_methods() {
        let signal = create_test_signal();
        let scan_id = ScanId::new();
        let metadata = create_test_metadata();

        let station = StationEntity::from_signal(&signal, scan_id, metadata);

        assert_eq!(station.frequency(), signal.frequency_hz);
        assert_eq!(station.signal_strength(), signal.signal_strength);
        assert_eq!(station.last_played_at(), None);
        assert_eq!(station.play_count(), 0);
        assert_eq!(station.total_play_duration(), Duration::ZERO);
    }
}
