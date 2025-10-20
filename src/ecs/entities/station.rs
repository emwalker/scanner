//! Station entity combining station components

use crate::core::types::Signal;
use crate::ecs::Entity;
use crate::ecs::components::scan::ScanId;
use crate::ecs::components::station::{
    StationDiscoveryComponent, StationHistoryComponent, StationId, StationInfoComponent,
    StationPlaybackComponent, TuneRequestComponent, TuneTransitionComponent,
};
use crate::scanning::window::WindowMetadata;
use std::time::{Duration, Instant};

/// Entity representing a discovered or known station
#[derive(Debug, Clone)]
pub struct StationEntity {
    id: StationId,
    pub info: StationInfoComponent,
    pub discovery: StationDiscoveryComponent,
    pub history: StationHistoryComponent,
    pub playback: StationPlaybackComponent,
    pub tune_request: Option<TuneRequestComponent>,
    pub transition: Option<TuneTransitionComponent>,
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
            playback: StationPlaybackComponent::new(),
            tune_request: None,
            transition: None,
        }
    }

    /// Request tuning to this station
    pub fn request_tune(&mut self, window_id: usize, center_frequency: f64) {
        self.tune_request = Some(TuneRequestComponent::new(window_id, center_frequency));
    }

    /// Clear tune request
    pub fn clear_tune_request(&mut self) {
        self.tune_request = None;
    }

    /// Check if station has pending tune request
    pub fn has_tune_request(&self) -> bool {
        self.tune_request.is_some()
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
    use proptest::prelude::*;
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

    fn arb_audio_quality() -> impl Strategy<Value = AudioQuality> {
        prop_oneof![
            Just(AudioQuality::Good),
            Just(AudioQuality::Moderate),
            Just(AudioQuality::Poor),
            Just(AudioQuality::NoAudio),
            Just(AudioQuality::Static),
            Just(AudioQuality::Unknown),
        ]
    }

    fn arb_station_entity() -> impl Strategy<Value = StationEntity> {
        (
            88.0e6..108.0e6f64,
            0.0..=1.0f32,
            prop::option::of(arb_audio_quality()),
        )
            .prop_map(|(frequency, signal_strength, audio_quality)| {
                let signal = Signal {
                    frequency_hz: frequency,
                    signal_strength,
                    bandwidth_hz: 200_000.0,
                    modulation: crate::core::types::ModulationType::WFM,
                    audio_sample_rate: 48000,
                    detected_at: SystemTime::now(),
                    analysis_duration_ms: 100,
                    detection_center_freq: frequency,
                    audio_quality: audio_quality.unwrap_or(AudioQuality::Good),
                };
                let metadata = WindowMetadata {
                    center_frequency_hz: frequency,
                    window_id: 1,
                };
                StationEntity::from_signal(&signal, ScanId::new(), metadata)
            })
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

        let station1 = StationEntity::from_signal(&signal, scan_id, metadata);
        let station2 = StationEntity::from_signal(&signal, scan_id, metadata);

        assert_ne!(
            station1.id(),
            station2.id(),
            "Each entity should have unique ID"
        );
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

    proptest! {
        #[test]
        fn prop_signal_strength_bounds(station in arb_station_entity()) {
            let strength = station.signal_strength();
            prop_assert!(strength >= 0.0, "Signal strength {} is negative", strength);
            prop_assert!(strength <= 1.0, "Signal strength {} exceeds 1.0", strength);
        }

        #[test]
        fn prop_frequency_in_fm_band(station in arb_station_entity()) {
            let freq = station.frequency();
            prop_assert!(freq >= 88.0e6);
            prop_assert!(freq <= 108.0e6);
        }

        #[test]
        fn prop_initial_state_consistency(station in arb_station_entity()) {
            prop_assert_eq!(station.play_count(), 0);
            prop_assert!(!station.is_playing());
            prop_assert_eq!(station.last_played_at(), None);
            prop_assert_eq!(station.total_play_duration(), Duration::ZERO);
        }

        #[test]
        fn prop_convenience_methods_match_components(station in arb_station_entity()) {
            prop_assert_eq!(station.frequency(), station.info.frequency);
            prop_assert_eq!(station.signal_strength(), station.info.signal_strength);
            prop_assert_eq!(station.play_count(), station.history.play_count);
        }
    }
}
