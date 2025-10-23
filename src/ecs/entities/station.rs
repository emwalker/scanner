//! Station entity combining station components

use std::time::{Duration, Instant};

use crate::{
    core::types::Signal,
    ecs::{
        Entity,
        components::{
            station::{
                StationDiscoveryComponent, StationHistoryComponent, StationId,
                StationInfoComponent, StationPlaybackComponent, TuneState, TuneTransitionComponent,
            },
            window::WindowId,
        },
    },
};

/// Entity representing a discovered or known station
#[derive(Debug, Clone)]
pub struct StationEntity {
    id: StationId,
    pub info: StationInfoComponent,
    pub discovery: StationDiscoveryComponent,
    pub history: StationHistoryComponent,
    pub playback: StationPlaybackComponent,
    pub tune_state: TuneState,
}

impl StationEntity {
    /// Create a new station entity from a discovered signal
    pub fn from_signal(signal: &Signal, window_id: WindowId) -> Self {
        Self {
            id: StationId::new(),
            info: StationInfoComponent::new(
                signal.frequency_hz,
                signal.signal_strength,
                Some(signal.audio_quality),
            ),
            discovery: StationDiscoveryComponent::new(window_id),
            history: StationHistoryComponent::new(),
            playback: StationPlaybackComponent::new(),
            tune_state: TuneState::Idle,
        }
    }

    /// Set tune transition state
    pub fn set_tune_transition(&mut self, window_id: WindowId, center_frequency: f64) {
        self.tune_state =
            TuneState::Transitioning(TuneTransitionComponent::new(window_id, center_frequency));
    }

    /// Check if station is awaiting tuner allocation
    pub fn is_awaiting_tuner(&self) -> bool {
        matches!(
            self.tune_state,
            TuneState::Transitioning(_) | TuneState::RequestQueued { .. }
        )
    }

    /// Check if station is actively tuned
    pub fn is_actively_tuned(&self) -> bool {
        matches!(self.tune_state, TuneState::Active { .. })
    }

    /// Clear tune state back to Idle
    pub fn clear_tune_state(&mut self) {
        self.tune_state = TuneState::Idle;
    }

    /// Temporary bridge method for existing code
    /// TODO: Remove after TunerAllocationSystem handles this
    pub fn clear_tune_request(&mut self) {
        self.tune_state = TuneState::Idle;
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
        self.playback.is_playing()
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
    use std::time::SystemTime;

    use proptest::prelude::*;

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        ecs::{TaskId, components::window::WindowId},
    };

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

    fn create_window_id() -> WindowId {
        WindowId::new(TaskId::new("test-scan"), 1)
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
                let window_id = WindowId::new(TaskId::new("test-scan"), 1);
                StationEntity::from_signal(&signal, window_id)
            })
    }

    #[test]
    fn test_from_signal() {
        let signal = create_test_signal();
        let window_id = create_window_id();

        let station = StationEntity::from_signal(&signal, window_id);

        assert_eq!(station.frequency(), 88.9e6);
        assert_eq!(station.signal_strength(), 0.8);
        assert_eq!(station.info.audio_quality, Some(AudioQuality::Good));
        assert_eq!(station.play_count(), 0);
        assert!(!station.is_playing());
    }

    #[test]
    fn test_play_tracking() {
        let signal = create_test_signal();
        let window_id = create_window_id();

        let mut station = StationEntity::from_signal(&signal, window_id);

        use crate::ecs::AudioId;
        let audio_id = AudioId::new();

        station.playback.start_playing(audio_id);
        station.history.record_play_start();
        assert!(station.is_playing());
        assert_eq!(station.play_count(), 1);

        std::thread::sleep(Duration::from_millis(10));

        station.playback.stop_playing();
        station.history.record_play_end();

        assert!(!station.is_playing());
        assert!(station.total_play_duration().as_millis() >= 10);
    }

    #[test]
    fn test_entity_trait() {
        let signal = create_test_signal();
        let window_id = create_window_id();

        let station1 = StationEntity::from_signal(&signal, window_id.clone());
        let station2 = StationEntity::from_signal(&signal, window_id);

        assert_ne!(
            station1.id(),
            station2.id(),
            "Each entity should have unique ID"
        );
    }

    #[test]
    fn test_convenience_methods() {
        let signal = create_test_signal();
        let window_id = create_window_id();

        let station = StationEntity::from_signal(&signal, window_id);

        assert_eq!(station.frequency(), signal.frequency_hz);
        assert_eq!(station.signal_strength(), signal.signal_strength);
        assert_eq!(station.last_played_at(), None);
        assert_eq!(station.play_count(), 0);
        assert_eq!(station.total_play_duration(), Duration::ZERO);
    }

    #[test]
    fn test_is_playing_uses_playback_not_history() {
        let signal = create_test_signal();
        let window_id = create_window_id();

        let mut station = StationEntity::from_signal(&signal, window_id);

        station.history.record_play_start();
        assert!(station.history.is_playing(), "history should say playing");

        station.playback.stop_playing();
        assert!(
            !station.playback.is_playing(),
            "playback should say not playing"
        );

        assert!(
            !station.is_playing(),
            "is_playing() should check playback, not history"
        );
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
