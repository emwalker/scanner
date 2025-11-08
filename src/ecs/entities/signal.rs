use crate::{
    core::signals::ModulationType,
    ecs::{
        Entity,
        components::{
            AnalysisInputComponent,
            signal::*,
            station::{TuneState, TuneTransitionComponent},
            window::WindowId,
        },
    },
};

pub struct SignalEntity {
    id: SignalId,
    pub info: SignalInfoComponent,
    pub discovery: SignalDiscoveryComponent,
    pub analysis: AnalysisStateComponent,
    pub analysis_input: Option<AnalysisInputComponent>,
    pub history: SignalHistoryComponent,
    pub playback: SignalPlaybackComponent,
    pub tune_state: TuneState,
    pub selection: SelectionComponent,
    pub coordination: SignalCoordinationComponent,
}

impl SignalEntity {
    pub fn new(frequency_hz: f64, window_id: WindowId, modulation: ModulationType) -> Self {
        Self {
            id: SignalId::new(frequency_hz, modulation.clone()),
            info: SignalInfoComponent::new(frequency_hz, modulation),
            discovery: SignalDiscoveryComponent::new(window_id),
            analysis: AnalysisStateComponent::new(),
            analysis_input: None,
            history: SignalHistoryComponent::new(),
            playback: SignalPlaybackComponent::new(),
            tune_state: TuneState::Idle,
            selection: SelectionComponent::new(),
            coordination: SignalCoordinationComponent::new(),
        }
    }

    pub fn frequency(&self) -> f64 {
        self.info.frequency()
    }

    pub fn window_id(&self) -> &WindowId {
        self.discovery.window_id()
    }

    pub fn is_selected(&self) -> bool {
        self.selection.is_selected()
    }

    pub fn request_tune_transition(
        &mut self,
        window_id: WindowId,
        center_freq: f64,
    ) -> std::result::Result<(), String> {
        if !self.analysis.is_confirmed() {
            return Err("Cannot tune unconfirmed signal".to_string());
        }

        if !matches!(self.tune_state, TuneState::Idle) {
            return Err("Signal already tuning".to_string());
        }

        self.tune_state =
            TuneState::Transitioning(TuneTransitionComponent::new(window_id, center_freq));
        Ok(())
    }

    pub fn clear_tune_state(&mut self) {
        self.tune_state = TuneState::Idle;
    }

    pub fn completion(&self) -> f64 {
        use crate::ecs::components::signal::PlaybackState;

        if self.playback.state() == PlaybackState::Completed {
            1.0
        } else if self.playback.state() == PlaybackState::Playing {
            0.8
        } else if self.analysis.is_confirmed() {
            0.6
        } else if self.analysis.is_in_progress() {
            0.5
        } else if self.analysis.is_rejected() {
            1.0
        } else {
            0.3
        }
    }

    pub fn status(&self) -> crate::ecs::components::AnalysisStatus {
        use crate::ecs::components::AnalysisStatus;

        if self.analysis.is_not_started() {
            AnalysisStatus::Detected
        } else if self.analysis.is_in_progress() {
            AnalysisStatus::Analyzing
        } else if self.analysis.is_confirmed() {
            let quality = self
                .info
                .audio_quality()
                .unwrap_or(crate::audio::quality::AudioQuality::NoAudio);
            let strength = self.info.signal_strength().unwrap_or(0.0);
            AnalysisStatus::Signal { quality, strength }
        } else if self.analysis.is_rejected() {
            let quality = self
                .info
                .audio_quality()
                .unwrap_or(crate::audio::quality::AudioQuality::NoAudio);
            let strength = self.info.signal_strength().unwrap_or(0.0);
            AnalysisStatus::Rejected { quality, strength }
        } else {
            AnalysisStatus::Detected
        }
    }

    pub fn set_analysis_input(&mut self, input: AnalysisInputComponent) {
        self.analysis_input = Some(input);
    }

    pub fn take_analysis_input(&mut self) -> Option<AnalysisInputComponent> {
        self.analysis_input.take()
    }
}

impl Entity for SignalEntity {
    type Id = SignalId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio::quality::AudioQuality, core::types::Result, ecs::TaskId};

    fn create_test_window_id() -> WindowId {
        WindowId::new(TaskId::new("test-scan"), 5)
    }

    #[test]
    fn test_signal_entity_creation() {
        let window_id = create_test_window_id();
        let signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        assert_eq!(signal.frequency(), 88.9e6);
        assert_eq!(signal.window_id(), &window_id);
        assert!(signal.analysis.is_not_started());
        assert!(matches!(signal.tune_state, TuneState::Idle));
        assert_eq!(signal.history.play_count(), 0);
        assert_eq!(signal.playback.state(), PlaybackState::NotPlaying);
        assert!(!signal.is_selected());
    }

    #[test]
    fn test_signal_id_format() {
        let window_id = create_test_window_id();
        let signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        assert_eq!(signal.id().as_str(), "000.088.900.000-WFM");
    }

    #[test]
    fn test_tune_requires_confirmed_analysis() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        let result = signal.request_tune_transition(window_id, 88.9e6);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot tune unconfirmed signal");
        assert!(matches!(signal.tune_state, TuneState::Idle));
    }

    #[test]
    fn test_tune_blocked_during_analysis() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        let (handle, rx) = create_test_analysis_thread();
        signal.analysis.start_analysis(handle, rx);

        let result = signal.request_tune_transition(window_id, 88.9e6);

        assert!(result.is_err());
        assert!(matches!(signal.tune_state, TuneState::Idle));
    }

    #[test]
    fn test_tune_blocked_when_rejected() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        signal.analysis.reject_analysis(AudioQuality::NoAudio, 0.05);

        let result = signal.request_tune_transition(window_id, 88.9e6);

        assert!(result.is_err());
        assert!(matches!(signal.tune_state, TuneState::Idle));
    }

    #[test]
    fn test_tune_succeeds_when_confirmed() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        signal.analysis.confirm_analysis(AudioQuality::Good, 0.85);

        let result = signal.request_tune_transition(window_id.clone(), 88.9e6);

        assert!(result.is_ok());
        assert!(matches!(signal.tune_state, TuneState::Transitioning(_)));
    }

    #[test]
    fn test_complete_signal_lifecycle() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        // Detection
        assert!(signal.analysis.is_not_started());
        assert_eq!(signal.history.play_count(), 0);

        // Analysis
        let (handle, rx) = create_test_analysis_thread();
        signal.analysis.start_analysis(handle, rx);
        assert!(signal.analysis.is_in_progress());

        // Confirmation
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.85);
        signal.info.set_audio_quality(Some(AudioQuality::Good));
        signal.info.set_signal_strength(Some(0.85));
        assert!(signal.analysis.is_confirmed());

        // Tuning
        assert!(
            signal
                .request_tune_transition(window_id.clone(), 88.9e6)
                .is_ok()
        );
        assert!(matches!(signal.tune_state, TuneState::Transitioning(_)));

        // Playback
        signal.history.start_play_session();
        signal.playback.transition_to(PlaybackState::Playing);
        assert_eq!(signal.playback.state(), PlaybackState::Playing);

        // Completion
        signal.playback.transition_to(PlaybackState::Completed);
        signal.history.end_play_session();
        assert_eq!(signal.history.play_count(), 1);
        assert!(signal.history.total_play_duration() > std::time::Duration::ZERO);
    }

    #[test]
    fn test_rejected_signal_cannot_tune() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        // Reject analysis
        signal.analysis.reject_analysis(AudioQuality::NoAudio, 0.05);
        signal
            .coordination
            .set_rejection_reason(Some("Below threshold".to_string()));

        // Tune should fail
        assert!(signal.request_tune_transition(window_id, 88.9e6).is_err());
        assert!(matches!(signal.tune_state, TuneState::Idle));

        // Reason should be available
        assert_eq!(
            signal.coordination.rejection_reason(),
            Some("Below threshold".to_string())
        );
    }

    #[test]
    fn test_signal_selection() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        assert!(!signal.is_selected());

        signal.selection.select();
        assert!(signal.is_selected());
        assert!(signal.selection.selected_at().is_some());

        signal.selection.deselect();
        assert!(!signal.is_selected());
    }

    #[test]
    fn test_audio_spawn_coordination() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.85);

        // First spawn succeeds
        assert!(!signal.coordination.audio_request_enqueued());
        signal.coordination.set_audio_request_enqueued(true);

        // Second spawn would check flag
        assert!(signal.coordination.audio_request_enqueued());
    }

    #[test]
    fn test_multiple_play_sessions() {
        let window_id = create_test_window_id();
        let mut signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);

        // Session 1
        signal.history.start_play_session();
        std::thread::sleep(std::time::Duration::from_millis(30));
        signal.history.end_play_session();

        // Session 2
        signal.history.start_play_session();
        std::thread::sleep(std::time::Duration::from_millis(30));
        signal.history.end_play_session();

        assert_eq!(signal.history.play_count(), 2);
        assert!(signal.history.total_play_duration() >= std::time::Duration::from_millis(60));
        assert!(signal.history.last_heard().is_some());
    }

    use crate::ecs::EntityWorld;

    #[test]
    fn test_signal_entity_in_world() {
        let mut world = EntityWorld::<SignalEntity>::new();
        let window_id = WindowId::new(TaskId::new("test-scan"), 5);

        let signal = SignalEntity::new(88.9e6, window_id, ModulationType::WFM);
        let signal_id = signal.id().clone();

        world.insert(signal);

        assert_eq!(world.len(), 1);
        let retrieved = world.get(&signal_id).unwrap();
        assert_eq!(retrieved.frequency(), 88.9e6);
    }

    #[test]
    fn test_multiple_signals_in_world() {
        let mut world = EntityWorld::<SignalEntity>::new();
        let window_id = WindowId::new(TaskId::new("test-scan"), 5);

        let signal1 = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
        let signal2 = SignalEntity::new(89.3e6, window_id, ModulationType::WFM);

        world.insert(signal1);
        world.insert(signal2);

        assert_eq!(world.len(), 2);
    }

    fn create_test_analysis_thread() -> (
        std::thread::JoinHandle<Result<AnalysisResults>>,
        std::sync::mpsc::Receiver<AnalysisResults>,
    ) {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            drop(tx);
            Ok(AnalysisResults {
                quality: AudioQuality::Good,
                strength: 0.85,
            })
        });
        (handle, rx)
    }
}
