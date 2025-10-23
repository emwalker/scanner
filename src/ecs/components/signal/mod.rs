use std::{
    fmt,
    sync::{Mutex, mpsc},
    thread::JoinHandle,
    time::Instant,
};

use crate::{
    audio::quality::AudioQuality, core::types::Result as ScannerResult,
    ecs::components::window::WindowId,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignalId(String);

impl SignalId {
    pub fn new(frequency_hz: f64, window_id: WindowId) -> Self {
        let frequency_mhz = frequency_hz / 1e6;
        let task_id = &window_id.task_id;
        let window_index = window_id.window_index;

        Self(format!("{}-{}-{}", frequency_mhz, task_id, window_index))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SignalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct SignalInfoComponent {
    frequency_hz: f64,
    signal_strength: Option<f64>,
    audio_quality: Option<AudioQuality>,
    name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SignalDiscoveryComponent {
    discovered_at: Instant,
    window_id: WindowId,
}

#[derive(Debug, Clone)]
pub struct AnalysisResults {
    pub quality: AudioQuality,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub enum AnalysisStatus {
    Detected,
    Analyzing,
    Signal {
        quality: AudioQuality,
        strength: f64,
    },
    Rejected {
        quality: AudioQuality,
        strength: f64,
    },
    Error,
}

#[derive(Debug)]
pub enum AnalysisState {
    NotStarted,
    InProgress {
        thread_handle: JoinHandle<ScannerResult<AnalysisResults>>,
        result_rx: Mutex<mpsc::Receiver<AnalysisResults>>,
        started_at: Instant,
    },
    Confirmed {
        quality: AudioQuality,
        strength: f64,
    },
    Rejected {
        quality: AudioQuality,
        strength: f64,
    },
    Error {
        error: String,
    },
}

#[derive(Debug)]
pub struct AnalysisStateComponent {
    state: AnalysisState,
}

impl AnalysisStateComponent {
    pub fn new() -> Self {
        Self {
            state: AnalysisState::NotStarted,
        }
    }

    pub fn is_not_started(&self) -> bool {
        matches!(self.state, AnalysisState::NotStarted)
    }

    pub fn is_in_progress(&self) -> bool {
        matches!(self.state, AnalysisState::InProgress { .. })
    }

    pub fn is_confirmed(&self) -> bool {
        matches!(self.state, AnalysisState::Confirmed { .. })
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self.state, AnalysisState::Rejected { .. })
    }

    pub fn state(&self) -> &AnalysisState {
        &self.state
    }

    pub fn start_analysis(
        &mut self,
        thread_handle: JoinHandle<ScannerResult<AnalysisResults>>,
        result_rx: mpsc::Receiver<AnalysisResults>,
    ) {
        self.state = AnalysisState::InProgress {
            thread_handle,
            result_rx: Mutex::new(result_rx),
            started_at: Instant::now(),
        };
    }

    pub fn confirm_analysis(&mut self, quality: AudioQuality, strength: f64) {
        self.state = AnalysisState::Confirmed { quality, strength };
    }

    pub fn reject_analysis(&mut self, quality: AudioQuality, strength: f64) {
        self.state = AnalysisState::Rejected { quality, strength };
    }

    pub fn error_analysis(&mut self, error: String) {
        self.state = AnalysisState::Error { error };
    }

    pub fn is_error(&self) -> bool {
        matches!(self.state, AnalysisState::Error { .. })
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            AnalysisState::Confirmed { .. }
                | AnalysisState::Rejected { .. }
                | AnalysisState::Error { .. }
        )
    }

    pub fn try_receive_results(&mut self) -> Option<AnalysisResults> {
        if let AnalysisState::InProgress { result_rx, .. } = &mut self.state {
            match result_rx.get_mut() {
                Ok(rx) => rx.try_recv().ok(),
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

impl Default for AnalysisStateComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalDiscoveryComponent {
    pub fn new(window_id: WindowId) -> Self {
        Self {
            discovered_at: Instant::now(),
            window_id,
        }
    }

    pub fn discovered_at(&self) -> Instant {
        self.discovered_at
    }

    pub fn window_id(&self) -> &WindowId {
        &self.window_id
    }

    /// Get how long ago this signal was discovered
    pub fn discovered_ago(&self) -> Duration {
        self.discovered_at.elapsed()
    }
}

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SignalHistoryComponent {
    last_heard: Option<Instant>,
    play_count: usize,
    total_play_duration: Duration,
    current_play_start: Option<Instant>,
}

use crate::ecs::components::audio::AudioId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    NotPlaying,
    Playing,
    Completed,
}

#[derive(Debug, Clone)]
pub struct SignalPlaybackComponent {
    state: PlaybackState,
    state_changed_at: Instant,
    audio_id: Option<AudioId>,
}

#[derive(Debug, Clone)]
pub struct SelectionComponent {
    selected_at: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct SignalCoordinationComponent {
    audio_request_enqueued: bool,
    rejection_reason: Option<String>,
    station_entity_id: Option<crate::ecs::StationId>,
}

impl SignalCoordinationComponent {
    pub fn new() -> Self {
        Self {
            audio_request_enqueued: false,
            rejection_reason: None,
            station_entity_id: None,
        }
    }

    pub fn audio_request_enqueued(&self) -> bool {
        self.audio_request_enqueued
    }

    pub fn set_audio_request_enqueued(&mut self, enqueued: bool) {
        self.audio_request_enqueued = enqueued;
    }

    pub fn rejection_reason(&self) -> Option<String> {
        self.rejection_reason.clone()
    }

    pub fn set_rejection_reason(&mut self, reason: Option<String>) {
        self.rejection_reason = reason;
    }

    pub fn station_entity_id(&self) -> Option<crate::ecs::StationId> {
        self.station_entity_id
    }

    pub fn set_station_entity_id(&mut self, id: Option<crate::ecs::StationId>) {
        self.station_entity_id = id;
    }
}

impl Default for SignalCoordinationComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionComponent {
    pub fn new() -> Self {
        Self { selected_at: None }
    }

    pub fn is_selected(&self) -> bool {
        self.selected_at.is_some()
    }

    pub fn select(&mut self) {
        self.selected_at = Some(Instant::now());
    }

    pub fn deselect(&mut self) {
        self.selected_at = None;
    }

    pub fn selected_at(&self) -> Option<Instant> {
        self.selected_at
    }
}

impl Default for SelectionComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalPlaybackComponent {
    pub fn new() -> Self {
        Self {
            state: PlaybackState::NotPlaying,
            state_changed_at: Instant::now(),
            audio_id: None,
        }
    }

    pub fn state(&self) -> PlaybackState {
        self.state
    }

    pub fn transition_to(&mut self, state: PlaybackState) {
        self.state = state;
        self.state_changed_at = Instant::now();
    }

    pub fn audio_id(&self) -> Option<AudioId> {
        self.audio_id
    }

    pub fn set_audio_id(&mut self, audio_id: Option<AudioId>) {
        self.audio_id = audio_id;
    }

    pub fn state_changed_at(&self) -> Instant {
        self.state_changed_at
    }

    pub fn is_playing(&self) -> bool {
        matches!(self.state, PlaybackState::Playing)
    }
}

impl Default for SignalPlaybackComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalHistoryComponent {
    pub fn new() -> Self {
        Self {
            last_heard: None,
            play_count: 0,
            total_play_duration: Duration::ZERO,
            current_play_start: None,
        }
    }

    pub fn start_play_session(&mut self) {
        self.current_play_start = Some(Instant::now());
    }

    pub fn end_play_session(&mut self) {
        if let Some(start) = self.current_play_start.take() {
            let duration = start.elapsed();
            self.play_count += 1;
            self.total_play_duration += duration;
            self.last_heard = Some(Instant::now());
        }
    }

    pub fn play_count(&self) -> usize {
        self.play_count
    }

    pub fn total_play_duration(&self) -> Duration {
        self.total_play_duration
    }

    pub fn last_heard(&self) -> Option<Instant> {
        self.last_heard
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.current_play_start.is_some()
    }

    /// Get current play duration (if playing)
    pub fn current_play_duration(&self) -> Option<Duration> {
        self.current_play_start.map(|start| start.elapsed())
    }

    /// Update last heard time without starting playback
    pub fn update_last_heard(&mut self) {
        self.last_heard = Some(Instant::now());
    }
}

impl Default for SignalHistoryComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalInfoComponent {
    pub fn new(frequency_hz: f64) -> Self {
        Self {
            frequency_hz,
            signal_strength: None,
            audio_quality: None,
            name: None,
        }
    }

    pub fn frequency(&self) -> f64 {
        self.frequency_hz
    }

    pub fn signal_strength(&self) -> Option<f64> {
        self.signal_strength
    }

    pub fn set_signal_strength(&mut self, strength: Option<f64>) {
        self.signal_strength = strength;
    }

    pub fn audio_quality(&self) -> Option<AudioQuality> {
        self.audio_quality
    }

    pub fn set_audio_quality(&mut self, quality: Option<AudioQuality>) {
        self.audio_quality = quality;
    }

    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::TaskId;

    fn create_test_window_id() -> WindowId {
        WindowId::new(TaskId::new("test-scan"), 5)
    }

    #[test]
    fn test_signal_id_creation() {
        let window_id = create_test_window_id();
        let signal_id = SignalId::new(88.9e6, window_id);

        assert!(signal_id.as_str().contains("88.9"));
        assert!(signal_id.as_str().contains("test-scan"));
        assert!(signal_id.as_str().contains("5"));
    }

    #[test]
    fn test_signal_id_format() {
        let window_id = create_test_window_id();
        let signal_id = SignalId::new(88.9e6, window_id);

        // Format: "{frequency_mhz}-{task_id}-{window_index}"
        let expected = "88.9-test-scan-5";
        assert_eq!(signal_id.as_str(), expected);
    }

    #[test]
    fn test_signal_id_display() {
        let window_id = create_test_window_id();
        let signal_id = SignalId::new(88.9e6, window_id);

        assert_eq!(format!("{}", signal_id), "88.9-test-scan-5");
    }

    #[test]
    fn test_signal_info_creation() {
        let info = SignalInfoComponent::new(88.9e6);

        assert_eq!(info.frequency(), 88.9e6);
        assert_eq!(info.signal_strength(), None);
        assert_eq!(info.audio_quality(), None);
        assert_eq!(info.name(), None);
    }

    #[test]
    fn test_set_signal_strength() {
        let mut info = SignalInfoComponent::new(88.9e6);

        info.set_signal_strength(Some(0.85));

        assert_eq!(info.signal_strength(), Some(0.85));
    }

    #[test]
    fn test_set_audio_quality() {
        let mut info = SignalInfoComponent::new(88.9e6);

        info.set_audio_quality(Some(AudioQuality::Good));

        assert_eq!(info.audio_quality(), Some(AudioQuality::Good));
    }

    #[test]
    fn test_set_name() {
        let mut info = SignalInfoComponent::new(88.9e6);

        info.set_name(Some("KQED".to_string()));

        assert_eq!(info.name(), Some("KQED".to_string()));
    }

    #[test]
    fn test_signal_discovery_creation() {
        let window_id = create_test_window_id();
        let discovery = SignalDiscoveryComponent::new(window_id.clone());

        assert_eq!(discovery.window_id(), &window_id);
        assert!(discovery.discovered_at() <= Instant::now());
    }

    #[test]
    fn test_discovery_time_is_recent() {
        let window_id = create_test_window_id();
        let before = Instant::now();
        let discovery = SignalDiscoveryComponent::new(window_id);
        let after = Instant::now();

        assert!(discovery.discovered_at() >= before);
        assert!(discovery.discovered_at() <= after);
    }

    #[test]
    fn test_discovered_ago() {
        let window_id = create_test_window_id();
        let discovery = SignalDiscoveryComponent::new(window_id);

        std::thread::sleep(Duration::from_millis(10));
        let elapsed = discovery.discovered_ago();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_analysis_state_creation() {
        let analysis = AnalysisStateComponent::new();

        assert!(analysis.is_not_started());
        assert!(!analysis.is_in_progress());
        assert!(!analysis.is_confirmed());
        assert!(!analysis.is_rejected());
    }

    #[test]
    fn test_analysis_starts() {
        let mut analysis = AnalysisStateComponent::new();
        assert!(analysis.is_not_started());

        let (handle, rx) = create_test_analysis_thread();
        analysis.start_analysis(handle, rx);

        assert!(analysis.is_in_progress());
        assert!(!analysis.is_not_started());
    }

    #[test]
    fn test_analysis_confirms_with_audio() {
        let mut analysis = AnalysisStateComponent::new();
        let (handle, rx) = create_test_analysis_thread();
        analysis.start_analysis(handle, rx);

        analysis.confirm_analysis(AudioQuality::Good, 0.85);

        assert!(analysis.is_confirmed());
        assert!(!analysis.is_in_progress());
        match analysis.state() {
            AnalysisState::Confirmed { quality, strength } => {
                assert_eq!(*quality, AudioQuality::Good);
                assert_eq!(*strength, 0.85);
            }
            _ => panic!("Expected Confirmed state"),
        }
    }

    #[test]
    fn test_analysis_rejects_without_audio() {
        let mut analysis = AnalysisStateComponent::new();
        let (handle, rx) = create_test_analysis_thread();
        analysis.start_analysis(handle, rx);

        analysis.reject_analysis(AudioQuality::NoAudio, 0.15);

        assert!(analysis.is_rejected());
        assert!(!analysis.is_in_progress());
        match analysis.state() {
            AnalysisState::Rejected { quality, strength } => {
                assert_eq!(*quality, AudioQuality::NoAudio);
                assert_eq!(*strength, 0.15);
            }
            _ => panic!("Expected Rejected state"),
        }
    }

    #[test]
    fn test_analysis_errors() {
        let mut analysis = AnalysisStateComponent::new();
        let (handle, rx) = create_test_analysis_thread();
        analysis.start_analysis(handle, rx);

        analysis.error_analysis("Thread panicked".to_string());

        assert!(analysis.is_error());
        assert!(!analysis.is_in_progress());
        match analysis.state() {
            AnalysisState::Error { error } => {
                assert_eq!(error, "Thread panicked");
            }
            _ => panic!("Expected Error state"),
        }
    }

    #[test]
    fn test_history_creation() {
        let history = SignalHistoryComponent::new();

        assert_eq!(history.play_count(), 0);
        assert_eq!(history.total_play_duration(), Duration::ZERO);
        assert_eq!(history.last_heard(), None);
    }

    #[test]
    fn test_play_session_increments_count() {
        let mut history = SignalHistoryComponent::new();

        history.start_play_session();
        history.end_play_session();

        assert_eq!(history.play_count(), 1);
    }

    #[test]
    fn test_play_session_without_start_does_nothing() {
        let mut history = SignalHistoryComponent::new();

        history.end_play_session();

        assert_eq!(history.play_count(), 0);
        assert_eq!(history.total_play_duration(), Duration::ZERO);
    }

    #[test]
    fn test_total_play_duration_accumulates() {
        let mut history = SignalHistoryComponent::new();

        history.start_play_session();
        std::thread::sleep(Duration::from_millis(50));
        history.end_play_session();

        let duration = history.total_play_duration();
        assert!(duration >= Duration::from_millis(50));
        assert!(duration < Duration::from_millis(100));
    }

    #[test]
    fn test_last_heard_updated() {
        let mut history = SignalHistoryComponent::new();
        assert_eq!(history.last_heard(), None);

        history.start_play_session();
        history.end_play_session();

        assert!(history.last_heard().is_some());
    }

    #[test]
    fn test_multiple_play_sessions() {
        let mut history = SignalHistoryComponent::new();

        history.start_play_session();
        std::thread::sleep(Duration::from_millis(30));
        history.end_play_session();

        history.start_play_session();
        std::thread::sleep(Duration::from_millis(30));
        history.end_play_session();

        assert_eq!(history.play_count(), 2);
        assert!(history.total_play_duration() >= Duration::from_millis(60));
        assert!(history.last_heard().is_some());
    }

    #[test]
    fn test_is_playing_state() {
        let mut history = SignalHistoryComponent::new();
        assert!(!history.is_playing());

        history.start_play_session();
        assert!(history.is_playing());

        history.end_play_session();
        assert!(!history.is_playing());
    }

    #[test]
    fn test_current_play_duration() {
        let mut history = SignalHistoryComponent::new();
        assert_eq!(history.current_play_duration(), None);

        history.start_play_session();
        std::thread::sleep(Duration::from_millis(10));

        let duration = history.current_play_duration();
        assert!(duration.is_some());
        assert!(duration.unwrap() >= Duration::from_millis(10));

        history.end_play_session();
        assert_eq!(history.current_play_duration(), None);
    }

    #[test]
    fn test_update_last_heard() {
        let mut history = SignalHistoryComponent::new();
        assert_eq!(history.last_heard(), None);

        history.update_last_heard();
        assert!(history.last_heard().is_some());
        assert_eq!(history.play_count(), 0);
        assert!(!history.is_playing());
    }

    #[test]
    fn test_playback_creation() {
        let playback = SignalPlaybackComponent::new();

        assert_eq!(playback.state(), PlaybackState::NotPlaying);
        assert_eq!(playback.audio_id(), None);
    }

    #[test]
    fn test_transition_to_playing() {
        let mut playback = SignalPlaybackComponent::new();

        playback.transition_to(PlaybackState::Playing);

        assert_eq!(playback.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_transition_to_completed() {
        let mut playback = SignalPlaybackComponent::new();
        playback.transition_to(PlaybackState::Playing);

        playback.transition_to(PlaybackState::Completed);

        assert_eq!(playback.state(), PlaybackState::Completed);
    }

    #[test]
    fn test_set_audio_id() {
        let mut playback = SignalPlaybackComponent::new();
        let audio_id = AudioId::new();

        playback.set_audio_id(Some(audio_id));

        assert_eq!(playback.audio_id(), Some(audio_id));
    }

    #[test]
    fn test_state_changed_at_updates() {
        let mut playback = SignalPlaybackComponent::new();
        let initial_time = playback.state_changed_at();

        std::thread::sleep(Duration::from_millis(10));
        playback.transition_to(PlaybackState::Playing);

        assert!(playback.state_changed_at() > initial_time);
    }

    #[test]
    fn test_selection_creation() {
        let selection = SelectionComponent::new();

        assert!(!selection.is_selected());
        assert_eq!(selection.selected_at(), None);
    }

    #[test]
    fn test_select() {
        let mut selection = SelectionComponent::new();

        selection.select();

        assert!(selection.is_selected());
        assert!(selection.selected_at().is_some());
    }

    #[test]
    fn test_deselect() {
        let mut selection = SelectionComponent::new();
        selection.select();

        selection.deselect();

        assert!(!selection.is_selected());
        assert_eq!(selection.selected_at(), None);
    }

    #[test]
    fn test_selected_at_timestamp() {
        let mut selection = SelectionComponent::new();
        let before = Instant::now();

        selection.select();

        let after = Instant::now();
        let selected_at = selection.selected_at().unwrap();
        assert!(selected_at >= before);
        assert!(selected_at <= after);
    }

    #[test]
    fn test_coordination_creation() {
        let coordination = SignalCoordinationComponent::new();

        assert!(!coordination.audio_request_enqueued());
        assert_eq!(coordination.rejection_reason(), None);
    }

    #[test]
    fn test_set_audio_request_enqueued() {
        let mut coordination = SignalCoordinationComponent::new();

        coordination.set_audio_request_enqueued(true);

        assert!(coordination.audio_request_enqueued());
    }

    #[test]
    fn test_clear_audio_request_enqueued() {
        let mut coordination = SignalCoordinationComponent::new();
        coordination.set_audio_request_enqueued(true);

        coordination.set_audio_request_enqueued(false);

        assert!(!coordination.audio_request_enqueued());
    }

    #[test]
    fn test_set_rejection_reason() {
        let mut coordination = SignalCoordinationComponent::new();
        let reason = "Below squelch threshold".to_string();

        coordination.set_rejection_reason(Some(reason.clone()));

        assert_eq!(coordination.rejection_reason(), Some(reason));
    }

    #[test]
    fn test_clear_rejection_reason() {
        let mut coordination = SignalCoordinationComponent::new();
        coordination.set_rejection_reason(Some("Failed".to_string()));

        coordination.set_rejection_reason(None);

        assert_eq!(coordination.rejection_reason(), None);
    }

    fn create_test_analysis_thread() -> (
        JoinHandle<ScannerResult<AnalysisResults>>,
        mpsc::Receiver<AnalysisResults>,
    ) {
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
