//! TUI module using The Elm Architecture pattern

use std::{
    collections::HashMap,
    io,
    sync::mpsc,
    time::{Duration, Instant},
};

use chrono::Utc;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use indexmap::IndexMap;
use ratatui::{Frame, Terminal, backend::CrosstermBackend};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{
    core::signals::ModulationType,
    ecs::{
        AudioEntity, Entities, Entity, SignalEntity, TaskComponents, TaskEntity,
        components::scan::PreviousPauseState, resources::LocationResource,
    },
    persistence::{
        location::{DEFAULT_LOCATION, Location, LocationDetector, UserSettings},
        storage::SignalStorage,
        types::PersistedSignal,
    },
    ui::TuiEvent,
};

pub mod colors;
pub mod layout;
pub mod model;
pub mod renderers;
pub mod themes;
pub mod widgets;

#[cfg(test)]
pub mod integration_tests;

use layout::Layout;
use model::{FocusState, Model};
use renderers::{
    activities::render_activities, console::ConsoleRenderer, header, instructions, signals_table,
    task_progress, tuners,
};
use themes::{Theme, ThemeName, create_theme};

/// TUI-based progress display for multiple signals using The Elm Architecture
pub struct TuiProgressDisplay {
    receiver: mpsc::Receiver<TuiEvent>,
    model: Model,
    _last_update: Instant,
    shutdown_token: CancellationToken,
    theme: Box<dyn Theme>,
    current_theme: ThemeName,
    ui_update_system: Option<crate::ecs::systems::UIUpdateSystem>,
    task_entities: Option<Entities<TaskEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    signal_entities: Option<Entities<SignalEntity>>,
    pause_request_queue: Option<crate::ecs::Resource<crate::ecs::PauseRequestQueue>>,
    location_resource: Option<LocationResource>,
    cached_locality: Option<String>,
    last_audio_gen: u64,
    last_scan_gen: u64,
    signal_storage: SignalStorage,
}

impl TuiProgressDisplay {
    /// Get the proper signals storage path
    /// Uses project-relative path that works regardless of working directory
    fn get_signals_storage_path() -> std::path::PathBuf {
        // Try to find the project root by looking for Cargo.toml
        let mut current_dir =
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

        // Walk up the directory tree to find Cargo.toml (project root)
        loop {
            if current_dir.join("Cargo.toml").exists() {
                return current_dir.join("data").join("signals");
            }
            if let Some(parent) = current_dir.parent() {
                current_dir = parent.to_path_buf();
            } else {
                break;
            }
        }

        // Fallback to relative path if we can't find project root
        std::path::PathBuf::from("data").join("signals")
    }

    /// Create new TUI progress display with default theme
    pub fn new(receiver: mpsc::Receiver<TuiEvent>, shutdown_token: CancellationToken) -> Self {
        let current_theme = ThemeName::CaladanDark;
        let theme = create_theme(&current_theme);
        Self {
            receiver,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
            ui_update_system: None,
            task_entities: None,
            audio_entities: None,
            signal_entities: None,
            pause_request_queue: None,
            location_resource: None,
            cached_locality: None,
            last_audio_gen: 0,
            last_scan_gen: 0,
            signal_storage: SignalStorage::new(Self::get_signals_storage_path()),
        }
    }

    /// Create new TUI progress display with specified theme
    pub fn new_with_theme(
        receiver: mpsc::Receiver<TuiEvent>,
        shutdown_token: CancellationToken,
        theme: Box<dyn Theme>,
        current_theme: ThemeName,
    ) -> Self {
        Self {
            receiver,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
            ui_update_system: None,
            task_entities: None,
            audio_entities: None,
            signal_entities: None,
            pause_request_queue: None,
            location_resource: None,
            cached_locality: None,
            last_audio_gen: 0,
            last_scan_gen: 0,
            signal_storage: SignalStorage::new(Self::get_signals_storage_path()),
        }
    }

    /// Set entity worlds for spectrum display integration
    pub fn with_entities(
        mut self,
        task_entities: Entities<TaskEntity>,
        audio_entities: Entities<AudioEntity>,
        signal_entities: Entities<SignalEntity>,
    ) -> Self {
        self.task_entities = Some(task_entities);
        self.audio_entities = Some(audio_entities);
        self.signal_entities = Some(signal_entities);
        self.ui_update_system = Some(crate::ecs::systems::UIUpdateSystem::new());
        self
    }

    pub fn with_pause_request_queue(
        mut self,
        queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
    ) -> Self {
        self.pause_request_queue = Some(queue);
        self
    }

    pub fn with_global_pause_resource(mut self, resource: crate::ecs::GlobalPauseResource) -> Self {
        self.model
            .set_global_pause_resource(std::sync::Arc::clone(&resource));
        self
    }

    pub fn with_location_resource(mut self, resource: LocationResource) -> Self {
        self.location_resource = Some(resource);
        self
    }

    /// Load persistent signals from storage during TUI initialization
    /// Following Elm Architecture - this updates Model state during startup
    pub fn with_persistence(mut self) -> Self {
        // Try to get actual user location, fall back to San Francisco default
        let location = if let Some(ref location_resource) = self.location_resource {
            // Try to get location using LocationResource
            if let Ok(mut resource) = location_resource.try_lock() {
                if let Ok(detected_location) = resource.detect_current_location() {
                    debug!(
                        "Using detected location for signal storage: lat={}, lon={}",
                        detected_location.lat, detected_location.lon
                    );
                    crate::persistence::location::Location {
                        lat: detected_location.lat,
                        lon: detected_location.lon,
                    }
                } else {
                    debug!("Location detection failed, using San Francisco default");
                    DEFAULT_LOCATION
                }
            } else {
                debug!("LocationResource locked, using San Francisco default");
                DEFAULT_LOCATION
            }
        } else {
            debug!("No LocationResource available, using San Francisco default");
            DEFAULT_LOCATION
        };

        debug!(
            "Starting persistent signal loading with location: lat={}, lon={}",
            location.lat, location.lon
        );

        match self
            .model
            .load_persistent_signals_from_storage(&self.signal_storage, location)
        {
            Ok(()) => {
                debug!("Successfully loaded persistent signals from storage");
            }
            Err(e) => {
                // Log error but don't fail TUI startup
                debug!("Failed to load persistent signals: {}", e);
            }
        }

        self
    }

    /// Run the TUI display loop
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check if we're in an interactive terminal
        if !self.is_terminal_interactive() {
            return self.run_text_fallback();
        }

        self.run_full_tui()
    }

    /// Run the full TUI with ratatui
    fn run_full_tui(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Setup terminal with error handling
        if let Err(_e) = enable_raw_mode() {
            return self.run_simple_tui();
        }

        let mut stdout = io::stdout();

        // Use alternate screen mode for full TUI
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(_) => {
                let _ = disable_raw_mode(); // Try to restore
                return self.run_simple_tui();
            }
        };

        let result = self.run_app(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(result?)
    }

    /// Check if the application should quit
    fn should_quit(&self) -> bool {
        self.model.should_quit || self.shutdown_token.is_cancelled()
    }

    /// Check if the application should auto-exit due to inactivity
    fn should_auto_exit(&self, iterations: u32) -> bool {
        iterations > 2000 && self.model.is_empty()
    }

    /// Handle quit keys (Ctrl-C, Ctrl-D, 'q')
    fn handle_quit_keys(&mut self, key: &event::KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                self.model.quit();
                true
            }
            KeyCode::Char('d') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                self.model.quit();
                true
            }
            KeyCode::Char('q') if !self.model.theme_selector_open => {
                self.model.quit();
                true
            }
            _ => false,
        }
    }

    /// Handle theme selector navigation
    fn handle_theme_selector(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                let all_themes = themes::ThemeName::all();
                self.model.theme_selector_prev(all_themes.len());
                self.current_theme = all_themes[self.model.theme_selector_index].clone();
                self.theme = create_theme(&self.current_theme);
            }
            KeyCode::Down => {
                let all_themes = themes::ThemeName::all();
                self.model.theme_selector_next(all_themes.len());
                self.current_theme = all_themes[self.model.theme_selector_index].clone();
                self.theme = create_theme(&self.current_theme);
            }
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => {
                self.model.close_theme_selector();
            }
            _ => {}
        }
    }

    /// Check if awaiting tune should transition to listening
    fn check_awaiting_tune_transition(&mut self) {
        if !matches!(self.model.ui_mode, model::UiMode::AwaitingTune { .. }) {
            return;
        }

        if let Some(ref task_entities) = self.task_entities
            && let Ok(entities) = task_entities.try_read()
            && let Some(task) = entities.iter().next()
            && let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components
            && progress.is_paused()
            && let model::UiMode::AwaitingTune {
                signal_index,
                tuning_signal_id: _,
                window_id,
            } = &self.model.ui_mode
            && let Some(info) = self.model.selected_signal_info()
        {
            // Wait for audio entity to actually start playing before transitioning
            let audio_is_playing = if let Some(ref audio_entities) = self.audio_entities
                && let Ok(audios) = audio_entities.try_read()
            {
                audios.iter().any(|audio| {
                    (audio.frequency() - info.signal_frequency).abs() < 1000.0 && audio.is_playing()
                })
            } else {
                false
            };

            if audio_is_playing {
                self.model.ui_mode = model::UiMode::Listening {
                    signal_index: *signal_index,
                    window_id: *window_id,
                    playing_signal_id: info.signal_id.clone(),
                };
                debug!(
                    signal_id = ?info.signal_id,
                    frequency_mhz = info.signal_frequency / 1e6,
                    "TUI: Transitioned from AwaitingTune to Listening (audio confirmed playing)"
                );
            }
        }
    }

    /// Update spectrum display data from entity worlds
    fn update_spectrum_from_entities(&mut self) {
        if let Some(system) = &mut self.ui_update_system
            && let (Some(task_entities), Some(audio_entities), Some(signal_entities)) = (
                &self.task_entities,
                &self.audio_entities,
                &self.signal_entities,
            )
        {
            let (task_gen, audio_gen) = match (task_entities.try_read(), audio_entities.try_read())
            {
                (Ok(tasks), Ok(audios)) => (tasks.generation(), audios.generation()),
                _ => {
                    return;
                }
            };

            if task_gen != self.last_scan_gen || audio_gen != self.last_audio_gen {
                use crate::ecs::{System, SystemContext};

                debug!(
                    task_gen = task_gen,
                    audio_gen = audio_gen,
                    last_scan_gen = self.last_scan_gen,
                    last_audio_gen = self.last_audio_gen,
                    "TUI: Running UIUpdateSystem due to generation change"
                );

                let mut context = SystemContext::new()
                    .with_task_entities(std::sync::Arc::clone(task_entities))
                    .with_audio_entities(std::sync::Arc::clone(audio_entities))
                    .with_signal_entities(std::sync::Arc::clone(signal_entities));

                if system.run(&mut context).is_ok() {
                    let spectrum_count = system.stations().len();
                    let active_count = system.stations().iter().filter(|s| s.is_active).count();
                    let task_count = system.tasks().len();

                    self.model.spectrum_stations = system.stations().to_vec();
                    self.model.active_audio_frequency = system.active_frequency();
                    self.model.active_tuner_id = system.active_tuner_id().cloned();
                    self.model.tasks = system.tasks().to_vec();

                    // Initialize displayed_task_id to first task if none selected
                    if self.model.displayed_task_id.is_none() && !self.model.tasks.is_empty() {
                        self.model.displayed_task_id = Some(self.model.tasks[0].task_id.clone());
                    }

                    // Clear displayed_task_id if task no longer exists
                    if let Some(ref task_id) = self.model.displayed_task_id
                        && !self.model.tasks.iter().any(|t| &t.task_id == task_id)
                    {
                        self.model.displayed_task_id = None;
                    }

                    let signals_by_window = system.signals_by_window().clone();
                    self.sync_signals_to_model(&signals_by_window);

                    debug!(
                        spectrum_count = spectrum_count,
                        active_count = active_count,
                        signal_count = signals_by_window.values().map(|v| v.len()).sum::<usize>(),
                        task_count = task_count,
                        "TUI: Updated model from entities"
                    );
                }

                self.last_scan_gen = task_gen;
                self.last_audio_gen = audio_gen;
            }
        }
    }

    /// Sync signal data from UIUpdateSystem to Model
    fn sync_signals_to_model(
        &mut self,
        signals_by_window: &IndexMap<
            crate::ecs::components::window::WindowId,
            Vec<crate::ecs::systems::ui::SignalData>,
        >,
    ) {
        use std::time::Instant;

        use crate::ui::tui::model::types::{
            AnalysisStatus, PlaybackState, SignalProgress, WindowProgress,
        };

        // Collect signals that need auto-save to process after all window updates
        let mut signals_to_auto_save: Vec<SignalProgress> = Vec::new();

        for (window_id, signal_data_list) in signals_by_window {
            let window_index = window_id.window_index;
            let window = self
                .model
                .windows
                .entry(window_index)
                .or_insert_with(|| WindowProgress {
                    window_id: window_index,
                    signals: Vec::new(),
                    is_complete: false,
                    signal_lookup: HashMap::new(),
                });

            for signal_data in signal_data_list {
                // Map ECS AnalysisStatus to TUI AnalysisStatus (keeping analysis and playback
                // separate)
                use crate::ecs::components::{
                    AnalysisStatus as EcsStatus, signal::PlaybackState as EcsPlaybackState,
                };

                let (status, audio_quality, signal_strength) = match signal_data.status {
                    EcsStatus::Detected => (AnalysisStatus::Detected, None, None),
                    EcsStatus::Analyzing => (AnalysisStatus::Analyzing, None, None),
                    EcsStatus::Signal { quality, strength } => {
                        (AnalysisStatus::Signal, Some(quality), Some(strength))
                    }
                    EcsStatus::Rejected { quality, strength } => {
                        (AnalysisStatus::Rejected, Some(quality), Some(strength))
                    }
                    EcsStatus::Error => (AnalysisStatus::Error, None, None),
                };

                let playback_state = match signal_data.playback_state {
                    EcsPlaybackState::NotPlaying => PlaybackState::NotPlaying,
                    EcsPlaybackState::Playing => PlaybackState::Playing,
                    EcsPlaybackState::Completed => PlaybackState::Completed,
                };

                let signal_progress = SignalProgress {
                    signal_id: signal_data.signal_id.clone(),
                    frequency_hz: signal_data.frequency_hz,
                    window_id: window_index,
                    center_frequency_hz: signal_data.frequency_hz,
                    completion: signal_data.completion,
                    status: status.clone(),
                    playback_state,
                    audio_quality,
                    signal_strength,
                    last_update: Instant::now(),
                    notes: None,
                };

                // Check if this signal needs auto-save
                let should_auto_save =
                    if let Some(&index) = window.signal_lookup.get(&signal_data.signal_id) {
                        // Signal exists - check if status changed to confirmed for auto-save
                        let previous_status = window.signals[index].status.clone();
                        let needs_save = previous_status != AnalysisStatus::Signal
                            && status == AnalysisStatus::Signal;
                        window.signals[index] = signal_progress.clone();
                        needs_save
                    } else {
                        // New signal - auto-save immediately if already confirmed
                        let needs_save = status == AnalysisStatus::Signal;
                        let index = window.signals.len();
                        window
                            .signal_lookup
                            .insert(signal_data.signal_id.clone(), index);
                        window.signals.push(signal_progress.clone());
                        needs_save
                    };

                if should_auto_save {
                    signals_to_auto_save.push(signal_progress);
                }
            }
        }

        // Process auto-save candidates after all window updates are complete
        for signal_progress in signals_to_auto_save {
            if let Err(e) = self.auto_save_confirmed_signal(&signal_progress) {
                tracing::warn!(
                    "Auto-save failed for signal {}: {}",
                    signal_progress.signal_id,
                    e
                );
            }
        }
    }

    /// Handle navigation keys (arrows)
    fn handle_navigation_keys(&mut self, key: KeyCode) {
        let tuner_count = self.model.tuners.len();

        match key {
            KeyCode::Up => self.model.navigate_up(),
            KeyCode::Down => self.model.navigate_down(),
            KeyCode::Left => self.model.navigate_left(tuner_count),
            KeyCode::Right => {
                self.model.navigate_right(tuner_count);
            }
            _ => {}
        }
    }

    fn handle_enter_browsing_mode(&mut self, _selected_index: usize) -> bool {
        // Get current flat index and window_id from the focused row
        let rows = self.model.build_signal_rows();
        if let FocusState::ScanProgress(flat_idx) = self.model.focus_state {
            if let Some(row) = rows.get(flat_idx) {
                let window_id = row.window_id;
                if let Some(info) = self.model.selected_signal_info() {
                    debug!(
                        flat_idx,
                        window_id,
                        signal_id = %info.signal_id,
                        frequency_mhz = info.signal_frequency / 1e6,
                        "TUI: Transitioning to AwaitingTune mode"
                    );
                    self.model.ui_mode = model::UiMode::AwaitingTune {
                        signal_index: flat_idx,
                        window_id,
                        tuning_signal_id: info.signal_id.clone(),
                    };
                } else {
                    debug!("TUI: selected_signal_info returned None");
                    return false;
                }
            } else {
                debug!(
                    flat_idx,
                    available_rows = rows.len(),
                    "TUI: focus_state index out of bounds"
                );
                return false;
            }
        } else {
            debug!(
                focus_state = ?self.model.focus_state,
                "TUI: focus_state is not ScanProgress"
            );
            return false;
        }

        // TODO: Re-implement pause request for hierarchical tasks
        // PauseAndTuneRequest expects ScanId but hierarchical tasks use TaskId
        let window_num = self.model.current_window;
        if let Some(info) = self.model.selected_signal_info() {
            debug!(
                window_num = window_num,
                station_frequency_mhz = info.signal_frequency / 1e6,
                "TUI: Pause request with station (pending implementation)"
            );
        } else {
            debug!(
                window_num = window_num,
                "TUI: Pause request (pending implementation)"
            );
        }

        true
    }

    fn handle_switch_station(
        &mut self,
        _selected_index: usize,
        info: model::SelectedSignalInfo,
    ) -> bool {
        debug!(
            signal_id = ?info.signal_id,
            window_id = info.window_id,
            signal_frequency_mhz = info.signal_frequency / 1e6,
            "TUI: Switching to different station"
        );

        // Get flat index and window_id from current UiMode
        if let Some(signal_index) = self.model.selected_signal_index() {
            let window_id = info.window_id;
            self.model.ui_mode = model::UiMode::AwaitingTune {
                signal_index,
                window_id,
                tuning_signal_id: info.signal_id.clone(),
            };
        } else {
            return false;
        }

        // TODO: Re-implement pause request for hierarchical tasks
        // PauseAndTuneRequest expects ScanId but hierarchical tasks use TaskId
        let window_num = self.model.current_window;
        debug!(
            window_num = window_num,
            station_frequency_mhz = info.signal_frequency / 1e6,
            "TUI: Pause request with new station (pending implementation)"
        );

        self.model.playback_active = true;
        true
    }

    fn handle_resume_scan(&mut self) -> bool {
        self.model.ui_mode = model::UiMode::Idle;

        // TODO: Re-implement resume request mechanism for hierarchical tasks
        // The resume_request field doesn't exist in TaskComponents::Scan yet
        let _window_num = self.model.current_window;
        debug!(
            window_num = _window_num,
            "TUI: Resume scan requested (mechanism pending implementation)"
        );

        // ECS Phase 5: Pure ECS - only set component, no commands
        if self.model.playback_active {
            if let Some(ref audio_entities) = self.audio_entities
                && let Ok(mut entities) = audio_entities.try_write()
                && let Some(audio) = entities.iter_mut().next()
            {
                audio.request_stop_listening();
                debug!(
                    audio_id = ?audio.id(),
                    "TUI: Set stop_listening_request on AudioEntity"
                );
            }

            self.model.playback_active = false;
        }

        true
    }

    /// Handle spacebar key for global pause/resume
    fn handle_spacebar_pause(&mut self) {
        if let Some(ref resource) = self.model.global_pause_resource {
            let resource = std::sync::Arc::clone(resource);
            if let Ok(mut state) = resource.lock() {
                match *state {
                    crate::ecs::GlobalPauseState::Active => {
                        debug!("TUI: Spacebar pressed - pausing globally");

                        let had_active_scans = self.has_active_scans();
                        let playing_stations = self.collect_playing_stations();

                        *state = crate::ecs::GlobalPauseState::Paused {
                            had_active_scans,
                            playing_stations,
                        };

                        drop(state);

                        self.pause_all_scans();
                        self.pause_all_audio();
                    }
                    crate::ecs::GlobalPauseState::Paused { .. } => {
                        debug!("TUI: Spacebar pressed - resuming from global pause");

                        *state = crate::ecs::GlobalPauseState::Active;

                        drop(state);

                        self.resume_all_scans();
                        self.resume_all_audio();
                    }
                }
            }
        }
    }

    /// Check if there are any active scans
    fn has_active_scans(&self) -> bool {
        if let Some(ref task_entities) = self.task_entities
            && let Ok(entities) = task_entities.try_read()
        {
            return entities.iter().any(|task| {
                let TaskComponents::Scan { progress, .. } = &task.components;
                progress.is_scanning()
            });
        }
        false
    }

    /// Collect information about currently playing signals
    fn collect_playing_stations(&self) -> Vec<crate::ecs::PlayingStationInfo> {
        let _playing_stations: Vec<crate::ecs::PlayingStationInfo> = Vec::new();

        // TODO: Update to work with SignalEntity instead of StationEntity
        // This method was collecting playing station information for global pause
        // Will need to be updated to work with signal entities

        Vec::new()
    }

    /// Pause all active scans globally
    fn pause_all_scans(&mut self) {
        if let Some(ref task_entities) = self.task_entities
            && let Ok(mut entities) = task_entities.try_write()
        {
            for task in entities.iter_mut() {
                let TaskComponents::Scan { progress, .. } = &mut task.components;
                let current_window = match &progress.current_window {
                    Some(w) => w.clone(),
                    None => continue,
                };

                let previous_state = if progress.is_scanning() {
                    PreviousPauseState::WasScanning
                } else if progress.is_listening() {
                    // TODO: Fix type mismatch - get_listening_station expects ScanId but we have
                    // TaskId For now, skip listening state preservation
                    continue;
                } else {
                    continue;
                };

                progress.pause_globally(current_window.clone(), previous_state);
                debug!(
                    task_id = ?task.id(),
                    window = ?current_window,
                    "TUI: Paused scan globally"
                );
            }
        }
    }

    /// Pause all active audio
    fn pause_all_audio(&mut self) {
        if let Some(ref audio_entities) = self.audio_entities
            && let Ok(mut entities) = audio_entities.try_write()
        {
            for audio in entities.iter_mut() {
                if audio.is_playing() {
                    audio.stop();
                    debug!(
                        audio_id = ?audio.id(),
                        frequency_hz = audio.frequency(),
                        "TUI: Stopped audio entity for global pause"
                    );
                }
            }
        }
    }

    /// Resume all globally paused scans
    fn resume_all_scans(&mut self) {
        if let Some(ref task_entities) = self.task_entities
            && let Ok(mut entities) = task_entities.try_write()
        {
            for task in entities.iter_mut() {
                let TaskComponents::Scan { progress, .. } = &mut task.components;
                if progress.is_globally_paused() {
                    progress.resume_from_global_pause();
                    debug!(
                        task_id = ?task.id(),
                        "TUI: Resumed scan from global pause"
                    );
                }
            }
        }
    }

    /// Resume all audio that was playing before global pause
    fn resume_all_audio(&mut self) {
        // TODO: Update to work with SignalEntity instead of StationEntity
        // This method was responsible for resuming audio playback after global pause
        // Will need to be updated to work with signal entities

        debug!("TUI: resume_all_audio temporarily disabled during StationEntity migration");
    }

    /// Handle tuning/playback actions (Enter key in various contexts)
    fn handle_tuning_actions(&mut self, key: &event::KeyEvent) -> bool {
        if key.code != KeyCode::Enter || self.model.theme_selector_open {
            return false;
        }

        // Case 1: Enter browsing mode from scan mode
        // Allow entering browsing mode when:
        // - Focus is on scan results
        // - Not already in browsing mode
        // - Have a signal selected
        // - Scan is paused/idle (including after scan completes)
        let has_scan_focus = matches!(self.model.focus_state, model::FocusState::ScanProgress(_));
        let not_browsing = !self.model.browsing_mode();
        let selected_idx = self.model.selected_signal_index();

        if !has_scan_focus || !not_browsing || selected_idx.is_none() {
            debug!(
                has_scan_focus,
                not_browsing,
                has_selected_index = selected_idx.is_some(),
                ui_mode = ?self.model.ui_mode,
                focus_state = ?self.model.focus_state,
                "TUI: ENTER key blocked - preconditions not met"
            );
        }

        if has_scan_focus
            && not_browsing
            && let Some(selected_index) = selected_idx
        {
            return self.handle_enter_browsing_mode(selected_index);
        }

        // Case 2: Switch station while listening
        if let model::UiMode::Listening {
            playing_signal_id, ..
        } = &self.model.ui_mode
            && !self.model.is_continue_scan_selected()
            && let Some(selected_index) = self.model.selected_signal_index()
            && let Some(info) = self.model.selected_signal_info()
            && &info.signal_id != playing_signal_id
        {
            return self.handle_switch_station(selected_index, info);
        }

        // Case 2b: Switch station while awaiting tune (allows canceling pending tune)
        if let model::UiMode::AwaitingTune {
            tuning_signal_id: _,
            signal_index: tuning_signal_index,
            ..
        } = &self.model.ui_mode
            && !self.model.is_continue_scan_selected()
            && let Some(selected_index) = self.model.selected_signal_index()
            && let Some(info) = self.model.selected_signal_info()
            && selected_index != *tuning_signal_index
        {
            return self.handle_switch_station(selected_index, info);
        }

        // Case 3: Resume scan from browsing mode
        if matches!(
            self.model.ui_mode,
            model::UiMode::Listening { .. } | model::UiMode::AwaitingTune { .. }
        ) && self.model.is_continue_scan_selected()
        {
            return self.handle_resume_scan();
        }

        false
    }

    /// Handle keyboard input for modal, returns true if handled
    fn handle_modal_input(&mut self, key: &event::KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                // Close modal with ESC
                self.model.handle_modal_escape_key(key);
                true
            }
            KeyCode::Char(_) => {
                // Handle text input for notes
                self.model.handle_modal_text_input(key);
                true
            }
            KeyCode::Backspace => {
                // Handle backspace in modal notes
                self.handle_modal_backspace();
                true
            }
            KeyCode::Enter => {
                // Save notes and close modal
                self.handle_modal_save_and_close();
                true
            }
            _ => false, // Let other keys pass through
        }
    }

    /// Handle backspace in modal notes input
    fn handle_modal_backspace(&mut self) {
        if let Some(modal) = &mut self.model.signal_detail_modal
            && !modal.notes_input.is_empty()
        {
            modal.notes_input.pop();
            modal.is_notes_dirty = true;
            self.model.mark_dirty();
        }
    }

    /// Save modal notes and close modal
    fn handle_modal_save_and_close(&mut self) {
        if let Some(modal) = &self.model.signal_detail_modal
            && modal.is_notes_dirty
        {
            let frequency_hz = modal.frequency_hz;
            let notes = modal.notes_input.clone();

            // Save to persistence layer using frequency-based lookup (more stable than SignalId)
            if let Err(e) = self.save_signal_notes_by_frequency(frequency_hz, &notes) {
                debug!(frequency_hz = frequency_hz, error = %e, "Failed to save modal notes to persistence layer");
            } else {
                debug!(
                    frequency_hz = frequency_hz,
                    notes_length = notes.len(),
                    "Modal notes saved for signal"
                );
            }
        }

        // Close the modal
        self.model.close_signal_detail_modal();
    }

    /// Handle keyboard input for notes editing, returns true if handled
    fn handle_notes_editing_keys(&mut self, key: &event::KeyEvent) -> bool {
        match key.code {
            KeyCode::Enter => {
                // Save notes
                if let Some((signal_id, notes)) = self.model.save_editing_notes() {
                    // Save to persistence layer
                    if let Err(e) = self.save_signal_notes(&signal_id, &notes) {
                        debug!(signal_id = %signal_id, error = %e, "Failed to save notes to persistence layer");
                    } else {
                        debug!(signal_id = %signal_id, notes_length = notes.len(), "Notes saved for signal");
                    }
                }
                true
            }
            KeyCode::Esc => {
                // Cancel editing
                self.model.cancel_editing_notes();
                true
            }
            KeyCode::Char(c) => {
                // Add character to input
                self.model.notes_input.handle_char(c);
                true
            }
            KeyCode::Backspace => {
                // Remove character
                self.model.notes_input.handle_backspace();
                true
            }
            KeyCode::Delete => {
                // Delete character at cursor
                self.model.notes_input.handle_delete();
                true
            }
            KeyCode::Left => {
                // Move cursor left
                self.model.notes_input.move_cursor_left();
                true
            }
            KeyCode::Right => {
                // Move cursor right
                self.model.notes_input.move_cursor_right();
                true
            }
            KeyCode::Home => {
                // Move cursor to start
                self.model.notes_input.move_cursor_home();
                true
            }
            KeyCode::End => {
                // Move cursor to end
                self.model.notes_input.move_cursor_end();
                true
            }
            _ => false,
        }
    }

    /// Save signal notes to persistence layer
    fn save_signal_notes(
        &mut self,
        signal_id: &crate::ecs::components::SignalId,
        notes: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Try to find signal in scan windows first, then persistent signals
        let (frequency_hz, signal_strength) =
            if let Some(signal_progress) = self.find_signal_by_id(signal_id) {
                // Found in scan windows
                (
                    signal_progress.frequency_hz,
                    signal_progress.signal_strength.unwrap_or(0.5),
                )
            } else {
                // Try to find in persistent signals using the Model's unified approach
                self.find_signal_info_from_persistent(signal_id)
                    .ok_or_else(|| {
                        format!(
                            "Signal not found in scan windows or persistent signals: {}",
                            signal_id
                        )
                    })?
            };

        // Get current location (with fallback for now)
        let fallback_location = Location {
            lat: 37.7749, // San Francisco default
            lon: -122.4194,
        };
        let location = LocationDetector::current_location(Some(fallback_location))?;

        // Create persisted signal with current data
        let persisted_signal = PersistedSignal {
            frequency_hz,
            signal_strength,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 1,
            modulation: ModulationType::WFM,
            notes: if notes.is_empty() {
                None
            } else {
                Some(notes.to_string())
            },
        };

        // Save signal to storage
        self.signal_storage
            .save_signal(&persisted_signal, location)?;

        // Also save user settings with updated location
        let user_settings = crate::persistence::location::UserSettings {
            version: "v1.0".to_string(),
            last_known_location: Some(crate::persistence::location::CachedLocation {
                lat: location.lat,
                lon: location.lon,
                timestamp: Utc::now(),
            }),
            preferences: crate::persistence::location::UserPreferences {
                auto_save_interval_seconds: 30,
            },
        };

        // Save user settings to ~/.scanner/settings.json
        crate::persistence::location::LocationDetector::save_user_settings(&user_settings)?;

        // ELM ARCHITECTURE FIX: Update Model state after storage save
        // The bug was that we saved to storage but didn't update the Model,
        // so the View continued showing stale data until app restart
        self.update_persistent_signal_notes(frequency_hz, notes)?;

        // ADDITIONAL FIX: Also update scan signal notes in model.windows if it exists
        // This fixes the case where both scan and persistent signals exist for same frequency
        self.update_scan_signal_notes(signal_id, notes)?;

        Ok(())
    }

    /// Save signal notes by frequency (stable identifier for persistent signals)
    /// This fixes the SignalId lookup bug by using frequency as the primary key
    fn save_signal_notes_by_frequency(
        &mut self,
        frequency_hz: f64,
        notes: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Find signal strength - try scan signals first, then persistent signals
        let signal_strength = if let Some(signal_progress) =
            self.find_signal_by_frequency_in_scan_windows(frequency_hz)
        {
            signal_progress.signal_strength.unwrap_or(0.5)
        } else if let Some(persisted_signal) =
            self.find_persistent_signal_by_frequency(frequency_hz)
        {
            persisted_signal.signal_strength
        } else {
            // Default signal strength for new persistent signals
            0.5
        };

        // Get current location (with fallback for now)
        let fallback_location = Location {
            lat: 37.7749, // San Francisco default
            lon: -122.4194,
        };
        let location = LocationDetector::current_location(Some(fallback_location))?;

        // Create persisted signal with current data
        let persisted_signal = PersistedSignal {
            frequency_hz,
            signal_strength,
            first_detected: Utc::now(),
            last_detected: Utc::now(),
            detection_count: 1,
            modulation: ModulationType::WFM,
            notes: if notes.is_empty() {
                None
            } else {
                Some(notes.to_string())
            },
        };

        // Save signal to storage
        self.signal_storage
            .save_signal(&persisted_signal, location)?;

        // Also save user settings with updated location
        let user_settings = crate::persistence::location::UserSettings {
            version: "v1.0".to_string(),
            last_known_location: Some(crate::persistence::location::CachedLocation {
                lat: location.lat,
                lon: location.lon,
                timestamp: Utc::now(),
            }),
            preferences: crate::persistence::location::UserPreferences {
                auto_save_interval_seconds: 30,
            },
        };

        // Save user settings to ~/.scanner/settings.json
        crate::persistence::location::LocationDetector::save_user_settings(&user_settings)?;

        // ELM ARCHITECTURE FIX: Update Model state after storage save
        self.update_persistent_signal_notes(frequency_hz, notes)?;

        Ok(())
    }

    /// Find signal by frequency in scan windows
    fn find_signal_by_frequency_in_scan_windows(
        &self,
        frequency_hz: f64,
    ) -> Option<&crate::ui::tui::model::types::SignalProgress> {
        for window in self.model.windows.values() {
            for signal in &window.signals {
                if (signal.frequency_hz - frequency_hz).abs() < 1000.0 {
                    return Some(signal);
                }
            }
        }
        None
    }

    /// Find persistent signal by frequency
    fn find_persistent_signal_by_frequency(&self, frequency_hz: f64) -> Option<&PersistedSignal> {
        self.model
            .persistent_signals
            .iter()
            .find(|s| (s.frequency_hz - frequency_hz).abs() < 1000.0)
    }

    /// Update persistent signal notes in the Model state (Elm Architecture pattern)
    /// This ensures the View shows updated data immediately after Update
    fn update_persistent_signal_notes(
        &mut self,
        frequency_hz: f64,
        notes: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Find and update the persistent signal in the Model
        for persistent_signal in &mut self.model.persistent_signals {
            if (persistent_signal.frequency_hz - frequency_hz).abs() < 1.0 {
                persistent_signal.notes = if notes.is_empty() {
                    None
                } else {
                    Some(notes.to_string())
                };
                persistent_signal.last_detected = chrono::Utc::now();

                debug!(
                    frequency_hz = frequency_hz,
                    notes_length = notes.len(),
                    "Updated persistent signal notes in Model state"
                );

                // Mark UI as dirty to trigger re-render
                self.model.mark_dirty();
                return Ok(());
            }
        }

        Err(format!(
            "Persistent signal not found in Model for frequency: {} Hz",
            frequency_hz
        )
        .into())
    }

    /// Update scan signal notes in model.windows (fixes UI update bug)
    /// This ensures the View shows updated data immediately after Update
    fn update_scan_signal_notes(
        &mut self,
        signal_id: &crate::ecs::components::SignalId,
        notes: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Find and update the scan signal in model.windows
        for window in &mut self.model.windows.values_mut() {
            if let Some(index) = window.signal_lookup.get(signal_id)
                && let Some(signal) = window.signals.get_mut(*index)
            {
                signal.notes = if notes.is_empty() {
                    None
                } else {
                    Some(notes.to_string())
                };

                debug!(
                    signal_id = %signal_id,
                    frequency_hz = signal.frequency_hz,
                    notes_length = notes.len(),
                    "Updated scan signal notes in Model state"
                );

                // Mark UI as dirty to trigger re-render
                self.model.mark_dirty();
                return Ok(());
            }
        }

        // Not finding the scan signal is OK - it might be a persistent-only signal
        debug!(
            signal_id = %signal_id,
            "Scan signal not found in windows (might be persistent-only signal)"
        );
        Ok(())
    }

    /// Auto-save newly confirmed signals to persistence
    fn auto_save_confirmed_signal(
        &mut self,
        signal_progress: &crate::ui::tui::model::types::SignalProgress,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let location = LocationDetector::current_location(Some(Location {
            lat: 37.7749, // San Francisco default
            lon: -122.4194,
        }))?;

        // Check if signal already exists in persistent storage to preserve metadata
        let existing_signals = self.signal_storage.load_signals_for_location(location)?;
        if let Some(existing) = existing_signals
            .iter()
            .find(|s| (s.frequency_hz - signal_progress.frequency_hz).abs() < 1000.0)
        {
            // Update existing signal: increment detection count, update last_detected, preserve
            // user notes
            let updated_signal = PersistedSignal {
                frequency_hz: existing.frequency_hz,
                signal_strength: existing
                    .signal_strength
                    .max(signal_progress.signal_strength.unwrap_or(0.5)),
                first_detected: existing.first_detected, // Preserve original
                last_detected: Utc::now(),               // Update to now
                detection_count: existing.detection_count + 1, // Increment
                modulation: existing.modulation.clone(), // Preserve
                notes: existing.notes.clone(),           // Preserve user notes
            };

            self.signal_storage.save_signal(&updated_signal, location)?;
        } else {
            // New signal - auto-save with initial metadata
            let new_signal = PersistedSignal {
                frequency_hz: signal_progress.frequency_hz,
                signal_strength: signal_progress.signal_strength.unwrap_or(0.5),
                first_detected: Utc::now(),
                last_detected: Utc::now(),
                detection_count: 1,              // First detection
                modulation: ModulationType::WFM, // Default modulation
                notes: None,                     // No notes initially
            };

            self.signal_storage.save_signal(&new_signal, location)?;
        }

        // Save user settings after signal save (maintains existing behavior)
        LocationDetector::save_user_settings(&UserSettings::default())?;

        Ok(())
    }

    /// Look up signal by ID directly from model data
    fn find_signal_by_id(
        &self,
        signal_id: &crate::ecs::components::SignalId,
    ) -> Option<&crate::ui::tui::model::types::SignalProgress> {
        for window in self.model.windows.values() {
            if let Some(index) = window.signal_lookup.get(signal_id) {
                return window.signals.get(*index);
            }
        }
        None
    }

    /// Find signal info from persistent signals for save operations
    fn find_signal_info_from_persistent(
        &self,
        signal_id: &crate::ecs::components::SignalId,
    ) -> Option<(f64, f64)> {
        // Search persistent signals by matching SignalId to frequency
        for (freq_key, stored_signal_id) in &self.model.persistent_signal_ids {
            if stored_signal_id == signal_id {
                let frequency_hz = *freq_key as f64;
                // Find the persistent signal with this frequency and return its info
                if let Some(persisted_signal) = self
                    .model
                    .persistent_signals
                    .iter()
                    .find(|s| (s.frequency_hz - frequency_hz).abs() < 1000.0)
                {
                    return Some((
                        persisted_signal.frequency_hz,
                        persisted_signal.signal_strength,
                    ));
                }
            }
        }
        None
    }

    /// Process TUI events (progress updates and discovery)
    fn process_tui_events(&mut self, iterations: &mut u32) -> io::Result<()> {
        while let Ok(event) = self.receiver.try_recv() {
            self.model.update_tui_event(event);
            *iterations = 0; // Reset iteration counter when we get events
        }
        Ok(())
    }

    /// Handle keyboard input with timeout, returns true if redraw needed
    fn handle_keyboard_input(&mut self, animation_interval: Duration) -> io::Result<bool> {
        if event::poll(animation_interval)? {
            match event::read() {
                Ok(Event::Key(key)) => {
                    // Check for quit keys first
                    if self.handle_quit_keys(&key) {
                        return Ok(false);
                    }

                    // Theme selector takes priority when open
                    if self.model.theme_selector_open {
                        self.handle_theme_selector(key.code);
                        return Ok(false);
                    }

                    // Modal takes priority when open
                    if self.model.should_handle_modal_input(&key) {
                        let handled = self.handle_modal_input(&key);
                        if handled {
                            return Ok(false);
                        }
                    }

                    // Toggle theme selector
                    if matches!(key.code, KeyCode::Char('T')) {
                        let all_themes = themes::ThemeName::all();
                        let current_idx = all_themes
                            .iter()
                            .position(|t| t == &self.current_theme)
                            .unwrap_or(0);
                        self.model.theme_selector_index = current_idx;
                        self.model.toggle_theme_selector();
                        return Ok(false);
                    }

                    // Handle spacebar for global pause/resume
                    if matches!(key.code, KeyCode::Char(' ')) {
                        self.handle_spacebar_pause();
                        return Ok(false);
                    }

                    // Handle notes editing
                    if self.model.is_editing_notes() {
                        let handled = self.handle_notes_editing_keys(&key);
                        if handled {
                            return Ok(false);
                        }
                    }

                    // Open signal detail modal with ENTER key when focused on signals table
                    if matches!(key.code, KeyCode::Enter)
                        && !self.model.is_editing_notes()
                        && matches!(self.model.focus_state, FocusState::SignalsTable(_))
                    {
                        self.model.handle_signal_table_enter_key(&key);
                        return Ok(false);
                    }

                    // Handle Tab for cycling focus between tables
                    if matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
                        let tuner_count = self.model.tuners.len();
                        if key.modifiers.contains(event::KeyModifiers::SHIFT)
                            || matches!(key.code, KeyCode::BackTab)
                        {
                            self.model.navigate_previous_table(tuner_count);
                        } else {
                            self.model.navigate_next_table(tuner_count);
                        }
                        return Ok(false);
                    }

                    // Handle navigation
                    if matches!(
                        key.code,
                        KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right
                    ) {
                        self.handle_navigation_keys(key.code);
                        return Ok(false);
                    }

                    // Handle tuning actions (Enter key)
                    let needs_redraw = self.handle_tuning_actions(&key);
                    return Ok(needs_redraw);
                }
                Ok(other_event) => {
                    debug!(event = ?other_event, "TUI: Non-key event received");
                }
                Err(e) => {
                    debug!(error = ?e, "TUI: Error reading event");
                }
            }
        }

        Ok(false)
    }

    /// Main TUI event loop
    fn run_app<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> io::Result<()> {
        let mut iterations = 0;
        let animation_interval = Duration::from_millis(100); // 10 FPS for slower, smoother animation

        // Initialize locality on first run
        if self.cached_locality.is_none() {
            self.update_cached_locality_sync();
        }

        loop {
            // Update spectrum data from entities
            self.update_spectrum_from_entities();

            // Always mark dirty for smooth spectrum animation at 10 FPS
            self.model.mark_dirty();

            // Redraw for animation
            terminal.draw(|f| self.ui(f))?;
            self.model.clear_dirty();
            iterations += 1;

            if self.should_quit() {
                break;
            }

            if self.should_auto_exit(iterations) {
                break;
            }

            // Handle keyboard input
            self.handle_keyboard_input(animation_interval)?;

            // Process TUI events
            self.process_tui_events(&mut iterations)?;

            // Check if we need to transition from AwaitingTune to Listening
            self.check_awaiting_tune_transition();
        }

        Ok(())
    }

    /// Update cached locality using location resource
    fn update_cached_locality_sync(&mut self) {
        // Try to get locality from LocationResource if available
        if let Some(ref location_resource) = self.location_resource
            && let Ok(mut resource) = location_resource.try_lock()
            && let Ok(detected_location) = resource.detect_current_location()
        {
            self.cached_locality = Some(detected_location.locality_name());
            return;
        }

        // Fallback to settings-based location
        let locality = if let Ok(settings) = LocationDetector::load_user_settings() {
            if settings.last_known_location.is_some() {
                Some("Local".to_string())
            } else {
                None
            }
        } else {
            None
        };

        self.cached_locality = locality;
    }

    fn ui(&mut self, f: &mut Frame) {
        let theme = self.theme.as_ref();
        let theme_name = self.current_theme.to_string();

        let tuner_count = self.model.tuners.len();
        let confirmed_signals_count = self.model.confirmed_signal_count();
        let total_signals_count = self.model.displayable_signal_count();
        let layout = Layout::new(
            f.area(),
            tuner_count,
            confirmed_signals_count,
            total_signals_count,
        );

        header::render_header(
            f,
            layout.header,
            &self.model,
            theme,
            self.cached_locality.as_deref(),
        );
        render_activities(f, layout.activities, &mut self.model, theme);

        if let Some(task_id) = self.model.displayed_task_id.clone() {
            task_progress::render_task_progress(
                f,
                layout.scan_progress,
                &mut self.model,
                theme,
                &task_id,
            );
        } else {
            task_progress::render_no_progress_message(
                f,
                layout.scan_progress,
                theme,
                "",
                "No task selected",
            );
        }

        tuners::render_tuners(f, layout.tuners, &mut self.model, theme);
        signals_table::render_signals_table(f, layout.signals_table, &mut self.model, theme);

        let all_themes: Vec<String> = themes::ThemeName::all()
            .iter()
            .map(|t| t.display_name().to_string())
            .collect();
        instructions::render_instructions(
            f,
            layout.instructions,
            theme,
            &theme_name,
            &self.model,
            &all_themes,
        );

        // Render modal on top of everything else if it's open
        if self.model.should_render_modal() {
            renderers::modal::render_signal_detail_modal(f, &self.model, theme);
        }
    }

    /// Check if we're running in an interactive terminal
    fn is_terminal_interactive(&self) -> bool {
        use std::io::IsTerminal;

        // Check if stdout is a TTY
        if !io::stdout().is_terminal() {
            return false;
        }

        // Check for CI environment variables
        if std::env::var("CI").is_ok()
            || std::env::var("GITHUB_ACTIONS").is_ok()
            || std::env::var("GITLAB_CI").is_ok()
            || std::env::var("JENKINS_URL").is_ok()
        {
            return false;
        }

        // Check if TERM is set to something reasonable
        match std::env::var("TERM") {
            Ok(term) => !term.is_empty() && term != "dumb",
            Err(_) => false,
        }
    }

    /// Fallback text-based progress display for non-interactive environments
    fn run_text_fallback(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ConsoleRenderer::tty_println("┌─ Scanning FM stations ... ───────────────┐");
        ConsoleRenderer::tty_println("│ Running in text mode                     │");
        ConsoleRenderer::tty_println("│ Press CTRL-C to exit                     │");
        ConsoleRenderer::tty_println("└──────────────────────────────────────────┘");

        let mut last_update = Instant::now();
        let update_interval = Duration::from_millis(1000); // Update every second

        loop {
            // Check for shutdown signal
            if self.shutdown_token.is_cancelled() {
                break;
            }

            // Process progress events
            while let Ok(event) = self.receiver.try_recv() {
                self.model.update_tui_event(event);
            }

            // Update display periodically
            if last_update.elapsed() >= update_interval {
                ConsoleRenderer::print_text_progress(&self.model);
                last_update = Instant::now();
            }

            // Small sleep to prevent busy waiting
            std::thread::sleep(Duration::from_millis(50));

            // Auto-exit after reasonable time if no activity, or if all signals are done
            if (self.model.is_empty() && last_update.elapsed() > Duration::from_secs(30))
                || (!self.model.is_empty()
                    && self.model.all_complete()
                    && last_update.elapsed() > Duration::from_secs(3))
            {
                break;
            }
        }

        ConsoleRenderer::tty_println("Scanning complete.");
        Ok(())
    }

    /// Simple TUI-like display that works without raw mode or alternate screen
    fn run_simple_tui(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Clear screen and move to top
        ConsoleRenderer::tty_print("\x1B[2J\x1B[H");

        ConsoleRenderer::tty_println("┌─ Scanning FM stations ... ───────────────┐");
        ConsoleRenderer::tty_println("│ TUI Mode (Simplified)                    │");
        ConsoleRenderer::tty_println("│ Press CTRL-C to exit                     │");
        ConsoleRenderer::tty_println("└──────────────────────────────────────────┘");
        ConsoleRenderer::tty_println("");

        let mut last_update = Instant::now();
        let update_interval = Duration::from_millis(500); // Update every 500ms
        let mut last_signal_count = 0;

        loop {
            // Check for shutdown signal
            if self.shutdown_token.is_cancelled() {
                break;
            }

            // Process progress events
            while let Ok(event) = self.receiver.try_recv() {
                self.model.update_tui_event(event);
            }

            // Update display periodically or when signals change
            let current_signal_count = self.model.signal_count();
            if last_update.elapsed() >= update_interval || current_signal_count != last_signal_count
            {
                // Move cursor up to overwrite previous output
                if last_signal_count > 0 {
                    let lines_to_clear = ConsoleRenderer::calculate_display_lines(&self.model);
                    ConsoleRenderer::tty_print(&format!("\x1B[{}A", lines_to_clear)); // Move cursor up
                }

                ConsoleRenderer::print_tui_style_progress(&self.model);
                last_update = Instant::now();
                last_signal_count = current_signal_count;
            }

            // Small sleep to prevent busy waiting
            std::thread::sleep(Duration::from_millis(50));

            // Auto-exit conditions
            if (self.model.is_empty() && last_update.elapsed() > Duration::from_secs(30))
                || (!self.model.is_empty()
                    && self.model.all_complete()
                    && last_update.elapsed() > Duration::from_secs(3))
            {
                break;
            }
        }

        ConsoleRenderer::tty_println("\nScanning complete.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, RwLock},
        time::SystemTime,
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::{signals::ModulationType, types::Signal},
        ecs::{AudioEntity, EntityWorld},
    };

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
    fn test_pause_all_audio_stops_playing_entities() {
        use tokio_util::sync::CancellationToken;

        let signal = create_test_signal();
        let audio1 = AudioEntity::new(signal.clone(), 88.9e6, None);
        let audio2 = AudioEntity::new(signal, 89.3e6, None);

        let mut audio_world = EntityWorld::new();
        audio_world.insert(audio1);
        audio_world.insert(audio2);

        let audio_entities = Arc::new(RwLock::new(audio_world));

        let (_, receiver) = mpsc::channel();
        let shutdown_token = CancellationToken::new();
        let mut tui = TuiProgressDisplay::new(receiver, shutdown_token);
        tui.audio_entities = Some(audio_entities.clone());

        {
            let entities = audio_entities.read().unwrap();
            assert_eq!(entities.len(), 2);
            assert!(
                entities.iter().all(|e| e.is_playing()),
                "All audio should be playing before pause"
            );
        }

        tui.pause_all_audio();

        {
            let entities = audio_entities.read().unwrap();
            assert!(
                entities.iter().all(|e| !e.is_playing()),
                "All audio should be stopped after pause_all_audio"
            );
        }
    }

    #[test]
    fn test_update_cached_locality_uses_location_resource() {
        // Create a TUI with location resource
        let (_tui_tx, tui_rx) = std::sync::mpsc::channel();
        let shutdown_token = tokio_util::sync::CancellationToken::new();

        let location_resource = crate::ecs::resources::new_location_resource();

        let mut tui = TuiProgressDisplay::new(tui_rx, shutdown_token)
            .with_location_resource(location_resource);

        // Initially cached_locality should be None
        assert!(tui.cached_locality.is_none());

        // Call the method - should detect location and set locality
        tui.update_cached_locality_sync();

        // Should now have a locality (not the stub "Local" string)
        assert!(tui.cached_locality.is_some());
        let locality = tui.cached_locality.unwrap();

        // Should be actual city name, not the stub "Local" string
        assert_ne!(locality, "Local");
        assert!(!locality.is_empty());
    }

    #[test]
    fn test_locality_display_basic_integration() {
        let (_tui_tx, tui_rx) = std::sync::mpsc::channel();
        let shutdown_token = tokio_util::sync::CancellationToken::new();

        // Test basic TUI functionality without location resource
        let mut tui = TuiProgressDisplay::new(tui_rx, shutdown_token);

        // Initially cached_locality should be None
        assert!(tui.cached_locality.is_none());

        // Update locality without location resource - should work without panic
        tui.update_cached_locality_sync();

        // May or may not have locality depending on settings
        // Test passes as long as it doesn't panic
    }

    #[test]
    fn test_locality_display_with_real_location_resource() {
        let (_tui_tx, tui_rx) = std::sync::mpsc::channel();
        let shutdown_token = tokio_util::sync::CancellationToken::new();

        // Use real location resource
        let location_resource = crate::ecs::resources::new_location_resource();

        let mut tui = TuiProgressDisplay::new(tui_rx, shutdown_token)
            .with_location_resource(location_resource);

        // Initially cached_locality should be None
        assert!(tui.cached_locality.is_none());

        // Update locality - should either detect location or fall back gracefully
        tui.update_cached_locality_sync();

        // Should either have a locality or None - test passes as long as no panic
        if let Some(locality) = &tui.cached_locality {
            assert!(!locality.is_empty());
            // Should not be placeholder values
            assert_ne!(locality, "Local");
            assert_ne!(locality, "Unknown");
        }
    }

    #[test]
    fn test_locality_caching_behavior() {
        let (_tui_tx, tui_rx) = std::sync::mpsc::channel();
        let shutdown_token = tokio_util::sync::CancellationToken::new();

        let location_resource = crate::ecs::resources::new_location_resource();

        let mut tui = TuiProgressDisplay::new(tui_rx, shutdown_token)
            .with_location_resource(location_resource);

        // First update
        tui.update_cached_locality_sync();
        let first_result = tui.cached_locality.clone();

        // Second update should be consistent (cached or same result)
        tui.update_cached_locality_sync();
        let second_result = tui.cached_locality.clone();

        // Results should be consistent
        assert_eq!(first_result, second_result);
    }

    #[test]
    fn test_ui_locality_display_thread_safety() {
        use std::thread;

        let location_resource = crate::ecs::resources::new_location_resource();

        // Test multiple threads accessing same location resource
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let location_resource = location_resource.clone();
                thread::spawn(move || {
                    let (_tui_tx, tui_rx) = std::sync::mpsc::channel();
                    let shutdown_token = tokio_util::sync::CancellationToken::new();

                    let mut tui = TuiProgressDisplay::new(tui_rx, shutdown_token)
                        .with_location_resource(location_resource);

                    tui.update_cached_locality_sync();
                    tui.cached_locality
                })
            })
            .collect();

        // All threads should complete successfully
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All should either succeed with same result or handle errors gracefully
        for locality in results.into_iter().flatten() {
            assert!(!locality.is_empty());
        }
    }
}
