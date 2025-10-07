//! TUI data model using The Elm Architecture pattern

use crate::{
    sdr::DeviceInfo,
    terminal::{ProgressEvent, ProgressEventType, TuiEvent},
};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};
use tracing::debug;

/// Selected candidate information
#[derive(Debug, Clone)]
pub struct SelectedCandidateInfo {
    pub candidate_id: String,
    pub metadata: crate::window::WindowMetadata,
    pub candidate_frequency: f64,
    pub signal_strength: Option<f64>,
    pub audio_quality: Option<crate::audio_quality::AudioQuality>,
}

/// Information about a candidate's progress
#[derive(Debug, Clone)]
pub struct CandidateProgress {
    pub candidate_id: String,
    pub frequency_hz: f64,
    pub metadata: crate::window::WindowMetadata,
    pub completion: f64,
    pub status: CandidateStatus,
    pub audio_quality: Option<crate::audio_quality::AudioQuality>,
    pub signal_strength: Option<f64>,
    pub last_update: Instant,
}

/// Information about a scanning window
#[derive(Debug, Clone)]
pub struct WindowProgress {
    #[allow(dead_code)] // Kept for debugging and potential future use
    pub window_id: usize,
    pub candidates: Vec<CandidateProgress>,
    pub is_complete: bool,
    pub candidate_lookup: HashMap<String, usize>, // candidate_id -> index in candidates vec
}

impl WindowProgress {
    /// Check if this window should be displayed in the UI
    /// Returns false if all candidates are rejected (noise) and window is complete
    pub fn should_display(&self) -> bool {
        // Always show incomplete windows
        if !self.is_complete {
            return true;
        }

        // For complete windows, only show if there's at least one non-rejected candidate
        self.candidates
            .iter()
            .any(|candidate| candidate.status != CandidateStatus::Rejected)
    }

    /// Get candidates that should be displayed for this window
    /// For complete windows with signals, hide rejected candidates
    /// For current window during scanning, show all candidates
    /// In selection mode, always hide rejected candidates
    pub fn displayable_candidates(
        &self,
        is_current_window: bool,
        in_selection_mode: bool,
    ) -> Vec<&CandidateProgress> {
        // In selection mode, always hide rejected candidates regardless of window status
        if in_selection_mode {
            return self
                .candidates
                .iter()
                .filter(|candidate| candidate.status != CandidateStatus::Rejected)
                .collect();
        }

        // For complete windows, always hide rejected candidates (even if current window)
        if self.is_complete {
            return self
                .candidates
                .iter()
                .filter(|candidate| candidate.status != CandidateStatus::Rejected)
                .collect();
        }

        // For incomplete windows, show all candidates
        // (including rejected ones, since they might still be processing)
        if !self.is_complete || is_current_window {
            self.candidates.iter().collect()
        } else {
            // This case should not be reachable, but handle it anyway
            self.candidates
                .iter()
                .filter(|candidate| candidate.status != CandidateStatus::Rejected)
                .collect()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    Detected,
    Analyzing,
    Rejected,
    Signal,
    Playing,
    Completed,
}

impl CandidateStatus {
    pub fn to_string(&self) -> &'static str {
        match self {
            CandidateStatus::Detected => "DETECTED",
            CandidateStatus::Analyzing => "ANALYZING",
            CandidateStatus::Rejected => "NOISE",
            CandidateStatus::Signal => "SIGNAL",
            CandidateStatus::Playing => "PLAYING",
            CandidateStatus::Completed => "DONE",
        }
    }
}

/// Focus state for component navigation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusState {
    Spectrum,
    Scan,
    Tuner(usize), // Index of focused tuner
}

/// Tuner state - what a specific tuner/SDR device is doing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunerState {
    /// Tuner is idle and available for use
    Available,
    /// Tuner is actively scanning for signals
    Scanning,
    /// Tuner is listening to a station
    Listening,
}

impl TunerState {
    pub fn display(&self) -> &'static str {
        match self {
            TunerState::Available => "Available",
            TunerState::Scanning => "Scanning",
            TunerState::Listening => "Listening",
        }
    }
}

/// UI interaction mode - what the user is currently doing
/// This is separate from scanner state (what SDRs are doing in background)
#[derive(Debug, Clone, PartialEq)]
pub enum UiMode {
    /// Watching scan progress (no candidate selected)
    Idle,

    /// Candidate selected, navigating scanner results while scan may still be running
    NavigatingScanner { selected_index: usize },

    /// Scan paused, waiting for Paused event before tuning to station
    AwaitingTune {
        navigation_index: usize,
        tuning_index: usize,
    },

    /// Actively listening to a station (scan paused, audio playing)
    Listening {
        navigation_index: usize,
        playing_index: usize,
        playing_candidate_id: String,
    },
}

/// Main application model following The Elm Architecture
#[derive(Debug)]
pub struct Model {
    pub windows: BTreeMap<usize, WindowProgress>, // window_id -> WindowProgress (ordered by window_id)
    pub current_window: usize,
    pub total_windows: Option<usize>, // Total number of windows in the scan (None until known)
    pub should_quit: bool,
    pub theme_selector_open: bool,
    pub theme_selector_index: usize,

    // State machine-based UI mode
    pub ui_mode: UiMode,

    pub scroll_offset: usize, // Number of candidates to skip when rendering (for scrolling)
    pub playback_active: bool, // Tracks if actively listening to a station
    pub focus_state: FocusState, // Which component has focus
    pub tuners: Vec<DeviceInfo>, // Discovered SDR devices
    pub tuner_states: HashMap<crate::sdr::DeviceId, TunerState>, // State of each tuner
    pub active_tuners: Option<crate::main_thread::ActiveTuners>, // Source of truth for tuner allocation
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self {
            windows: BTreeMap::new(),
            current_window: 0,
            total_windows: None,
            should_quit: false,
            theme_selector_open: false,
            theme_selector_index: 0,
            ui_mode: UiMode::Idle,
            scroll_offset: 0,
            playback_active: false,
            focus_state: FocusState::Spectrum,
            tuners: Vec::new(),
            tuner_states: HashMap::new(),
            active_tuners: None,
        }
    }

    /// Add a newly discovered tuner
    pub fn add_device(&mut self, tuner: DeviceInfo) {
        // Check if tuner already exists
        if !self.tuners.iter().any(|d| d.id == tuner.id) {
            debug!(tuner_id = ?tuner.id, label = %tuner.label, "Tuner added to TUI model");

            // All newly discovered tuners start as Available
            // Progress events will update state to Scanning/Listening for the tuner actually in use
            self.tuner_states
                .insert(tuner.id.clone(), TunerState::Available);
            self.tuners.push(tuner);
        }
    }

    /// Remove a tuner that was unplugged
    pub fn remove_device(&mut self, tuner_id: &crate::sdr::DeviceId) {
        if let Some(pos) = self.tuners.iter().position(|d| &d.id == tuner_id) {
            let tuner = self.tuners.remove(pos);
            self.tuner_states.remove(&tuner.id);
            debug!(tuner_id = ?tuner.id, label = %tuner.label, "Tuner removed from TUI model");
        }
    }

    /// Get the state of a specific tuner based on active tuners allocation
    pub fn tuner_state(&self, tuner_id: &crate::sdr::DeviceId) -> TunerState {
        if let Some(ref active) = self.active_tuners {
            if active.scanning.contains(tuner_id) {
                return TunerState::Scanning;
            }
            if active.listening.contains(tuner_id) {
                return TunerState::Listening;
            }
        }
        // Fall back to HashMap for backward compatibility during transition
        // (e.g., Paused event still uses HashMap)
        self.tuner_states
            .get(tuner_id)
            .cloned()
            .unwrap_or(TunerState::Available)
    }

    /// Get count of discovered tuners
    pub fn device_count(&self) -> usize {
        self.tuners.len()
    }

    /// Update the model based on a TUI event (progress or discovery)
    pub fn update_tui_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::Progress(progress_event) => self.update(progress_event),
            TuiEvent::TunerAdded(tuner) => self.add_device(tuner),
            TuiEvent::TunerRemoved(tuner_id) => self.remove_device(&tuner_id),
            TuiEvent::Paused { tuner_id } => {
                debug!(tuner_id = ?tuner_id, "Scanning paused, tuner now available");
                self.tuner_states.insert(tuner_id, TunerState::Available);
            }
            TuiEvent::ActiveTunersUpdated {
                available,
                scanning,
                listening,
            } => {
                debug!(
                    available_count = available.len(),
                    scanning_count = scanning.len(),
                    listening_count = listening.len(),
                    "Active tuners updated"
                );
                self.active_tuners = Some(crate::main_thread::ActiveTuners {
                    available,
                    scanning,
                    listening,
                });
            }
        }
    }

    /// Update the model based on a progress event
    pub fn update(&mut self, event: ProgressEvent) {
        if !self.should_process_event(&event) {
            return;
        }

        self.update_current_window(&event);
        // Tuner state now managed by ActiveTunersUpdated events

        if let Some(candidate_id) = event.candidate_id.clone() {
            self.update_candidate(event, &candidate_id);
        }

        self.complete_window_if_done();
    }

    fn should_process_event(&self, event: &ProgressEvent) -> bool {
        if self.is_interactive()
            && event.event_type != ProgressEventType::AudioPlaybackStarted
            && event.event_type != ProgressEventType::AudioPlaybackCompleted
        {
            return false;
        }

        !matches!(event.event_type, ProgressEventType::PeakDetected)
    }

    fn update_current_window(&mut self, event: &ProgressEvent) {
        if event.metadata.window_id > self.current_window {
            self.current_window = event.metadata.window_id;
            for (window_id, window) in self.windows.iter_mut() {
                if *window_id < self.current_window {
                    window.is_complete = true;
                }
            }
        }
    }

    // Tuner state is now managed by ActiveTunersUpdated events from MainThread
    // No longer need to infer state from progress events

    fn update_candidate(&mut self, event: ProgressEvent, candidate_id: &str) {
        debug!(
            event_type = ?event.event_type,
            candidate_id = ?candidate_id,
            window_id = event.metadata.window_id,
            current_window = self.current_window,
            ui_mode = ?self.ui_mode,
            "Processing event with candidate_id"
        );

        if event.metadata.window_id < self.current_window
            && !(self.is_interactive()
                && (event.event_type == ProgressEventType::AudioPlaybackStarted
                    || event.event_type == ProgressEventType::AudioPlaybackCompleted))
        {
            debug!("Ignoring event for old window");
            return;
        }

        if event.event_type == ProgressEventType::AudioPlaybackStarted {
            self.clear_playing_candidates(candidate_id);
        }

        let window_id = event.metadata.window_id;
        let window = self.get_or_create_window(window_id);

        let candidate_index = if let Some(&index) = window.candidate_lookup.get(candidate_id) {
            debug!(index = index, "Found existing candidate");
            index
        } else {
            debug!("Creating new candidate");
            let new_candidate = CandidateProgress {
                candidate_id: candidate_id.to_string(),
                frequency_hz: event.frequency_hz,
                metadata: event.metadata,
                completion: 0.0,
                status: CandidateStatus::Detected,
                audio_quality: None,
                signal_strength: None,
                last_update: Instant::now(),
            };
            let index = window.candidates.len();
            window.candidates.push(new_candidate);
            window
                .candidate_lookup
                .insert(candidate_id.to_string(), index);
            index
        };

        {
            let candidate = &mut window.candidates[candidate_index];

            match event.event_type {
                ProgressEventType::CandidateCreated => {
                    candidate.status = CandidateStatus::Detected;
                    candidate.completion = 0.3;
                }
                ProgressEventType::AudioAnalysisStarted => {
                    candidate.status = CandidateStatus::Analyzing;
                    candidate.completion = 0.5;
                }
                ProgressEventType::AudioAnalysisCompleted => {
                    if candidate.status == CandidateStatus::Signal {
                    } else if candidate.status != CandidateStatus::Rejected {
                        candidate.status = CandidateStatus::Signal;
                        candidate.completion = 0.6;
                    } else {
                        candidate.completion = 1.0;
                    }
                }
                ProgressEventType::CandidateRejected => {
                    candidate.status = CandidateStatus::Rejected;
                    candidate.completion = 1.0;
                }
                ProgressEventType::SignalGenerated => {
                    candidate.status = CandidateStatus::Signal;
                    candidate.completion = 0.6;
                    if let Some(quality) = event.audio_quality {
                        candidate.audio_quality = Some(quality);
                    }
                    if let Some(strength) = event.signal_strength {
                        candidate.signal_strength = Some(strength);
                    }
                }
                ProgressEventType::AudioPlaybackStarted => {
                    debug!(
                        frequency_mhz = event.frequency_hz / 1e6,
                        candidate_id = ?candidate_id,
                        "Setting candidate to Playing status"
                    );
                    candidate.status = CandidateStatus::Playing;
                    candidate.completion = 0.8;
                }
                ProgressEventType::AudioPlaybackCompleted => {
                    candidate.status = CandidateStatus::Completed;
                    candidate.completion = 1.0;
                }
                ProgressEventType::ThreadCompleted | ProgressEventType::PeakDetected => {}
            }

            if let Some(quality) = event.audio_quality {
                candidate.audio_quality = Some(quality);
            }
            candidate.last_update = Instant::now();
        }

        if event.event_type == ProgressEventType::AudioPlaybackStarted {
            match &self.ui_mode {
                UiMode::AwaitingTune {
                    navigation_index,
                    tuning_index,
                }
                | UiMode::Listening {
                    navigation_index,
                    playing_index: tuning_index,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: *navigation_index,
                        playing_index: *tuning_index,
                        playing_candidate_id: candidate_id.to_string(),
                    };
                    // Note: Tuner state is set to Listening by update_tuner_state()
                }
                _ => {}
            }
        }
    }

    fn clear_playing_candidates(&mut self, new_playing_id: &str) {
        debug!(
            new_playing_candidate = ?new_playing_id,
            "Clearing all other Playing candidates before setting new one"
        );
        for window in self.windows.values_mut() {
            for candidate in &mut window.candidates {
                if candidate.status == CandidateStatus::Playing {
                    debug!(
                        cleared_candidate = ?candidate.candidate_id,
                        "Clearing Playing status from candidate"
                    );
                    candidate.status = CandidateStatus::Completed;
                    candidate.completion = 1.0;
                }
            }
        }
    }

    fn get_or_create_window(&mut self, window_id: usize) -> &mut WindowProgress {
        self.windows
            .entry(window_id)
            .or_insert_with(|| WindowProgress {
                window_id,
                candidates: Vec::new(),
                is_complete: false,
                candidate_lookup: HashMap::new(),
            })
    }

    fn complete_window_if_done(&mut self) {
        if self.all_complete()
            && let Some(window) = self.windows.get_mut(&self.current_window)
        {
            window.is_complete = true;
        }
    }

    /// Check if all windows are empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty() || self.windows.values().all(|w| w.candidates.is_empty())
    }

    /// Check if all candidates in all windows are complete (not checking if scan itself is done)
    pub fn all_candidates_complete(&self) -> bool {
        !self.windows.is_empty()
            && self.windows.values().all(|window| {
                window.candidates.iter().all(|candidate| {
                    candidate.completion >= 1.0
                        && (candidate.status == CandidateStatus::Completed
                            || candidate.status == CandidateStatus::Rejected)
                })
            })
    }

    /// Check if scan is complete (all windows scanned AND all candidates complete)
    pub fn all_complete(&self) -> bool {
        if let Some(total) = self.total_windows {
            // We know total windows - check if we've reached it
            self.current_window >= total && self.all_candidates_complete()
        } else {
            // We don't know total windows yet - scan can't be complete
            false
        }
    }

    /// Get total candidate count across all windows
    pub fn candidate_count(&self) -> usize {
        self.windows.values().map(|w| w.candidates.len()).sum()
    }

    /// Request to quit the application
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    // UiMode helper methods
    pub fn is_idle(&self) -> bool {
        matches!(self.ui_mode, UiMode::Idle)
    }

    pub fn is_navigating(&self) -> bool {
        matches!(self.ui_mode, UiMode::NavigatingScanner { .. })
    }

    pub fn is_awaiting_tune(&self) -> bool {
        matches!(self.ui_mode, UiMode::AwaitingTune { .. })
    }

    pub fn is_listening(&self) -> bool {
        matches!(self.ui_mode, UiMode::Listening { .. })
    }

    pub fn is_interactive(&self) -> bool {
        !matches!(self.ui_mode, UiMode::Idle)
    }

    /// Computed property: selection_mode derived from UiMode
    /// Returns true when user can navigate with arrow keys
    pub fn selection_mode(&self) -> bool {
        self.is_interactive()
    }

    /// Computed property: browsing_mode derived from UiMode
    /// Returns true when scan is paused for manual browsing/listening
    pub fn browsing_mode(&self) -> bool {
        matches!(
            self.ui_mode,
            UiMode::AwaitingTune { .. } | UiMode::Listening { .. }
        )
    }

    /// Computed property: selected_candidate_index derived from UiMode
    /// Returns the navigation index (where arrow keys are positioned)
    pub fn selected_candidate_index(&self) -> Option<usize> {
        match &self.ui_mode {
            UiMode::NavigatingScanner { selected_index } => Some(*selected_index),
            UiMode::AwaitingTune {
                navigation_index, ..
            } => Some(*navigation_index),
            UiMode::Listening {
                navigation_index, ..
            } => Some(*navigation_index),
            UiMode::Idle => None,
        }
    }

    pub fn toggle_theme_selector(&mut self) {
        self.theme_selector_open = !self.theme_selector_open;
    }

    pub fn close_theme_selector(&mut self) {
        self.theme_selector_open = false;
    }

    pub fn theme_selector_next(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + 1) % theme_count;
        }
    }

    pub fn theme_selector_prev(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + theme_count - 1) % theme_count;
        }
    }

    /// Enter selection mode - pauses scanning and allows browsing candidates
    pub fn enter_selection_mode(&mut self) {
        // Start with most recent candidate selected
        let candidate_count = self.get_selectable_candidate_count();
        if candidate_count > 0 {
            let selected_index = candidate_count - 1;
            self.ui_mode = UiMode::NavigatingScanner { selected_index };
        }
    }

    /// Exit selection mode - returns to normal scanning
    pub fn exit_selection_mode(&mut self) {
        self.ui_mode = UiMode::Idle;
    }

    /// Exit browsing mode and return to normal scanning (clears both modes)
    pub fn exit_browsing_mode(&mut self) {
        self.ui_mode = UiMode::Idle;
        // Tuner state is now managed by ActiveTunersUpdated events from MainThread
    }

    /// Get ordered list of displayable windows (oldest to newest)
    pub fn get_displayable_windows(&self) -> Vec<(&usize, &WindowProgress)> {
        self.windows
            .iter()
            .filter(|(_, window)| window.should_display())
            .collect()
    }

    /// Get count of displayable windows
    pub fn get_displayable_window_count(&self) -> usize {
        self.windows
            .values()
            .filter(|window| window.should_display())
            .count()
    }

    /// Get flattened list of displayable candidates across all windows
    /// This includes rejected candidates for display purposes (during scanning)
    /// In selection mode, rejected candidates are filtered out
    pub fn get_displayable_candidates(&self) -> Vec<(usize, &CandidateProgress)> {
        let mut candidates = Vec::new();
        for (window_id, window) in self.get_displayable_windows() {
            let is_current = *window_id == self.current_window;
            for candidate in window.displayable_candidates(is_current, self.selection_mode()) {
                candidates.push((*window_id, candidate));
            }
        }
        candidates
    }

    /// Get flattened list of selectable candidates across all windows
    /// Filters out rejected candidates - users should not be able to select rejected stations
    pub fn get_selectable_candidates(&self) -> Vec<(usize, &CandidateProgress)> {
        let mut candidates = Vec::new();
        for (window_id, window) in self.get_displayable_windows() {
            let is_current = *window_id == self.current_window;
            for candidate in window.displayable_candidates(is_current, self.selection_mode()) {
                // Skip rejected candidates - they shouldn't be selectable
                if candidate.status != CandidateStatus::Rejected {
                    candidates.push((*window_id, candidate));
                }
            }
        }
        candidates
    }

    /// Get count of displayable candidates (includes rejected for display)
    pub fn get_displayable_candidate_count(&self) -> usize {
        self.get_displayable_candidates().len()
    }

    /// Get count of selectable candidates (excludes rejected)
    pub fn get_selectable_candidate_count(&self) -> usize {
        self.get_selectable_candidates().len()
    }

    /// Get the window_id, center frequency, and candidate frequency for the currently selected candidate
    pub fn selected_candidate_info(&self) -> Option<SelectedCandidateInfo> {
        if !self.selection_mode() {
            return None;
        }

        let selected_idx = self.selected_candidate_index()?;
        let candidates = self.get_selectable_candidates();

        if selected_idx >= candidates.len() {
            return None;
        }

        let (window_id, candidate) = candidates[selected_idx];

        debug!(
            window_id = window_id,
            frequency_mhz = candidate.frequency_hz / 1e6,
            signal_strength = ?candidate.signal_strength,
            audio_quality = ?candidate.audio_quality,
            "Selected candidate info"
        );

        Some(SelectedCandidateInfo {
            candidate_id: candidate.candidate_id.clone(),
            metadata: candidate.metadata,
            candidate_frequency: candidate.frequency_hz,
            signal_strength: candidate.signal_strength,
            audio_quality: candidate.audio_quality,
        })
    }

    /// Select next candidate (moving forward in time)
    pub fn select_next_candidate(&mut self) {
        self.select_next_candidate_with_viewport(20); // Default viewport height
    }

    /// Select next candidate with viewport height for scroll adjustment
    pub fn select_next_candidate_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let candidate_count = self.get_selectable_candidate_count();
        if candidate_count == 0 {
            return;
        }

        let current = self.selected_candidate_index().unwrap_or(0);
        // Can move past last candidate to "Continue scan" position
        let next = (current + 1).min(candidate_count);

        if next != current {
            // Update navigation index based on mode
            match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => {
                    self.ui_mode = UiMode::NavigatingScanner {
                        selected_index: next,
                    };
                }
                UiMode::AwaitingTune { tuning_index, .. } => {
                    self.ui_mode = UiMode::AwaitingTune {
                        navigation_index: next,
                        tuning_index: *tuning_index,
                    };
                }
                UiMode::Listening {
                    playing_index,
                    playing_candidate_id,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: next,
                        playing_index: *playing_index,
                        playing_candidate_id: playing_candidate_id.clone(),
                    };
                }
                UiMode::Idle => {}
            }
            self.adjust_scroll_to_selection(viewport_height);
        }
    }

    /// Select previous candidate (moving backward in time)
    pub fn select_previous_candidate(&mut self) {
        self.select_previous_candidate_with_viewport(20); // Default viewport height
    }

    /// Select previous candidate with viewport height for scroll adjustment
    pub fn select_previous_candidate_with_viewport(&mut self, viewport_height: usize) {
        if !self.selection_mode() {
            return;
        }

        let current = self.selected_candidate_index().unwrap_or(0);
        if current > 0 {
            let prev = current - 1;
            // Update navigation index based on mode
            match &self.ui_mode {
                UiMode::NavigatingScanner { .. } => {
                    self.ui_mode = UiMode::NavigatingScanner {
                        selected_index: prev,
                    };
                }
                UiMode::AwaitingTune { tuning_index, .. } => {
                    self.ui_mode = UiMode::AwaitingTune {
                        navigation_index: prev,
                        tuning_index: *tuning_index,
                    };
                }
                UiMode::Listening {
                    playing_index,
                    playing_candidate_id,
                    ..
                } => {
                    self.ui_mode = UiMode::Listening {
                        navigation_index: prev,
                        playing_index: *playing_index,
                        playing_candidate_id: playing_candidate_id.clone(),
                    };
                }
                UiMode::Idle => {}
            }
            self.adjust_scroll_to_selection(viewport_height);
        }
    }

    /// Check if "Continue scan" option is currently selected
    pub fn is_continue_scan_selected(&self) -> bool {
        if !self.selection_mode() {
            return false;
        }

        let candidate_count = self.get_selectable_candidate_count();
        self.selected_candidate_index() == Some(candidate_count)
    }

    /// Adjust scroll offset to ensure the selected candidate is visible
    pub fn adjust_scroll_to_selection(&mut self, viewport_height: usize) {
        if let Some(selected_idx) = self.selected_candidate_index() {
            // Ensure selected item is within visible range
            if selected_idx < self.scroll_offset {
                // Scrolled too far down, need to scroll up
                self.scroll_offset = selected_idx;
            } else if selected_idx >= self.scroll_offset + viewport_height {
                // Selected item is below viewport, scroll down
                self.scroll_offset = selected_idx.saturating_sub(viewport_height - 1);
            }
        }
    }

    /// Scroll up by one line
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    /// Scroll down by one line
    pub fn scroll_down(&mut self, total_candidates: usize, viewport_height: usize) {
        if self.scroll_offset + viewport_height < total_candidates {
            self.scroll_offset += 1;
        }
    }

    /// Handle arrow down navigation based on current focus state
    pub fn navigate_down(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {
                self.focus_state = FocusState::Scan;
            }
            FocusState::Scan => {
                if self.selection_mode() {
                    self.select_next_candidate();
                }
            }
            FocusState::Tuner(_) => {}
        }
    }

    /// Handle arrow up navigation based on current focus state
    pub fn navigate_up(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {
                if !self.selection_mode() {
                    // Start with most recent candidate selected (don't enter browsing mode yet)
                    let candidate_count = self.get_selectable_candidate_count();
                    if candidate_count > 0 {
                        let selected_index = candidate_count - 1;
                        // Transition: Idle → NavigatingScanner
                        self.ui_mode = UiMode::NavigatingScanner { selected_index };
                    }
                    self.focus_state = FocusState::Scan;
                } else {
                    // Try to select previous candidate (in browsing mode)
                    let prev_idx = self.selected_candidate_index();
                    self.select_previous_candidate();

                    // If selection went to None, we've gone past the first candidate
                    // Move focus to Spectrum and exit selection mode
                    if self.selected_candidate_index().is_none() && prev_idx.is_some() {
                        self.focus_state = FocusState::Spectrum;
                        self.exit_selection_mode();
                    }
                }
            }
            FocusState::Tuner(_) => {}
        }
    }

    /// Handle arrow right navigation based on current focus state
    pub fn navigate_right(&mut self, tuner_count: usize) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {
                if self.selection_mode() && tuner_count > 0 {
                    self.focus_state = FocusState::Tuner(0);
                }
            }
            FocusState::Tuner(idx) => {
                if idx + 1 < tuner_count {
                    self.focus_state = FocusState::Tuner(idx + 1);
                }
            }
        }
    }

    /// Handle arrow left navigation based on current focus state
    pub fn navigate_left(&mut self) {
        match self.focus_state {
            FocusState::Spectrum => {}
            FocusState::Scan => {}
            FocusState::Tuner(idx) => {
                if idx == 0 {
                    self.focus_state = FocusState::Scan;
                } else {
                    self.focus_state = FocusState::Tuner(idx - 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::{ProgressEvent, ProgressEventType};
    use std::time::Instant;

    /// Test that candidates progress through all expected states
    #[test]
    fn test_complete_candidate_lifecycle() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Step 1: Candidate created
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Detected);
        assert_eq!(candidate.completion, 0.3); // 30%

        // Step 2: Audio analysis started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Analyzing);
        assert_eq!(candidate.completion, 0.5); // 50%

        // Step 3: Signal generated (good signal path)
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6); // 60%

        // Step 4: Audio playback started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8); // 80%

        // Step 5: Audio playback completed
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Completed);
        assert_eq!(candidate.completion, 1.0); // 100%
    }

    /// Test that rejected candidates reach terminal state correctly
    #[test]
    fn test_rejected_candidate_lifecycle() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Step 1: Candidate created
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 2: Audio analysis started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 3: Candidate rejected (noise)
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Rejected);
        assert_eq!(candidate.completion, 1.0); // 100% - terminal state
    }

    /// Test that no candidates remain stuck in intermediate states
    #[test]
    fn test_no_stuck_intermediate_states() {
        let mut model = Model::new();
        let window_id = 1;

        // Create multiple candidates in different states
        let candidates = vec![
            ("88.1-1", 88_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("88.5-1", 88_500_000.0),
            ("88.7-1", 88_700_000.0),
            ("88.9-1", 88_900_000.0),
        ];

        // Create all candidates
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Start analysis for all
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioAnalysisStarted,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Resolve all candidates to terminal states
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[0].1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: candidates[0].1,
                window_id,
            },
            candidate_id: Some(candidates[0].0.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[1].1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: candidates[1].1,
                window_id,
            },
            candidate_id: Some(candidates[1].0.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Complete signal paths for others
        for (id, freq) in &candidates[2..] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackStarted,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Verify no candidates are stuck in intermediate states
        let window = model.windows.get(&window_id).unwrap();
        for candidate in &window.candidates {
            match candidate.status {
                CandidateStatus::Detected | CandidateStatus::Analyzing => {
                    panic!(
                        "Candidate at {:.1} MHz stuck in intermediate state: {:?}",
                        candidate.frequency_hz / 1e6,
                        candidate.status
                    );
                }
                CandidateStatus::Rejected | CandidateStatus::Completed => {
                    // Terminal states are good
                    assert_eq!(candidate.completion, 1.0);
                }
                CandidateStatus::Signal | CandidateStatus::Playing => {
                    // These are valid but should have progressed to Completed
                    panic!(
                        "Candidate at {:.1} MHz should have completed: {:?}",
                        candidate.frequency_hz / 1e6,
                        candidate.status
                    );
                }
            }
        }
    }

    /// Test that windows complete sequentially, not overlapping
    #[test]
    fn test_sequential_window_completion() {
        let mut model = Model::new();

        // Create candidates in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id: 1,
            },
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        assert_eq!(model.current_window, 1);
        assert!(!model.windows.get(&1).unwrap().is_complete);

        // Start window 2 - should mark window 1 as complete
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id: 2,
            },
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        assert_eq!(model.current_window, 2);
        assert!(model.windows.get(&1).unwrap().is_complete);
        assert!(!model.windows.get(&2).unwrap().is_complete);

        // Start window 3 - should mark window 2 as complete
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_300_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_300_000.0,
                window_id: 3,
            },
            candidate_id: Some("89.3-3".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        assert_eq!(model.current_window, 3);
        assert!(model.windows.get(&1).unwrap().is_complete);
        assert!(model.windows.get(&2).unwrap().is_complete);
        assert!(!model.windows.get(&3).unwrap().is_complete);
    }

    /// Test that old window events are ignored after window completion
    #[test]
    fn test_old_window_events_ignored() {
        let mut model = Model::new();

        // Create candidate in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id: 1,
            },
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Start window 2 (marks window 1 complete)
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id: 2,
            },
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window1_candidate_count = model.windows.get(&1).unwrap().candidates.len();

        // Try to add another candidate to completed window 1 - should be ignored
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_700_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_700_000.0,
                window_id: 1,
            },
            candidate_id: Some("88.7-1".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Window 1 should still have the same number of candidates
        assert_eq!(
            model.windows.get(&1).unwrap().candidates.len(),
            window1_candidate_count
        );

        // Try to update existing candidate in window 1 - should be ignored
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id: 1,
            },
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Candidate should still be in original state
        let candidate = &model.windows.get(&1).unwrap().candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Detected);
        assert_eq!(candidate.completion, 0.3);
    }

    /// Test window filtering behavior - only non-rejected candidates shown for complete windows
    #[test]
    fn test_window_candidate_filtering() {
        let mut model = Model::new();
        let window_id = 1;

        // Create multiple candidates
        let candidates = vec![
            ("88.1-1", 88_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("88.5-1", 88_500_000.0),
        ];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Reject first candidate, complete others
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[0].1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: candidates[0].1,
                window_id,
            },
            candidate_id: Some(candidates[0].0.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        for (id, freq) in &candidates[1..] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackStarted,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Mark window complete by starting window 2
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id: 2,
            },
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        assert!(window.is_complete);

        // For complete windows, rejected candidates are always filtered out
        // (even if it's the current window, even if not in selection mode)
        let current_displayable = window.displayable_candidates(true, false);
        assert_eq!(current_displayable.len(), 2); // Only non-rejected

        // Same for non-current complete windows
        let completed_displayable = window.displayable_candidates(false, false);
        assert_eq!(completed_displayable.len(), 2); // Only non-rejected

        // Verify the rejected candidate is filtered out
        for candidate in current_displayable {
            assert_ne!(candidate.status, CandidateStatus::Rejected);
        }
        for candidate in completed_displayable {
            assert_ne!(candidate.status, CandidateStatus::Rejected);
        }
    }

    /// Test that window should_display logic works correctly
    #[test]
    fn test_window_display_logic() {
        let mut model = Model::new();
        let window_id = 1;

        // Create window with all rejected candidates
        let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateRejected,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Mark window complete by starting window 2
        model.total_windows = Some(2);
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id: 2,
            },
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // After window 2 is created, window 1 should be marked complete
        let window = model.windows.get(&window_id).unwrap();
        assert!(window.is_complete);

        // Complete window with only rejected candidates should not display
        assert!(!window.should_display());
    }

    /// Test deterministic candidate ordering within windows
    #[test]
    fn test_deterministic_candidate_ordering() {
        let mut model = Model::new();
        let window_id = 1;

        // Create candidates in specific order
        let candidates = vec![
            ("89.1-1", 89_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("90.5-1", 90_500_000.0),
            ("87.9-1", 87_900_000.0),
        ];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        let window = model.windows.get(&window_id).unwrap();

        // Candidates should maintain insertion order
        assert_eq!(window.candidates.len(), 4);
        assert_eq!(window.candidates[0].frequency_hz, 89_100_000.0);
        assert_eq!(window.candidates[1].frequency_hz, 88_300_000.0);
        assert_eq!(window.candidates[2].frequency_hz, 90_500_000.0);
        assert_eq!(window.candidates[3].frequency_hz, 87_900_000.0);

        // displayable_candidates should also maintain this order
        let displayable = window.displayable_candidates(true, false);
        assert_eq!(displayable.len(), 4);
        assert_eq!(displayable[0].frequency_hz, 89_100_000.0);
        assert_eq!(displayable[1].frequency_hz, 88_300_000.0);
        assert_eq!(displayable[2].frequency_hz, 90_500_000.0);
        assert_eq!(displayable[3].frequency_hz, 87_900_000.0);
    }

    /// Test model utility functions
    #[test]
    fn test_model_utility_functions() {
        let mut model = Model::new();

        // Empty model - all_complete returns false for empty models
        assert!(model.is_empty());
        assert!(!model.all_complete()); // Empty model returns false for all_complete
        assert_eq!(model.candidate_count(), 0);

        // Add some candidates
        let window_id = 1;
        let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Model with incomplete candidates
        assert!(!model.is_empty());
        assert!(!model.all_complete());
        assert_eq!(model.candidate_count(), 2);

        // Complete all candidates
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateRejected,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Model with complete candidates
        assert!(!model.is_empty());
        model.total_windows = Some(1);
        assert!(model.all_complete());
        assert_eq!(model.candidate_count(), 2);
    }

    /// Test quit functionality
    #[test]
    fn test_quit_functionality() {
        let mut model = Model::new();

        assert!(!model.should_quit);

        model.quit();

        assert!(model.should_quit);
    }

    /// Test AudioAnalysisCompleted event handling preserves Signal status
    #[test]
    fn test_audio_analysis_completed_preserves_signal() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Create candidate and start analysis
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Generate signal first
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6);

        // AudioAnalysisCompleted should not override Signal status
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6); // Should remain unchanged
    }

    /// Test that status text mapping remains exactly the same
    #[test]
    fn test_status_text_mapping_unchanged() {
        // These exact strings must be preserved across refactoring
        assert_eq!(CandidateStatus::Detected.to_string(), "DETECTED");
        assert_eq!(CandidateStatus::Analyzing.to_string(), "ANALYZING");
        assert_eq!(CandidateStatus::Rejected.to_string(), "NOISE");
        assert_eq!(CandidateStatus::Signal.to_string(), "SIGNAL");
        assert_eq!(CandidateStatus::Playing.to_string(), "PLAYING");
        assert_eq!(CandidateStatus::Completed.to_string(), "DONE");
    }

    /// Test that progress percentage calculations remain exact
    #[test]
    fn test_progress_percentages_unchanged() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Test each state's exact completion percentage
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.3); // DETECTED = 30%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.5); // ANALYZING = 50%

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.6); // SIGNAL = 60%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.8); // PLAYING = 80%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 1.0); // DONE = 100%

        // Test rejected path
        let rejected_id = "89.1-1".to_string();
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id,
            },
            candidate_id: Some(rejected_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: 89_100_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 89_100_000.0,
                window_id,
            },
            candidate_id: Some(rejected_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        let window = model.windows.get(&window_id).unwrap();
        let rejected_candidate = &window.candidates[1];
        assert_eq!(rejected_candidate.completion, 1.0); // NOISE = 100%
    }

    #[test]
    fn test_browsing_mode_playing_correct_candidate() {
        let mut model = Model::new();
        let window_id = 1;

        // Create three candidates at different frequencies
        let freq1 = 88_500_000.0;
        let freq2 = 88_900_000.0;
        let freq3 = 89_300_000.0;
        let candidate1_id = "88.5-1".to_string();
        let candidate2_id = "88.9-1".to_string();
        let candidate3_id = "89.3-1".to_string();

        // Create all three candidates in Signal state
        for (freq, candidate_id) in [
            (freq1, candidate1_id.clone()),
            (freq2, candidate2_id.clone()),
            (freq3, candidate3_id.clone()),
        ] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id.clone()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id.clone()),
                audio_quality: Some(crate::audio_quality::AudioQuality::Good),
                signal_strength: Some(50.0),
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        model.current_window = window_id;

        // Verify all three candidates are in Signal state
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(window.candidates.len(), 3);
        assert_eq!(window.candidates[0].frequency_hz, freq1);
        assert_eq!(window.candidates[1].frequency_hz, freq2);
        assert_eq!(window.candidates[2].frequency_hz, freq3);
        assert_eq!(window.candidates[0].status, CandidateStatus::Signal);
        assert_eq!(window.candidates[1].status, CandidateStatus::Signal);
        assert_eq!(window.candidates[2].status, CandidateStatus::Signal);

        // Enter browsing mode and transition to AwaitingTune
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 1,
            tuning_index: 1,
        };

        // Send AudioPlaybackStarted for the middle candidate (88.9)
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify ONLY the middle candidate is Playing
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(
            window.candidates[0].status,
            CandidateStatus::Signal,
            "First candidate should still be Signal"
        );
        assert_eq!(
            window.candidates[1].status,
            CandidateStatus::Playing,
            "Second candidate should be Playing"
        );
        assert_eq!(
            window.candidates[2].status,
            CandidateStatus::Signal,
            "Third candidate should still be Signal"
        );

        // Now switch to a different candidate (89.3)
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 2 };

        // Send AudioPlaybackStarted for the third candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq3,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq3,
                window_id,
            },
            candidate_id: Some(candidate3_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify only the third candidate is Playing - the second should have been auto-completed
        let window = model.windows.get(&window_id).unwrap();
        assert_eq!(
            window.candidates[0].status,
            CandidateStatus::Signal,
            "First candidate should still be Signal"
        );
        assert_eq!(
            window.candidates[1].status,
            CandidateStatus::Completed,
            "Second candidate should be Completed (was replaced)"
        );
        assert_eq!(
            window.candidates[2].status,
            CandidateStatus::Playing,
            "Third candidate should be Playing"
        );
    }

    #[test]
    fn test_browsing_mode_allows_old_window_playback() {
        let mut model = Model::new();

        // Create candidate in window 1
        let window1_id = 1;
        let freq1 = 88_900_000.0;
        let candidate1_id = "88.9-1".to_string();

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Create candidate in window 2 (this marks window 1 as complete)
        let window2_id = 2;
        let freq2 = 89_300_000.0;
        let candidate2_id = "89.3-2".to_string();

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify we're now in window 2
        assert_eq!(model.current_window, window2_id);
        assert!(model.windows.get(&window1_id).unwrap().is_complete);

        // In normal scanning mode, events to old windows should be blocked
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Status should still be Signal (event was blocked)
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(window1.candidates[0].status, CandidateStatus::Signal);

        // Now enter browsing mode by transitioning to Navigating mode
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        // Send AudioPlaybackStarted for the old window candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // In browsing mode, AudioPlaybackStarted should work even for old windows
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "AudioPlaybackStarted should work for old windows in browsing mode"
        );
    }

    #[test]
    fn test_playing_candidates_remain_playing_when_entering_selection_mode() {
        let mut model = Model::new();

        let window_id = 1;
        let freq = 88_900_000.0;
        let candidate_id = "88.9-1".to_string();

        // Create candidate and advance to Playing state
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Set current window to match the candidate's window
        model.current_window = window_id;

        // Verify candidate is Playing
        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);

        // Enter selection mode (simulates pressing Up to browse)
        model.enter_selection_mode();

        // Verify candidate remains Playing (navigation doesn't stop playback)
        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8);
    }

    #[test]
    fn test_playing_candidates_remain_when_entering_selection_mode() {
        let mut model = Model::new();

        // Create two windows with candidates
        let window1_id = 1;
        let window2_id = 2;
        let freq1 = 88_900_000.0;
        let freq2 = 89_100_000.0;
        let candidate1_id = "88.9-1".to_string();
        let candidate2_id = "89.1-2".to_string();

        // Window 1 candidate - create and advance to Playing state
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify candidate is Playing
        let window = model.windows.get(&window1_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);

        // Window 2 candidate - create and advance to Signal state
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Moderate),
            signal_strength: Some(40.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Set current window to window 1 (where the Playing candidate is)
        model.current_window = window1_id;

        // Enter selection mode - candidates should remain in their current state
        model.enter_selection_mode();

        // Verify window 1 candidate remains Playing
        let window = model.windows.get(&window1_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8);

        // Verify window 2 candidate remains Signal
        let window = model.windows.get(&window2_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6);
    }

    #[test]
    fn test_signal_candidates_remain_signal_when_entering_selection_mode() {
        let mut model = Model::new();

        let window_id = 1;
        let freq = 88_900_000.0;
        let candidate_id = "88.9-1".to_string();

        // Create candidate and advance to Signal state
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Set current window to match the candidate's window
        model.current_window = window_id;

        // Verify candidate is Signal
        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);

        // Enter selection mode (simulates pressing Up to browse)
        model.enter_selection_mode();

        // Verify candidate remains Signal (navigation doesn't complete candidates)
        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6);
    }

    /// Regression test: Navigating between windows with arrow keys should not stop playback
    /// This tests the fix for the bug where a playing station would lose its Playing status
    /// when the user navigated to a different window or candidate using arrow keys.
    #[test]
    fn test_playing_candidate_persists_during_cross_window_navigation() {
        let mut model = Model::new();

        // Create two windows with candidates
        let window1_id = 1;
        let window2_id = 2;
        let freq1 = 88_900_000.0;
        let freq2 = 89_100_000.0;
        let candidate1_id = "88.9-1".to_string();
        let candidate2_id = "89.1-2".to_string();

        // Window 1: Create candidate and set to Playing
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: freq1,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq1,
                window_id: window1_id,
            },
            candidate_id: Some(candidate1_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Window 2: Create another candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: freq2,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: freq2,
                window_id: window2_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Moderate),
            signal_strength: Some(40.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Enter selection mode and set up selection on window 2's candidate
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 1 };

        // Verify window 1 candidate is Playing
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(window1.candidates[0].status, CandidateStatus::Playing);

        // Simulate navigating with arrow keys - move up to window 1's candidate
        model.select_previous_candidate();

        // REGRESSION TEST: Window 1 candidate should STILL be Playing after navigation
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "Playing candidate should remain Playing when navigating with arrow keys"
        );
        assert_eq!(window1.candidates[0].completion, 0.8);

        // Navigate back down to window 2
        model.select_next_candidate();

        // Window 1 candidate should STILL be Playing
        let window1 = model.windows.get(&window1_id).unwrap();
        assert_eq!(
            window1.candidates[0].status,
            CandidateStatus::Playing,
            "Playing candidate should persist across multiple navigation actions"
        );
    }

    /// Test that rejected candidates disappear from the last window when scan completes
    /// This is a regression test for the behavior where rejected candidates should
    /// disappear as soon as all candidates finish processing, not just when entering
    /// browse mode.
    #[test]
    fn test_rejected_candidates_disappear_when_scan_completes() {
        let mut model = Model::new();
        let window_id = 1;

        // Create a mix of signal and rejected candidates in the window
        let candidates = vec![
            ("88.1-1", 88_100_000.0, false), // Signal
            ("88.3-1", 88_300_000.0, true),  // Rejected
            ("88.5-1", 88_500_000.0, false), // Signal
            ("88.7-1", 88_700_000.0, true),  // Rejected
        ];

        for (id, freq, is_rejected) in &candidates {
            // Create candidate
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: *freq,
                    window_id,
                },
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            if *is_rejected {
                // Mark as rejected
                model.update(ProgressEvent {
                    event_type: ProgressEventType::CandidateRejected,
                    frequency_hz: *freq,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: *freq,
                        window_id,
                    },
                    candidate_id: Some(id.to_string()),
                    audio_quality: None,
                    signal_strength: None,
                    timestamp: Instant::now(),
                    tuner_id: None,
                });
            } else {
                // Complete as signal
                model.update(ProgressEvent {
                    event_type: ProgressEventType::SignalGenerated,
                    frequency_hz: *freq,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: *freq,
                        window_id,
                    },
                    candidate_id: Some(id.to_string()),
                    audio_quality: Some(crate::audio_quality::AudioQuality::Good),
                    signal_strength: Some(50.0),
                    timestamp: Instant::now(),
                    tuner_id: None,
                });

                model.update(ProgressEvent {
                    event_type: ProgressEventType::AudioPlaybackCompleted,
                    frequency_hz: *freq,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: *freq,
                        window_id,
                    },
                    candidate_id: Some(id.to_string()),
                    audio_quality: None,
                    signal_strength: None,
                    timestamp: Instant::now(),
                    tuner_id: None,
                });
            }
        }

        // Verify all candidates exist
        assert_eq!(model.windows.get(&window_id).unwrap().candidates.len(), 4);

        // Set total_windows and verify all_complete returns true
        model.total_windows = Some(1);

        // Verify current_window and all_candidates_complete
        assert_eq!(model.current_window, 1);
        assert!(
            model.all_candidates_complete(),
            "all_candidates_complete should be true"
        );
        assert!(model.all_complete(), "all_complete should be true");

        // Manually mark the window complete (since no more events will trigger it)
        if let Some(window) = model.windows.get_mut(&window_id) {
            window.is_complete = true;
        }

        // After manually marking complete, verify window is complete
        let window = model.windows.get(&window_id).unwrap();
        assert!(window.is_complete);

        // For a complete window, rejected candidates should be filtered out
        // even if it's the current window (is_current_window=true)
        let displayable_after_complete = window.displayable_candidates(true, false);
        assert_eq!(displayable_after_complete.len(), 2); // Only 2 signals visible

        // Verify only non-rejected candidates are shown
        for candidate in displayable_after_complete {
            assert_ne!(candidate.status, CandidateStatus::Rejected);
        }

        // In selection mode, rejected should also be filtered
        let displayable_in_selection = window.displayable_candidates(true, true);
        assert_eq!(displayable_in_selection.len(), 2); // Only 2 signals visible

        for candidate in displayable_in_selection {
            assert_ne!(candidate.status, CandidateStatus::Rejected);
        }
    }

    // UiMode State Machine Tests

    #[test]
    fn test_ui_mode_transition_idle_to_navigating() {
        let mut model = Model::new();
        assert!(matches!(model.ui_mode, UiMode::Idle));

        // Simulate pressing Up arrow (first navigation)
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        assert!(model.is_navigating());
        assert!(!model.is_idle());
    }

    #[test]
    fn test_ui_mode_transition_navigating_to_awaiting_tune() {
        let mut model = Model::new();
        model.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };

        // Simulate pressing Enter
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        assert!(model.is_awaiting_tune());
        assert!(!model.is_navigating());
    }

    #[test]
    fn test_ui_mode_transition_awaiting_tune_to_listening() {
        let mut model = Model::new();
        let window_id = 1;
        let candidate_id = "88.9-1".to_string();

        // Setup: Create a candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        // Simulate AudioPlaybackStarted event
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should transition to Listening
        assert!(model.is_listening());
        match &model.ui_mode {
            UiMode::Listening {
                playing_candidate_id,
                ..
            } => {
                assert_eq!(playing_candidate_id, &candidate_id);
            }
            _ => panic!("Expected Listening mode"),
        }
    }

    #[test]
    fn test_ui_mode_transition_listening_to_listening_switch_station() {
        let mut model = Model::new();
        let window_id = 1;

        // Create two candidates
        let candidate1_id = "88.5-1".to_string();
        let candidate2_id = "88.9-1".to_string();

        for (id, freq) in [
            (candidate1_id.clone(), 88_500_000.0),
            (candidate2_id.clone(), 88_900_000.0),
        ] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(id),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Start listening to first station
        model.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: candidate1_id.clone(),
        };

        // Switch to second station while still in Listening mode
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate2_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should still be Listening but with new candidate
        assert!(model.is_listening());
        match &model.ui_mode {
            UiMode::Listening {
                playing_candidate_id,
                navigation_index,
                ..
            } => {
                assert_eq!(playing_candidate_id, &candidate2_id);
                assert_eq!(*navigation_index, 0); // Preserves original navigation_index from Listening mode
            }
            _ => panic!("Expected Listening mode"),
        }
    }

    #[test]
    fn test_ui_mode_transition_listening_to_idle() {
        let mut model = Model::new();
        model.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: "88.9-1".to_string(),
        };

        // Simulate exiting browsing mode (Continue scan)
        model.ui_mode = UiMode::Idle;

        assert!(model.is_idle());
        assert!(!model.is_listening());
    }

    #[test]
    fn test_ui_mode_helper_methods() {
        let model_idle = Model::new();
        assert!(model_idle.is_idle());
        assert!(!model_idle.is_interactive());

        let mut model_navigating = Model::new();
        model_navigating.ui_mode = UiMode::NavigatingScanner { selected_index: 0 };
        assert!(model_navigating.is_navigating());
        assert!(model_navigating.is_interactive());

        let mut model_awaiting = Model::new();
        model_awaiting.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };
        assert!(model_awaiting.is_awaiting_tune());
        assert!(model_awaiting.is_interactive());

        let mut model_listening = Model::new();
        model_listening.ui_mode = UiMode::Listening {
            navigation_index: 0,
            playing_index: 0,
            playing_candidate_id: "88.9-1".to_string(),
        };
        assert!(model_listening.is_listening());
        assert!(model_listening.is_interactive());
    }

    #[test]
    fn test_ui_mode_invalid_transitions_prevented() {
        let mut model = Model::new();
        let window_id = 1;
        let candidate_id = "88.9-1".to_string();

        // Create candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // AudioPlaybackStarted in Idle mode - should not transition
        model.ui_mode = UiMode::Idle;

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should still be Idle (transition only happens in AwaitingTune/Listening)
        assert!(model.is_idle());
    }

    #[test]
    fn test_browsing_mode_only_true_when_scan_paused() {
        let mut model = Model::new();
        let window_id = 0;

        // Add a candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some("test-candidate".to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Idle mode - browsing_mode should be false
        assert!(model.is_idle());
        assert!(!model.browsing_mode());

        // Enter selection mode (NavigatingScanner) - browsing_mode should still be false
        model.enter_selection_mode();
        assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
        assert!(model.selection_mode());
        assert!(!model.browsing_mode()); // Key assertion: browsing_mode is false while navigating

        // Transition to AwaitingTune - browsing_mode should now be true
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }
        assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
        assert!(model.browsing_mode()); // Now true because scan is paused

        // Transition to Listening - browsing_mode should remain true
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::Listening {
                navigation_index: selected_index,
                playing_index: selected_index,
                playing_candidate_id: "test-candidate".to_string(),
            };
        }
        assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
        assert!(model.browsing_mode()); // Still true when listening
    }

    #[test]
    fn test_enter_key_tunes_to_selected_station() {
        let mut model = Model::new();
        let window_id = 0;
        let candidate_id = "test-candidate".to_string();

        // Add a Signal candidate
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: Some(0.8),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Start in Idle mode
        assert!(model.is_idle());

        // User presses UP arrow to enter selection mode (NavigatingScanner)
        model.enter_selection_mode();
        assert!(matches!(model.ui_mode, UiMode::NavigatingScanner { .. }));
        assert!(model.selection_mode());
        assert!(!model.browsing_mode()); // Not in browsing mode yet

        // User presses ENTER - should transition to AwaitingTune
        // This simulates the ENTER key handler logic
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }

        // Verify transition to AwaitingTune
        assert!(matches!(model.ui_mode, UiMode::AwaitingTune { .. }));
        assert!(model.browsing_mode()); // Now in browsing mode (scan paused)

        // Verify selected_candidate_info works in AwaitingTune mode
        let info = model.selected_candidate_info();
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.candidate_id, candidate_id);
        assert_eq!(info.candidate_frequency, 88_900_000.0);

        // Simulate receiving AudioPlaybackStarted event
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: 88_900_000.0,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: 88_900_000.0,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: Some(0.8),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Should transition to Listening mode
        assert!(matches!(model.ui_mode, UiMode::Listening { .. }));
        if let UiMode::Listening {
            playing_candidate_id,
            ..
        } = &model.ui_mode
        {
            assert_eq!(playing_candidate_id, &candidate_id);
        }
    }

    #[test]
    fn test_navigation_and_highlight_separate_in_listening_mode() {
        let mut model = Model::new();
        let window_id = 0;

        // Add three candidates
        for i in 0..3 {
            let freq = 88_100_000.0 + (i as f64 * 200_000.0); // 88.1, 88.3, 88.5 MHz
            let candidate_id = format!("candidate_{}", i);

            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id.clone()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: freq,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: freq,
                    window_id,
                },
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: Some(0.8),
                timestamp: Instant::now(),
                tuner_id: None,
            });
        }

        // Enter selection mode and select first candidate (index 0)
        model.enter_selection_mode();
        assert_eq!(model.selected_candidate_index(), Some(2)); // Most recent

        // Move to first candidate
        model.select_previous_candidate();
        model.select_previous_candidate();
        assert_eq!(model.selected_candidate_index(), Some(0));

        // Press ENTER on first candidate - transition to AwaitingTune
        model.ui_mode = UiMode::AwaitingTune {
            navigation_index: 0,
            tuning_index: 0,
        };

        // Verify we're tuning to index 0
        if let UiMode::AwaitingTune {
            navigation_index,
            tuning_index,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 0);
            assert_eq!(*tuning_index, 0);
        }

        // Arrow down to navigate to second candidate
        model.select_next_candidate();

        // Verify navigation moved but tuning index stayed the same
        if let UiMode::AwaitingTune {
            navigation_index,
            tuning_index,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 1, "Navigation should move to index 1");
            assert_eq!(*tuning_index, 0, "Tuning should stay at index 0");
        } else {
            panic!("Should still be in AwaitingTune mode");
        }

        // Transition to Listening mode
        model.ui_mode = UiMode::Listening {
            navigation_index: 1,
            playing_index: 0,
            playing_candidate_id: "candidate_0".to_string(),
        };

        // Arrow down again to third candidate
        model.select_next_candidate();

        // Verify navigation moved but playing index stayed the same
        if let UiMode::Listening {
            navigation_index,
            playing_index,
            playing_candidate_id,
        } = &model.ui_mode
        {
            assert_eq!(*navigation_index, 2, "Navigation should move to index 2");
            assert_eq!(*playing_index, 0, "Playing should stay at index 0");
            assert_eq!(playing_candidate_id, "candidate_0");
        } else {
            panic!("Should still be in Listening mode");
        }

        // Arrow up back to second candidate
        model.select_previous_candidate();

        // Verify navigation moved back but playing index still unchanged
        if let UiMode::Listening {
            navigation_index,
            playing_index,
            ..
        } = &model.ui_mode
        {
            assert_eq!(
                *navigation_index, 1,
                "Navigation should move back to index 1"
            );
            assert_eq!(*playing_index, 0, "Playing should still be at index 0");
        }
    }

    #[test]
    fn test_stop_listening_transitions_candidate_to_completed() {
        let mut model = Model::default();
        let window_id = 1;
        let frequency = 88_900_000.0;
        let candidate_id = format!("{:.1}-{}", frequency / 1e6, window_id);

        // Step 1: Create candidate in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 2: Complete audio analysis
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 3: Generate signal
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Step 4: Pause scanning and enter interactive mode
        model.enter_selection_mode();
        if let Some(selected_index) = model.selected_candidate_index() {
            model.ui_mode = UiMode::AwaitingTune {
                navigation_index: selected_index,
                tuning_index: selected_index,
            };
        }
        assert!(model.browsing_mode());

        // Step 5: Start playing audio from window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id,
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify candidate is in Playing state
        let window = model.windows.get(&window_id).unwrap();
        let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
        let candidate = &window.candidates[*candidate_index];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8);

        // Step 6: Simulate scanning having progressed to window 2 (making window 1 an "old window")
        // This tests the "old window" filtering bug where AudioPlaybackCompleted was rejected
        // In a real scenario, this could happen if scanning resumed briefly or if there are
        // multiple tuners scanning while one is listening
        model.current_window = 2;

        // Verify current_window has advanced to 2
        assert_eq!(model.current_window, 2);

        // Verify we're still in interactive mode
        assert!(model.is_interactive());

        // Step 7: Stop listening to the station from window 1 (now an "old window")
        // Regression test for TWO bugs:
        // 1. AudioPlaybackCompleted was filtered out in interactive mode by should_process_event()
        // 2. AudioPlaybackCompleted was filtered out for old windows by update_candidate()
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            metadata: crate::window::WindowMetadata {
                center_frequency_hz: frequency,
                window_id, // window 1 is now "old" since current_window is 2
            },
            candidate_id: Some(candidate_id.clone()),
            audio_quality: Some(crate::audio_quality::AudioQuality::Good),
            signal_strength: Some(50.0),
            timestamp: Instant::now(),
            tuner_id: None,
        });

        // Verify candidate transitioned to Completed state despite being in an old window
        let window = model.windows.get(&window_id).unwrap();
        let candidate_index = window.candidate_lookup.get(&candidate_id).unwrap();
        let candidate = &window.candidates[*candidate_index];
        assert_eq!(
            candidate.status,
            CandidateStatus::Completed,
            "Candidate should transition to Completed when AudioPlaybackCompleted is sent, \
             even when in interactive mode (bug #1) and from an old window (bug #2)"
        );
        assert_eq!(candidate.completion, 1.0);
    }

    #[test]
    fn test_only_used_tuner_shows_scanning_state() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery service finds RTL-SDR first (alphabetically or by enumeration order)
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        // RTL-SDR should be Available, not Scanning
        assert_eq!(
            model.tuner_states.get(&rtlsdr_tuner.id),
            Some(&TunerState::Available),
            "First discovered tuner should be Available, not auto-set to Scanning"
        );

        // Discovery service then finds SDRplay
        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available
        assert_eq!(
            model.tuner_states.get(&sdrplay_tuner.id),
            Some(&TunerState::Available)
        );
        assert_eq!(
            model.tuner_states.get(&rtlsdr_tuner.id),
            Some(&TunerState::Available)
        );

        // MainThread starts scan with SDRplay - sends ActiveTunersUpdated event
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // SDRplay should now be Scanning
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "SDRplay should be Scanning when MainThread allocated it for scanning"
        );

        // RTL-SDR should still be Available (regression test for incorrect auto-scanning)
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not in active tuners"
        );

        // Scan continues - active tuners remain unchanged
        // Progress events no longer affect tuner state

        // SDRplay should still be Scanning
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

        // RTL-SDR should STILL be Available
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should never transition to Scanning since it's not in active tuners"
        );
    }

    #[test]
    fn test_only_used_tuner_shows_listening_state() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds both tuners
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available initially
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread starts scan with SDRplay
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // SDRplay is now Scanning
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

        // User presses Enter to tune to the candidate - MainThread moves tuner to listening
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![],
            listening: vec![sdrplay_tuner.id.clone()],
        });

        // SDRplay should transition to Listening
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Listening,
            "SDRplay should be Listening when MainThread allocated it to listening"
        );

        // RTL-SDR should still be Available (regression test for incorrect listening state)
        // The bug was: update_candidate() set self.tuners.first() to Listening
        // instead of using event.tuner_id
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not in active tuners"
        );

        // Stop listening doesn't change active tuners
        // (MainThread would send new ActiveTunersUpdated when user presses Escape)
        // For this test, we're just verifying state stays as-is

        // SDRplay remains in Listening state (still allocated to listening)
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Listening,
            "SDRplay remains Listening until MainThread reallocates it"
        );

        // RTL-SDR should STILL be Available throughout
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should never transition to Listening since it's not in active tuners"
        );
    }

    #[test]
    fn test_tuner_stays_scanning_during_automatic_audio_playback() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds SDRplay tuner
        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Should be Available initially
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread allocates SDRplay for scanning
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // SDRplay is now Scanning
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should be Scanning when MainThread allocated it for scanning"
        );

        // Model is still in Idle mode (not AwaitingTune) - user has NOT pressed Enter
        assert!(matches!(model.ui_mode, UiMode::Idle));

        // During scanning, audio playback starts automatically for quality analysis
        // Even though audio is playing, MainThread keeps the tuner in scanning list
        // because user has not pressed Enter (no TuneToCandidate command sent)

        // MainThread continues to report tuner as scanning during automatic playback
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // The tuner should remain in Scanning state during automatic audio playback
        // Only when user presses Enter (sends TuneToCandidate) should it go to Listening
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should remain Scanning during automatic audio playback (user has not pressed Enter)"
        );

        // Audio playback completes automatically, tuner still scanning
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // Should still be Scanning after automatic playback completes
        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "Tuner should remain Scanning after automatic audio playback completes"
        );
    }

    #[test]
    fn test_correct_tuner_shows_scanning_when_returning_from_listening() {
        use crate::sdr::DeviceId;

        let mut model = Model::default();

        // Discovery finds both tuners (RTL-SDR first, SDRplay second)
        let rtlsdr_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("rtlsdr", "00000001"),
            label: "Generic RTL-SDR".to_string(),
        };
        model.add_device(rtlsdr_tuner.clone());

        let sdrplay_tuner = crate::sdr::DeviceInfo {
            id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
            label: "SDRplay RSPduo".to_string(),
        };
        model.add_device(sdrplay_tuner.clone());

        // Both should be Available initially
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

        // MainThread allocates SDRplay for scanning (not RTL-SDR)
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // SDRplay should be Scanning, RTL-SDR should remain Available
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

        // User presses Enter to listen - MainThread moves SDRplay to listening list
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![],
            listening: vec![sdrplay_tuner.id.clone()],
        });

        // SDRplay should be Listening, RTL-SDR should remain Available
        assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Listening);
        assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

        // User presses Escape to go back to scanning
        // MainThread moves SDRplay back to scanning list
        model.update_tui_event(crate::terminal::TuiEvent::ActiveTunersUpdated {
            available: vec![sdrplay_tuner.id.clone()],
            scanning: vec![sdrplay_tuner.id.clone()],
            listening: vec![],
        });

        // After returning from listening to scanning, only SDRplay should be Scanning
        // RTL-SDR should remain Available (never used)
        assert_eq!(
            model.tuner_state(&rtlsdr_tuner.id),
            TunerState::Available,
            "RTL-SDR should remain Available since it's not being used"
        );

        assert_eq!(
            model.tuner_state(&sdrplay_tuner.id),
            TunerState::Scanning,
            "SDRplay should transition back to Scanning when MainThread returns it to scanning list"
        );

        // Verify that exactly one tuner is in Scanning state by checking active_tuners
        if let Some(ref active) = model.active_tuners {
            assert_eq!(
                active.scanning.len(),
                1,
                "Exactly one tuner should be in scanning list"
            );
            assert_eq!(
                active.scanning[0], sdrplay_tuner.id,
                "Only SDRplay should be in scanning list"
            );
        } else {
            panic!("active_tuners should be set");
        }
    }
}
