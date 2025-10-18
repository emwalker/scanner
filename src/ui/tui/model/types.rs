//! Type definitions for TUI model

use std::{collections::HashMap, time::Instant};

/// Selected candidate information
#[derive(Debug, Clone)]
pub struct SelectedCandidateInfo {
    pub candidate_id: String,
    pub metadata: crate::scanning::window::WindowMetadata,
    pub candidate_frequency: f64,
    pub signal_strength: Option<f64>,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
}

/// Information about a candidate's progress
#[derive(Debug, Clone)]
pub struct CandidateProgress {
    pub candidate_id: String,
    pub frequency_hz: f64,
    pub metadata: crate::scanning::window::WindowMetadata,
    pub completion: f64,
    pub status: CandidateStatus,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
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

/// View model for tuner display state
/// Makes the state-to-label mapping explicit and testable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TunerDisplayState {
    pub tuner_id: crate::hardware::pool::TunerId,
    pub label: String,
    pub status_label: &'static str,
}

/// Tuner information for flat list display
#[derive(Debug, Clone)]
pub struct TunerDisplayInfo {
    pub id: crate::hardware::pool::TunerId,
    pub label: String,
    pub state: TunerState,
}

/// Station marker for spectrum display
#[derive(Debug, Clone)]
pub struct SpectrumStation {
    pub frequency_hz: f64,
    pub signal_strength: f32,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
    pub is_active: bool,
}
