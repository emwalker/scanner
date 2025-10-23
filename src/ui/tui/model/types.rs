//! Type definitions for TUI model

use std::{collections::HashMap, time::Instant};

use crate::ui::tui::renderers::table_styles::{RowGroup, TableRow};

/// Selected signal information
#[derive(Debug, Clone)]
pub struct SelectedSignalInfo {
    pub signal_id: String,
    pub window_id: usize,
    pub center_frequency_hz: f64,
    pub signal_frequency: f64,
    pub signal_strength: Option<f64>,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
}

/// Information about a signal's progress
#[derive(Debug, Clone)]
pub struct SignalProgress {
    pub signal_id: String,
    pub frequency_hz: f64,
    pub window_id: usize,
    pub center_frequency_hz: f64,
    pub completion: f64,
    pub status: AnalysisStatus,
    pub playback_state: PlaybackState,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
    pub signal_strength: Option<f64>,
    pub last_update: Instant,
}

/// Information about a scanning window
#[derive(Debug, Clone)]
pub struct WindowProgress {
    #[allow(dead_code)] // Kept for debugging and potential future use
    pub window_id: usize,
    pub signals: Vec<SignalProgress>,
    pub is_complete: bool,
    pub signal_lookup: HashMap<String, usize>, // signal_id -> index in signals vec
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisStatus {
    Detected,
    Analyzing,
    Rejected,
    Signal,
    Error,
}

impl AnalysisStatus {
    pub fn to_string(&self) -> &'static str {
        match self {
            AnalysisStatus::Detected => "Detected",
            AnalysisStatus::Analyzing => "Analyzing",
            AnalysisStatus::Rejected => "Rejected",
            AnalysisStatus::Signal => "Signal",
            AnalysisStatus::Error => "Error",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlaybackState {
    NotPlaying,
    Playing,
    Completed,
}

impl PlaybackState {
    pub fn to_string(&self) -> &'static str {
        match self {
            PlaybackState::NotPlaying => "",
            PlaybackState::Playing => "Playing",
            PlaybackState::Completed => "Completed",
        }
    }
}

/// Which table currently has focus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedTable {
    Activities,
    Tuners,
    ScanProgress,
}

/// Focus state for table navigation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusState {
    Activities(usize),   // Index of selected row in Activities table
    TunersTable(usize),  // Index of selected row in Tuners table
    ScanProgress(usize), // Index of selected row in ScanProgress table
}

impl FocusState {
    pub fn focused_table(&self) -> FocusedTable {
        match self {
            FocusState::Activities(_) => FocusedTable::Activities,
            FocusState::TunersTable(_) => FocusedTable::Tuners,
            FocusState::ScanProgress(_) => FocusedTable::ScanProgress,
        }
    }
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
    /// Watching scan progress (no signal selected)
    Idle,

    /// signal selected, navigating scanner results while scan may still be running
    NavigatingScanner {
        signal_index: usize, // Stable index into full flattened list
        window_id: usize,    // Window this signal belongs to
    },

    /// Scan paused, waiting for Paused event before tuning to station
    AwaitingTune {
        signal_index: usize, // Stable index into full flattened list
        window_id: usize,
        tuning_signal_id: String, // ID of signal being tuned
    },

    /// Actively listening to a station (scan paused, audio playing)
    Listening {
        signal_index: usize, // Stable index for navigation
        window_id: usize,
        playing_signal_id: String, // ID of playing signal
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
    pub is_playing: bool,
}

impl TableRow for TunerDisplayInfo {
    fn build_cells(
        &self,
        _theme: &dyn crate::ui::tui::themes::Theme,
    ) -> Vec<ratatui::widgets::Cell<'static>> {
        use ratatui::widgets::Cell;

        vec![
            Cell::from(self.label.clone()),
            Cell::from(self.state.display()),
        ]
    }

    fn special_style(
        &self,
        theme: &dyn crate::ui::tui::themes::Theme,
    ) -> Option<ratatui::style::Style> {
        use ratatui::style::Style;

        if self.is_playing {
            Some(
                Style::default()
                    .bg(theme.active_highlight_bg())
                    .fg(theme.active_highlight_fg()),
            )
        } else {
            None
        }
    }
}

impl RowGroup for TunerDisplayInfo {
    type GroupId = ();

    fn group_id(&self) -> Self::GroupId {}
}

/// Station marker for spectrum display
#[derive(Debug, Clone)]
pub struct SpectrumStation {
    pub frequency_hz: f64,
    pub signal_strength: f32,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
    pub is_active: bool,
}

/// Flattened signal row for rendering and filtering
#[derive(Clone)]
pub struct SignalRow {
    pub window_id: usize,
    pub frequency_hz: f64,
    pub status: AnalysisStatus,
    pub playback_state: PlaybackState,
    pub audio_quality: Option<crate::audio::quality::AudioQuality>,
    pub is_window_complete: bool,
    pub completion: f64,
}

impl RowGroup for SignalRow {
    type GroupId = usize;

    fn group_id(&self) -> Self::GroupId {
        self.window_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_mode_navigating_contains_window_id() {
        let ui_mode = UiMode::NavigatingScanner {
            signal_index: 5,
            window_id: 2,
        };

        match ui_mode {
            UiMode::NavigatingScanner {
                signal_index,
                window_id,
            } => {
                assert_eq!(signal_index, 5);
                assert_eq!(window_id, 2);
            }
            _ => panic!("Expected NavigatingScanner variant"),
        }
    }

    #[test]
    fn test_ui_mode_awaiting_tune_uses_signal_id() {
        let ui_mode = UiMode::AwaitingTune {
            signal_index: 3,
            window_id: 1,
            tuning_signal_id: "test_signal".to_string(),
        };

        match ui_mode {
            UiMode::AwaitingTune {
                signal_index,
                window_id,
                tuning_signal_id,
            } => {
                assert_eq!(signal_index, 3);
                assert_eq!(window_id, 1);
                assert_eq!(tuning_signal_id, "test_signal");
            }
            _ => panic!("Expected AwaitingTune variant"),
        }
    }

    #[test]
    fn test_ui_mode_listening_structure() {
        let ui_mode = UiMode::Listening {
            signal_index: 7,
            window_id: 3,
            playing_signal_id: "playing_signal".to_string(),
        };

        match ui_mode {
            UiMode::Listening {
                signal_index,
                window_id,
                playing_signal_id,
            } => {
                assert_eq!(signal_index, 7);
                assert_eq!(window_id, 3);
                assert_eq!(playing_signal_id, "playing_signal");
            }
            _ => panic!("Expected Listening variant"),
        }
    }
}
