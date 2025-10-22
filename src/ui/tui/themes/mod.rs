//! Theming system for the TUI interface
//!
//! This module provides a trait-based theming system that supports multiple
//! visual themes while preserving all functional behavior of the scanner.

use ratatui::style::Color;

/// Shared text configuration for all themes
pub struct SharedText;

impl SharedText {
    pub fn title() -> &'static str {
        "SPECTRUM SCANNER"
    }

    pub fn subtitle() -> &'static str {
        "FM • 88–108 MHz"
    }

    pub fn candidates_label() -> &'static str {
        "Candidates"
    }

    pub fn stations_label() -> &'static str {
        "Stations"
    }

    pub fn status_detected() -> &'static str {
        "Detected"
    }

    pub fn status_analyzing() -> &'static str {
        "Analyzing"
    }

    pub fn status_rejected() -> &'static str {
        "Skipped"
    }

    pub fn status_signal() -> &'static str {
        "Queued"
    }

    pub fn status_playing() -> &'static str {
        "Listening"
    }

    pub fn status_completed() -> &'static str {
        "Signal"
    }

    pub fn quality_good() -> &'static str {
        "good"
    }

    pub fn quality_moderate() -> &'static str {
        "moderate"
    }

    pub fn quality_poor() -> &'static str {
        "poor"
    }

    pub fn quality_no_audio() -> &'static str {
        "no-audio"
    }

    pub fn quality_static() -> &'static str {
        "static"
    }

    pub fn quality_unknown() -> &'static str {
        "unknown"
    }
}

/// Color scheme trait defining all colors used in the UI
pub trait ColorScheme {
    // Core colors
    fn primary(&self) -> Color;
    fn secondary(&self) -> Color;
    fn accent(&self) -> Color;
    fn background(&self) -> Color;
    fn foreground(&self) -> Color;

    // Status colors (preserving exact state machine behavior)
    fn status_detected(&self) -> Color;
    fn status_analyzing(&self) -> Color;
    fn status_rejected(&self) -> Color;
    fn status_signal(&self) -> Color;
    fn status_playing(&self) -> Color;
    fn status_completed(&self) -> Color;

    // Audio quality colors
    fn quality_good(&self) -> Color;
    fn quality_moderate(&self) -> Color;
    fn quality_poor(&self) -> Color;
    fn quality_no_audio(&self) -> Color;
    fn quality_static(&self) -> Color;
    fn quality_unknown(&self) -> Color;

    // UI element colors
    fn header_accent(&self) -> Color;
    fn spectrum_window(&self) -> Color;
    fn instructions_dim(&self) -> Color;
    fn window_header(&self) -> Color;

    // Selection highlight color
    fn selection_highlight(&self) -> Color;

    // Active playing/tuning highlight colors
    fn active_highlight_bg(&self) -> Color;
    fn active_highlight_fg(&self) -> Color;
    fn active_highlight_status(&self) -> Color;
    fn active_highlight_quality(&self) -> Color;
}

/// Symbol set trait defining all Unicode symbols used in the UI
pub trait SymbolSet {
    // Status symbols (preserving exact visual indicators)
    fn symbol_detected(&self) -> &'static str;
    fn symbol_analyzing(&self) -> &'static str;
    fn symbol_rejected(&self) -> &'static str;
    fn symbol_signal(&self) -> &'static str;
    fn symbol_playing(&self) -> &'static str;
    fn symbol_completed(&self) -> &'static str;

    // Progress bar characters
    fn progress_empty(&self) -> &'static str;
    fn progress_full(&self) -> &'static str;

    // Spectrum visualization
    fn spectrum_baseline(&self) -> char;
    fn spectrum_window_char(&self) -> char;

    // Window header decoration
    fn window_bullet(&self) -> &'static str;

    // Header border
    fn header_border(&self) -> char;

    // Selection indicator
    fn selection_indicator(&self) -> &'static str;
}

/// Text style trait defining terminology and formatting
pub trait TextStyle {
    // Header text
    fn title(&self) -> &'static str {
        SharedText::title()
    }
    fn subtitle(&self) -> &'static str {
        SharedText::subtitle()
    }

    // Status text (preserving exact terminology)
    fn status_detected_text(&self) -> &'static str {
        SharedText::status_detected()
    }
    fn status_analyzing_text(&self) -> &'static str {
        SharedText::status_analyzing()
    }
    fn status_rejected_text(&self) -> &'static str {
        SharedText::status_rejected()
    }
    fn status_signal_text(&self) -> &'static str {
        SharedText::status_signal()
    }
    fn status_playing_text(&self) -> &'static str {
        SharedText::status_playing()
    }
    fn status_completed_text(&self) -> &'static str {
        SharedText::status_completed()
    }

    // Audio quality text
    fn quality_good_text(&self) -> &'static str {
        SharedText::quality_good()
    }
    fn quality_moderate_text(&self) -> &'static str {
        SharedText::quality_moderate()
    }
    fn quality_poor_text(&self) -> &'static str {
        SharedText::quality_poor()
    }
    fn quality_no_audio_text(&self) -> &'static str {
        SharedText::quality_no_audio()
    }
    fn quality_static_text(&self) -> &'static str {
        SharedText::quality_static()
    }
    fn quality_unknown_text(&self) -> &'static str {
        SharedText::quality_unknown()
    }
}

/// Combined theme trait that provides all theming aspects
pub trait Theme: ColorScheme + SymbolSet + TextStyle + Send + Sync {
    fn name(&self) -> &str;
    fn is_dark(&self) -> bool;
}

/// Theme name enumeration for CLI argument parsing
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThemeName {
    BasicDark,
    BasicLight,
    BladerunnerDark,
    BladerunnerLight,
    InterstellarDark,
    InterstellarLight,
    DuneDark,
    DuneLight,
    TransportDark,
    TransportLight,
    ArchiveDark,
    ArchiveLight,
    MinimalDark,
    MinimalLight,
    ImperialDark,
    ImperialLight,
    CaladanDark,
    CaladanLight,
}

impl std::str::FromStr for ThemeName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "basic-dark" => Ok(ThemeName::BasicDark),
            "basic-light" => Ok(ThemeName::BasicLight),
            "bladerunner-dark" => Ok(ThemeName::BladerunnerDark),
            "bladerunner-light" => Ok(ThemeName::BladerunnerLight),
            "interstellar-dark" => Ok(ThemeName::InterstellarDark),
            "interstellar-light" => Ok(ThemeName::InterstellarLight),
            "dune-dark" => Ok(ThemeName::DuneDark),
            "dune-light" => Ok(ThemeName::DuneLight),
            "transport-dark" => Ok(ThemeName::TransportDark),
            "transport-light" => Ok(ThemeName::TransportLight),
            "archive-dark" => Ok(ThemeName::ArchiveDark),
            "archive-light" => Ok(ThemeName::ArchiveLight),
            "minimal-dark" => Ok(ThemeName::MinimalDark),
            "minimal-light" => Ok(ThemeName::MinimalLight),
            "imperial-dark" => Ok(ThemeName::ImperialDark),
            "imperial-light" => Ok(ThemeName::ImperialLight),
            "caladan-dark" => Ok(ThemeName::CaladanDark),
            "caladan-light" => Ok(ThemeName::CaladanLight),
            _ => Err(format!("Unknown theme: {}", s)),
        }
    }
}

impl std::fmt::Display for ThemeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThemeName::BasicDark => write!(f, "basic-dark"),
            ThemeName::BasicLight => write!(f, "basic-light"),
            ThemeName::BladerunnerDark => write!(f, "bladerunner-dark"),
            ThemeName::BladerunnerLight => write!(f, "bladerunner-light"),
            ThemeName::InterstellarDark => write!(f, "interstellar-dark"),
            ThemeName::InterstellarLight => write!(f, "interstellar-light"),
            ThemeName::DuneDark => write!(f, "dune-dark"),
            ThemeName::DuneLight => write!(f, "dune-light"),
            ThemeName::TransportDark => write!(f, "transport-dark"),
            ThemeName::TransportLight => write!(f, "transport-light"),
            ThemeName::ArchiveDark => write!(f, "archive-dark"),
            ThemeName::ArchiveLight => write!(f, "archive-light"),
            ThemeName::MinimalDark => write!(f, "minimal-dark"),
            ThemeName::MinimalLight => write!(f, "minimal-light"),
            ThemeName::ImperialDark => write!(f, "imperial-dark"),
            ThemeName::ImperialLight => write!(f, "imperial-light"),
            ThemeName::CaladanDark => write!(f, "caladan-dark"),
            ThemeName::CaladanLight => write!(f, "caladan-light"),
        }
    }
}

/// Theme factory function
pub fn create_theme(theme_name: &ThemeName) -> Box<dyn Theme> {
    match theme_name {
        ThemeName::BasicDark => Box::new(crate::ui::tui::themes::basic::BasicDarkTheme),
        ThemeName::BasicLight => Box::new(crate::ui::tui::themes::basic::BasicLightTheme),
        ThemeName::BladerunnerDark => {
            Box::new(crate::ui::tui::themes::bladerunner::BladerunnerDarkTheme)
        }
        ThemeName::BladerunnerLight => {
            Box::new(crate::ui::tui::themes::bladerunner::BladerunnerLightTheme)
        }
        ThemeName::InterstellarDark => {
            Box::new(crate::ui::tui::themes::interstellar::InterstellarDarkTheme)
        }
        ThemeName::InterstellarLight => {
            Box::new(crate::ui::tui::themes::interstellar::InterstellarLightTheme)
        }
        ThemeName::DuneDark => Box::new(crate::ui::tui::themes::dune::DuneDarkTheme),
        ThemeName::DuneLight => Box::new(crate::ui::tui::themes::dune::DuneLightTheme),
        ThemeName::TransportDark => Box::new(crate::ui::tui::themes::transport::TransportDarkTheme),
        ThemeName::TransportLight => {
            Box::new(crate::ui::tui::themes::transport::TransportLightTheme)
        }
        ThemeName::ArchiveDark => Box::new(crate::ui::tui::themes::archive::ArchiveDarkTheme),
        ThemeName::ArchiveLight => Box::new(crate::ui::tui::themes::archive::ArchiveLightTheme),
        ThemeName::MinimalDark => Box::new(crate::ui::tui::themes::minimal::DarkTheme),
        ThemeName::MinimalLight => Box::new(crate::ui::tui::themes::minimal::LightTheme),
        ThemeName::ImperialDark => Box::new(crate::ui::tui::themes::imperial::DarkTheme),
        ThemeName::ImperialLight => Box::new(crate::ui::tui::themes::imperial::LightTheme),
        ThemeName::CaladanDark => Box::new(crate::ui::tui::themes::caladan::DarkTheme),
        ThemeName::CaladanLight => Box::new(crate::ui::tui::themes::caladan::LightTheme),
    }
}

pub mod archive;
pub mod basic;
pub mod bladerunner;
pub mod caladan;
pub mod dune;
pub mod imperial;
pub mod interstellar;
pub mod minimal;
pub mod transport;

impl ThemeName {
    pub fn display_name(&self) -> &'static str {
        match self {
            ThemeName::BasicDark => "Basic (Dark)",
            ThemeName::BasicLight => "Basic (Light)",
            ThemeName::BladerunnerDark => "Blade Runner (Dark)",
            ThemeName::BladerunnerLight => "Blade Runner (Light)",
            ThemeName::InterstellarDark => "Interstellar (Dark)",
            ThemeName::InterstellarLight => "Interstellar (Light)",
            ThemeName::DuneDark => "Dune (Dark)",
            ThemeName::DuneLight => "Dune (Light)",
            ThemeName::TransportDark => "Transport (Dark)",
            ThemeName::TransportLight => "Transport (Light)",
            ThemeName::ArchiveDark => "Archive (Dark)",
            ThemeName::ArchiveLight => "Archive (Light)",
            ThemeName::MinimalDark => "Minimal (Dark)",
            ThemeName::MinimalLight => "Minimal (Light)",
            ThemeName::ImperialDark => "Imperial (Dark)",
            ThemeName::ImperialLight => "Imperial (Light)",
            ThemeName::CaladanDark => "Caladan (Dark)",
            ThemeName::CaladanLight => "Caladan (Light)",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            ThemeName::BasicDark,
            ThemeName::BasicLight,
            ThemeName::BladerunnerDark,
            ThemeName::BladerunnerLight,
            ThemeName::InterstellarDark,
            ThemeName::InterstellarLight,
            ThemeName::DuneDark,
            ThemeName::DuneLight,
            ThemeName::TransportDark,
            ThemeName::TransportLight,
            ThemeName::ArchiveDark,
            ThemeName::ArchiveLight,
            ThemeName::MinimalDark,
            ThemeName::MinimalLight,
            ThemeName::ImperialDark,
            ThemeName::ImperialLight,
            ThemeName::CaladanDark,
            ThemeName::CaladanLight,
        ]
    }

    /// Get the next theme in the cycle
    pub fn next(&self) -> Self {
        match self {
            ThemeName::BasicDark => ThemeName::BasicLight,
            ThemeName::BasicLight => ThemeName::BladerunnerDark,
            ThemeName::BladerunnerDark => ThemeName::BladerunnerLight,
            ThemeName::BladerunnerLight => ThemeName::InterstellarDark,
            ThemeName::InterstellarDark => ThemeName::InterstellarLight,
            ThemeName::InterstellarLight => ThemeName::DuneDark,
            ThemeName::DuneDark => ThemeName::DuneLight,
            ThemeName::DuneLight => ThemeName::TransportDark,
            ThemeName::TransportDark => ThemeName::TransportLight,
            ThemeName::TransportLight => ThemeName::ArchiveDark,
            ThemeName::ArchiveDark => ThemeName::ArchiveLight,
            ThemeName::ArchiveLight => ThemeName::MinimalDark,
            ThemeName::MinimalDark => ThemeName::MinimalLight,
            ThemeName::MinimalLight => ThemeName::ImperialDark,
            ThemeName::ImperialDark => ThemeName::ImperialLight,
            ThemeName::ImperialLight => ThemeName::CaladanDark,
            ThemeName::CaladanDark => ThemeName::CaladanLight,
            ThemeName::CaladanLight => ThemeName::BasicDark,
        }
    }
}
