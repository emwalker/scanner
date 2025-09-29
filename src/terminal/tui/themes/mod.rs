//! Theming system for the TUI interface
//!
//! This module provides a trait-based theming system that supports multiple
//! visual themes while preserving all functional behavior of the scanner.

use ratatui::style::Color;

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
}

/// Text style trait defining terminology and formatting
pub trait TextStyle {
    // Header text
    fn title(&self) -> &'static str;
    fn subtitle(&self) -> &'static str;

    // Status text (preserving exact terminology)
    fn status_detected_text(&self) -> &'static str;
    fn status_analyzing_text(&self) -> &'static str;
    fn status_rejected_text(&self) -> &'static str;
    fn status_signal_text(&self) -> &'static str;
    fn status_playing_text(&self) -> &'static str;
    fn status_completed_text(&self) -> &'static str;
}

/// Combined theme trait that provides all theming aspects
pub trait Theme: ColorScheme + SymbolSet + TextStyle + Send + Sync {
    fn name(&self) -> &str;
    fn is_dark(&self) -> bool;
}

/// Theme name enumeration for CLI argument parsing
#[derive(Clone, Debug)]
pub enum ThemeName {
    BasicDark,
    BasicLight,
    BladerunnerDark,
    BladerunnerLight,
    InterstellarDark,
    InterstellarLight,
    DuneDark,
    DuneLight,
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
        }
    }
}

/// Theme factory function
pub fn create_theme(theme_name: &ThemeName) -> Box<dyn Theme> {
    match theme_name {
        ThemeName::BasicDark => Box::new(crate::terminal::tui::themes::basic::BasicDarkTheme),
        ThemeName::BasicLight => Box::new(crate::terminal::tui::themes::basic::BasicLightTheme),
        ThemeName::BladerunnerDark => {
            Box::new(crate::terminal::tui::themes::bladerunner::BladerunnerDarkTheme)
        }
        ThemeName::BladerunnerLight => {
            Box::new(crate::terminal::tui::themes::bladerunner::BladerunnerLightTheme)
        }
        ThemeName::InterstellarDark => {
            Box::new(crate::terminal::tui::themes::interstellar::InterstellarDarkTheme)
        }
        ThemeName::InterstellarLight => {
            Box::new(crate::terminal::tui::themes::interstellar::InterstellarLightTheme)
        }
        ThemeName::DuneDark => Box::new(crate::terminal::tui::themes::dune::DuneDarkTheme),
        ThemeName::DuneLight => Box::new(crate::terminal::tui::themes::dune::DuneLightTheme),
    }
}

pub mod basic;
pub mod bladerunner;
pub mod dune;
pub mod interstellar;
