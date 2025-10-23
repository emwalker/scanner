//! Interstellar theme implementations - atmospheric amber aesthetic

use ratatui::style::Color;

use super::{ColorScheme, SymbolSet, TextStyle, Theme};

/// Interstellar dark theme - atmospheric amber monochrome
pub struct InterstellarDarkTheme;

impl ColorScheme for InterstellarDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn secondary(&self) -> Color {
        Color::Rgb(255, 165, 0) // Warm amber
    }

    fn accent(&self) -> Color {
        Color::Rgb(255, 140, 0) // Deep amber
    }

    fn background(&self) -> Color {
        Color::Reset // Terminal default (typically black)
    }

    fn foreground(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    // Status colors - atmospheric amber hierarchy
    fn status_detected(&self) -> Color {
        Color::Rgb(255, 165, 0) // Warm amber
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(139, 100, 35) // Dim amber
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(255, 140, 0) // Deep amber
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(200, 140, 60) // Medium amber
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(255, 165, 0) // Warm amber
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(200, 140, 60) // Medium amber
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(139, 100, 35) // Dim amber
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(200, 140, 60) // Medium amber
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(139, 100, 35) // Dim amber
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(139, 100, 35) // Dim amber
    }

    fn window_header(&self) -> Color {
        Color::Rgb(255, 165, 0) // Warm amber
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(100, 60, 0) // Deep amber-brown
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(255, 191, 0) // Bright amber
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(255, 165, 0) // Warm amber
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(255, 140, 0) // Deep amber
    }

    fn selected_row_bg(&self) -> Color {
        Color::Rgb(30, 60, 100)
    }

    fn selected_row_fg(&self) -> Color {
        Color::Rgb(200, 220, 240)
    }
}

impl SymbolSet for InterstellarDarkTheme {
    // Status symbols - geometric precision
    fn symbol_detected(&self) -> &'static str {
        "▫"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "▪"
    }

    fn symbol_rejected(&self) -> &'static str {
        "▬"
    }

    fn symbol_signal(&self) -> &'static str {
        "■"
    }

    fn symbol_playing(&self) -> &'static str {
        "▶"
    }

    fn symbol_completed(&self) -> &'static str {
        "█"
    }

    // Progress bar characters - geometric
    fn progress_empty(&self) -> &'static str {
        "▱"
    }

    fn progress_full(&self) -> &'static str {
        "▰"
    }

    // Spectrum visualization
    fn spectrum_baseline(&self) -> char {
        '─'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    // Window header decoration
    fn window_bullet(&self) -> &'static str {
        "▫"
    }

    // Header border
    fn header_border(&self) -> char {
        '▄'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for InterstellarDarkTheme {}

impl Theme for InterstellarDarkTheme {
    fn name(&self) -> &str {
        "interstellar-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Interstellar light theme - golden bronze on cream
pub struct InterstellarLightTheme;

impl ColorScheme for InterstellarLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn secondary(&self) -> Color {
        Color::Rgb(146, 64, 14) // Bronze
    }

    fn accent(&self) -> Color {
        Color::Rgb(120, 53, 15) // Dark bronze
    }

    fn background(&self) -> Color {
        Color::Rgb(254, 252, 232) // Cream
    }

    fn foreground(&self) -> Color {
        Color::Rgb(92, 44, 20) // Dark brown
    }

    // Status colors - golden bronze hierarchy
    fn status_detected(&self) -> Color {
        Color::Rgb(202, 138, 4) // Medium gold
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(120, 113, 108) // Warm gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(146, 64, 14) // Bronze
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(164, 120, 58) // Medium bronze
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(202, 138, 4) // Medium gold
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(164, 120, 58) // Medium bronze
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(120, 113, 108) // Warm gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(164, 120, 58) // Medium bronze
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(120, 113, 108) // Warm gray
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(146, 64, 14) // Bronze
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(120, 113, 108) // Warm gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(202, 138, 4) // Medium gold
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(254, 252, 232) // Cream
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(146, 64, 14) // Bronze
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(202, 138, 4) // Medium gold
    }

    fn selected_row_bg(&self) -> Color {
        Color::Rgb(20, 50, 90)
    }

    fn selected_row_fg(&self) -> Color {
        Color::Rgb(255, 255, 255)
    }
}

impl SymbolSet for InterstellarLightTheme {
    // Same symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "▫"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "▪"
    }

    fn symbol_rejected(&self) -> &'static str {
        "▬"
    }

    fn symbol_signal(&self) -> &'static str {
        "■"
    }

    fn symbol_playing(&self) -> &'static str {
        "▶"
    }

    fn symbol_completed(&self) -> &'static str {
        "█"
    }

    fn progress_empty(&self) -> &'static str {
        "▱"
    }

    fn progress_full(&self) -> &'static str {
        "▰"
    }

    fn spectrum_baseline(&self) -> char {
        '─'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    fn window_bullet(&self) -> &'static str {
        "▫"
    }

    fn header_border(&self) -> char {
        '▄'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for InterstellarLightTheme {}

impl Theme for InterstellarLightTheme {
    fn name(&self) -> &str {
        "interstellar-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
