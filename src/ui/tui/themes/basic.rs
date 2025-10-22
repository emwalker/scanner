//! Basic theme implementations - original scanner aesthetic

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Basic dark theme - original main branch colors
pub struct BasicDarkTheme;

impl ColorScheme for BasicDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(255, 215, 0) // Gold
    }

    fn secondary(&self) -> Color {
        Color::Rgb(100, 149, 237) // Cornflower blue
    }

    fn accent(&self) -> Color {
        Color::Rgb(255, 215, 0) // Gold
    }

    fn background(&self) -> Color {
        Color::Reset // Terminal default (typically black)
    }

    fn foreground(&self) -> Color {
        Color::Reset // Terminal default (typically white)
    }

    // Status colors - exact original from main branch
    fn status_detected(&self) -> Color {
        Color::Rgb(255, 215, 0) // Gold
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(100, 149, 237) // Cornflower blue
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(70, 130, 180) // Steel blue
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(186, 85, 211) // Medium orchid
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(60, 179, 113) // Medium sea green
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(60, 179, 113) // Medium sea green
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(255, 165, 0) // Orange
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(255, 99, 71) // Tomato
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(255, 165, 0) // Orange
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(255, 215, 0) // Gold
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(100, 149, 237) // Cornflower blue
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(128, 128, 128) // Gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(255, 165, 0) // Orange
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0) // Bright yellow-orange for selection
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(80, 40, 0) // Dark orange-brown background
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(255, 255, 255) // White text
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(255, 200, 128) // Light orange
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(255, 220, 100) // Bright yellow-orange (matches theme)
    }
}

impl SymbolSet for BasicDarkTheme {
    // Status symbols - exact original from main branch
    fn symbol_detected(&self) -> &'static str {
        "◦"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌"
    }

    fn symbol_signal(&self) -> &'static str {
        "●"
    }

    fn symbol_playing(&self) -> &'static str {
        "♬"
    }

    fn symbol_completed(&self) -> &'static str {
        "◆"
    }

    // Progress bar characters - elegant geometric
    fn progress_empty(&self) -> &'static str {
        "⠀" // Braille blank
    }

    fn progress_full(&self) -> &'static str {
        "⣿" // Braille full
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
        "◆"
    }

    // Header border
    fn header_border(&self) -> char {
        '─'
    }

    // Selection indicator
    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for BasicDarkTheme {}

impl Theme for BasicDarkTheme {
    fn name(&self) -> &str {
        "basic-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Basic light theme - inverted color scheme for light terminals
pub struct BasicLightTheme;

impl ColorScheme for BasicLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn secondary(&self) -> Color {
        Color::Rgb(30, 64, 175) // Dark blue
    }

    fn accent(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn background(&self) -> Color {
        Color::Rgb(255, 255, 255) // White
    }

    fn foreground(&self) -> Color {
        Color::Rgb(0, 0, 0) // Black
    }

    // Status colors - darker variants for light background
    fn status_detected(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(30, 64, 175) // Dark blue
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(107, 114, 128) // Dark gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(30, 58, 138) // Dark steel blue
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(126, 34, 206) // Dark orchid
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(21, 128, 61) // Dark green
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(21, 128, 61) // Dark green
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(194, 65, 12) // Dark orange
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(185, 28, 28) // Dark red
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(107, 114, 128) // Dark gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(194, 65, 12) // Dark orange
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(107, 114, 128) // Dark gray
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(30, 64, 175) // Dark blue
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(107, 114, 128) // Dark gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(194, 65, 12) // Dark orange
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 140, 0) // Bright orange for selection
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(255, 165, 0) // Bright orange background
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(0, 0, 0) // Black text for light theme
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(139, 69, 19) // Dark brown
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark gold (matches theme)
    }
}

impl SymbolSet for BasicLightTheme {
    // Same symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "◦"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌"
    }

    fn symbol_signal(&self) -> &'static str {
        "●"
    }

    fn symbol_playing(&self) -> &'static str {
        "♬"
    }

    fn symbol_completed(&self) -> &'static str {
        "◆"
    }

    fn progress_empty(&self) -> &'static str {
        "⠀" // Braille blank
    }

    fn progress_full(&self) -> &'static str {
        "⣿" // Braille full
    }

    fn spectrum_baseline(&self) -> char {
        '─'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    fn window_bullet(&self) -> &'static str {
        "◆"
    }

    fn header_border(&self) -> char {
        '─'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for BasicLightTheme {}

impl Theme for BasicLightTheme {
    fn name(&self) -> &str {
        "basic-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
