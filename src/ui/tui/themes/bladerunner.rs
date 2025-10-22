//! Blade Runner theme implementations - cyberpunk aesthetic

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Blade Runner dark theme - cyberpunk neon aesthetic
pub struct BladerunnerDarkTheme;

impl ColorScheme for BladerunnerDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(0, 255, 255) // Cyan neon
    }

    fn secondary(&self) -> Color {
        Color::Rgb(255, 140, 0) // Dark orange
    }

    fn accent(&self) -> Color {
        Color::Rgb(255, 20, 147) // Deep pink
    }

    fn background(&self) -> Color {
        Color::Rgb(10, 10, 20) // Very dark blue-black
    }

    fn foreground(&self) -> Color {
        Color::Rgb(192, 255, 238) // Light cyan
    }

    // Status colors - cyberpunk neon palette
    fn status_detected(&self) -> Color {
        Color::Rgb(255, 255, 0) // Bright yellow
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(0, 255, 255) // Cyan
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(139, 69, 19) // Saddle brown
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(50, 205, 50) // Lime green
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(255, 20, 147) // Deep pink
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(255, 140, 0) // Dark orange
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(50, 205, 50) // Lime green
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(255, 255, 0) // Yellow
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(255, 69, 0) // Red orange
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(139, 69, 19) // Saddle brown
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(255, 69, 0) // Red orange
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(139, 69, 19) // Saddle brown
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(0, 255, 255) // Cyan
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(255, 20, 147) // Deep pink
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(139, 69, 19) // Saddle brown
    }

    fn window_header(&self) -> Color {
        Color::Rgb(255, 140, 0) // Dark orange
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(20, 0, 40) // Deep purple-black
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(0, 255, 255) // Cyan neon
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(255, 20, 147) // Deep pink
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(255, 255, 0) // Yellow
    }
}

impl SymbolSet for BladerunnerDarkTheme {
    // Status symbols - cyberpunk geometric
    fn symbol_detected(&self) -> &'static str {
        "◈"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌"
    }

    fn symbol_signal(&self) -> &'static str {
        "◆"
    }

    fn symbol_playing(&self) -> &'static str {
        "▶"
    }

    fn symbol_completed(&self) -> &'static str {
        "■"
    }

    // Progress bar characters - cyberpunk blocks
    fn progress_empty(&self) -> &'static str {
        "░"
    }

    fn progress_full(&self) -> &'static str {
        "█"
    }

    // Spectrum visualization
    fn spectrum_baseline(&self) -> char {
        '═'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    // Window header decoration
    fn window_bullet(&self) -> &'static str {
        "▣"
    }

    // Header border
    fn header_border(&self) -> char {
        '═'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for BladerunnerDarkTheme {}

impl Theme for BladerunnerDarkTheme {
    fn name(&self) -> &str {
        "bladerunner-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Blade Runner light theme - corporate dystopia aesthetic
pub struct BladerunnerLightTheme;

impl ColorScheme for BladerunnerLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(0, 139, 139) // Dark cyan
    }

    fn secondary(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark amber
    }

    fn accent(&self) -> Color {
        Color::Rgb(139, 0, 139) // Dark magenta
    }

    fn background(&self) -> Color {
        Color::Rgb(245, 245, 245) // Light gray
    }

    fn foreground(&self) -> Color {
        Color::Rgb(47, 79, 79) // Dark slate gray
    }

    // Status colors - muted corporate palette
    fn status_detected(&self) -> Color {
        Color::Rgb(218, 165, 32) // Golden rod
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(0, 139, 139) // Dark cyan
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(34, 139, 34) // Forest green
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(139, 0, 139) // Dark magenta
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark amber
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(34, 139, 34) // Forest green
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(218, 165, 32) // Golden rod
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(178, 34, 34) // Fire brick
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(178, 34, 34) // Fire brick
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(0, 139, 139) // Dark cyan
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(139, 0, 139) // Dark magenta
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(105, 105, 105) // Dim gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(184, 134, 11) // Dark amber
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(0, 139, 139) // Dark cyan
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(245, 245, 245) // Light gray
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(139, 0, 139) // Dark magenta
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(218, 165, 32) // Golden rod
    }
}

impl SymbolSet for BladerunnerLightTheme {
    // Same symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "◈"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌"
    }

    fn symbol_signal(&self) -> &'static str {
        "◆"
    }

    fn symbol_playing(&self) -> &'static str {
        "▶"
    }

    fn symbol_completed(&self) -> &'static str {
        "■"
    }

    fn progress_empty(&self) -> &'static str {
        "░"
    }

    fn progress_full(&self) -> &'static str {
        "█"
    }

    fn spectrum_baseline(&self) -> char {
        '═'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    fn window_bullet(&self) -> &'static str {
        "▣"
    }

    fn header_border(&self) -> char {
        '═'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for BladerunnerLightTheme {}

impl Theme for BladerunnerLightTheme {
    fn name(&self) -> &str {
        "bladerunner-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
