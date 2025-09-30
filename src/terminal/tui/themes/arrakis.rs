//! Arrakis theme implementations - authentic cinematic desert survival aesthetic

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Arrakis dark theme - muted desert survival with holographic light sheets
pub struct ArrakisDarkTheme;

impl ColorScheme for ArrakisDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(180, 150, 110) // Muted sand
    }

    fn secondary(&self) -> Color {
        Color::Rgb(140, 120, 90) // Darker sand
    }

    fn accent(&self) -> Color {
        Color::Rgb(200, 170, 120) // Light sand accent
    }

    fn background(&self) -> Color {
        Color::Reset // Terminal default
    }

    fn foreground(&self) -> Color {
        Color::Rgb(200, 180, 140) // Desaturated sand light
    }

    // Status colors - muted survival hierarchy
    fn status_detected(&self) -> Color {
        Color::Rgb(160, 140, 100) // Muted sand
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(120, 140, 160) // Subtle blue-gray
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(100, 90, 80) // Dark dust
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(180, 150, 110) // Functional sand
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(140, 160, 180) // Muted blue
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(120, 110, 90) // Deep muted sand
    }

    // Audio quality colors - muted and practical
    fn quality_good(&self) -> Color {
        Color::Rgb(140, 160, 120) // Muted green-sand
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(180, 150, 110) // Neutral sand
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(140, 120, 100) // Darker sand
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(100, 90, 80) // Deep dust
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(140, 120, 100) // Darker sand
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(100, 90, 80) // Deep dust
    }

    // UI element colors - subtle holographic light
    fn header_accent(&self) -> Color {
        Color::Rgb(180, 160, 130) // Muted light sand
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(160, 140, 110) // Subtle scanning light
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(120, 100, 80) // Dust gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(180, 150, 120) // Muted sand header
    }
}

impl SymbolSet for ArrakisDarkTheme {
    // Status symbols - simple, functional indicators
    fn symbol_detected(&self) -> &'static str {
        "○" // Simple open circle
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◔" // Quarter circle - minimal processing indicator
    }

    fn symbol_rejected(&self) -> &'static str {
        "◦" // Small empty circle - filtered out
    }

    fn symbol_signal(&self) -> &'static str {
        "●" // Simple filled circle - clear signal
    }

    fn symbol_playing(&self) -> &'static str {
        "▸" // Simple arrow - minimal play indicator
    }

    fn symbol_completed(&self) -> &'static str {
        "◼" // Small square - task complete
    }

    // Progress bar characters - simple and functional
    fn progress_empty(&self) -> &'static str {
        "░" // Light fill
    }

    fn progress_full(&self) -> &'static str {
        "▒" // Medium fill - less dense than other themes
    }

    // Spectrum visualization - minimal but clear
    fn spectrum_baseline(&self) -> char {
        '·' // Simple dots
    }

    fn spectrum_window_char(&self) -> char {
        '━' // Simple line - clear scanning indicator
    }

    // Window header decoration - minimal
    fn window_bullet(&self) -> &'static str {
        "▪" // Small square bullet
    }

    // Header border - clean line
    fn header_border(&self) -> char {
        '─' // Simple horizontal line
    }
}

impl TextStyle for ArrakisDarkTheme {
    fn title(&self) -> &'static str {
        "RADIO SCANNER"
    }

    fn subtitle(&self) -> &'static str {
        "Monitoring broadcast spectrum • FM • 88–108 MHz"
    }

    // Status text - practical monitoring terminology
    fn status_detected_text(&self) -> &'static str {
        "Located"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Testing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Filtered"
    }

    fn status_signal_text(&self) -> &'static str {
        "Captured"
    }

    fn status_playing_text(&self) -> &'static str {
        "Active"
    }

    fn status_completed_text(&self) -> &'static str {
        "Complete"
    }
}

impl Theme for ArrakisDarkTheme {
    fn name(&self) -> &str {
        "arrakis-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Arrakis light theme - bright desert day with deep spice accents
pub struct ArrakisLightTheme;

impl ColorScheme for ArrakisLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(40, 80, 120) // Deep spice
    }

    fn secondary(&self) -> Color {
        Color::Rgb(140, 110, 80) // Dark sand
    }

    fn accent(&self) -> Color {
        Color::Rgb(180, 80, 40) // Deep orange
    }

    fn background(&self) -> Color {
        Color::Rgb(250, 240, 220) // Light sand
    }

    fn foreground(&self) -> Color {
        Color::Rgb(60, 45, 30) // Desert shadow
    }

    // Status colors - bright desert variants
    fn status_detected(&self) -> Color {
        Color::Rgb(200, 150, 50) // Dark sand yellow
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(40, 80, 120) // Deep spice
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(130, 120, 110) // Desert dust
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(180, 80, 40) // Deep orange
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(30, 90, 150) // Deeper spice
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(120, 90, 60) // Dark sand
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(30, 90, 150) // Deeper spice
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(180, 80, 40) // Deep orange
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(120, 90, 60) // Dark sand
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(130, 120, 110) // Desert dust
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(120, 90, 60) // Dark sand
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(130, 120, 110) // Desert dust
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(180, 80, 40) // Deep orange
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(30, 90, 150) // Deeper spice
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(130, 120, 110) // Desert dust
    }

    fn window_header(&self) -> Color {
        Color::Rgb(140, 110, 80) // Dark sand
    }
}

impl SymbolSet for ArrakisLightTheme {
    // Same practical symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "○"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◔"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◦"
    }

    fn symbol_signal(&self) -> &'static str {
        "●"
    }

    fn symbol_playing(&self) -> &'static str {
        "▸"
    }

    fn symbol_completed(&self) -> &'static str {
        "◼"
    }

    fn progress_empty(&self) -> &'static str {
        "░"
    }

    fn progress_full(&self) -> &'static str {
        "▒"
    }

    fn spectrum_baseline(&self) -> char {
        '·'
    }

    fn spectrum_window_char(&self) -> char {
        '━'
    }

    fn window_bullet(&self) -> &'static str {
        "▪"
    }

    fn header_border(&self) -> char {
        '─'
    }
}

impl TextStyle for ArrakisLightTheme {
    fn title(&self) -> &'static str {
        "RADIO SCANNER"
    }

    fn subtitle(&self) -> &'static str {
        "Monitoring broadcast spectrum • FM • 88–108 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "Located"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Testing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Filtered"
    }

    fn status_signal_text(&self) -> &'static str {
        "Captured"
    }

    fn status_playing_text(&self) -> &'static str {
        "Active"
    }

    fn status_completed_text(&self) -> &'static str {
        "Complete"
    }
}

impl Theme for ArrakisLightTheme {
    fn name(&self) -> &str {
        "arrakis-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
