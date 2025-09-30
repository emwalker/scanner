//! Minimal theme - High-contrast clean monitoring interface

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Dark theme - clean high-contrast display
pub struct DarkTheme;

impl ColorScheme for DarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(200, 220, 240) // Bright ice blue - priority data
    }

    fn secondary(&self) -> Color {
        Color::Rgb(140, 150, 160) // Steel gray - supporting info
    }

    fn accent(&self) -> Color {
        Color::Rgb(255, 200, 80) // Amber alert - high priority
    }

    fn background(&self) -> Color {
        Color::Rgb(12, 14, 16) // Deep tactical black
    }

    fn foreground(&self) -> Color {
        Color::Rgb(230, 235, 240) // High-contrast white
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(180, 200, 220) // Cool blue - new contact
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(200, 220, 240) // Bright blue - active analysis
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(60, 65, 70) // Dark gray - filtered out
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(255, 200, 80) // Amber - confirmed target
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(220, 240, 255) // Bright ice - active monitor
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(100, 110, 120) // Medium gray - archived
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(200, 220, 240) // Clear ice blue
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(180, 200, 140) // Pale green-blue
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(160, 140, 120) // Dull tan
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(70, 75, 80) // Dark steel
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(120, 110, 100) // Muddy gray
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(90, 95, 100) // Neutral gray
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(255, 200, 80) // Amber - command emphasis
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(220, 240, 255) // Bright scan indicator
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(100, 110, 120) // Subdued steel
    }

    fn window_header(&self) -> Color {
        Color::Rgb(180, 200, 220) // Section steel blue
    }
}

impl SymbolSet for DarkTheme {
    fn symbol_detected(&self) -> &'static str {
        "●" // Solid dot - initial contact
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉" // Ringed dot - under analysis
    }

    fn symbol_rejected(&self) -> &'static str {
        "○" // Empty circle - dismissed
    }

    fn symbol_signal(&self) -> &'static str {
        "◆" // Diamond - priority target
    }

    fn symbol_playing(&self) -> &'static str {
        "▸" // Arrow - active monitoring
    }

    fn symbol_completed(&self) -> &'static str {
        "■" // Square - completed
    }

    fn progress_empty(&self) -> &'static str {
        "▱" // Empty bar
    }

    fn progress_full(&self) -> &'static str {
        "▰" // Filled bar
    }

    fn spectrum_baseline(&self) -> char {
        '·' // Dots for baseline
    }

    fn spectrum_window_char(&self) -> char {
        '█' // Solid block for active scan
    }

    fn window_bullet(&self) -> &'static str {
        "▸" // Arrow bullet
    }

    fn header_border(&self) -> char {
        '━' // Heavy line
    }
}

impl TextStyle for DarkTheme {
    fn title(&self) -> &'static str {
        "SPECTRUM MONITOR"
    }

    fn subtitle(&self) -> &'static str {
        "FM Receiver • 88–108 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "Found"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Testing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Filtered"
    }

    fn status_signal_text(&self) -> &'static str {
        "Active"
    }

    fn status_playing_text(&self) -> &'static str {
        "Playing"
    }

    fn status_completed_text(&self) -> &'static str {
        "Done"
    }
}

impl Theme for DarkTheme {
    fn name(&self) -> &str {
        "minimal-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Light theme - high-visibility clean display
pub struct LightTheme;

impl ColorScheme for LightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(30, 60, 100) // Deep navy blue
    }

    fn secondary(&self) -> Color {
        Color::Rgb(80, 90, 100) // Steel gray
    }

    fn accent(&self) -> Color {
        Color::Rgb(200, 120, 0) // Deep amber
    }

    fn background(&self) -> Color {
        Color::Rgb(240, 242, 245) // Light gray console
    }

    fn foreground(&self) -> Color {
        Color::Rgb(20, 25, 30) // Near black
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(40, 80, 120) // Medium blue
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(30, 60, 100) // Deep blue
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(160, 165, 170) // Light gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(200, 120, 0) // Deep amber
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(20, 50, 90) // Darker blue
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(120, 130, 140) // Medium steel
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(30, 60, 100) // Deep navy
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(60, 100, 60) // Forest green
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(140, 100, 60) // Tan brown
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(150, 155, 160) // Light steel
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(130, 120, 110) // Muddy gray
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(140, 145, 150) // Neutral steel
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(200, 120, 0) // Deep amber
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(20, 50, 90) // Dark navy
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(140, 145, 150) // Light steel
    }

    fn window_header(&self) -> Color {
        Color::Rgb(60, 90, 130) // Medium navy
    }
}

impl SymbolSet for LightTheme {
    fn symbol_detected(&self) -> &'static str {
        "●"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉"
    }

    fn symbol_rejected(&self) -> &'static str {
        "○"
    }

    fn symbol_signal(&self) -> &'static str {
        "◆"
    }

    fn symbol_playing(&self) -> &'static str {
        "▸"
    }

    fn symbol_completed(&self) -> &'static str {
        "■"
    }

    fn progress_empty(&self) -> &'static str {
        "▱"
    }

    fn progress_full(&self) -> &'static str {
        "▰"
    }

    fn spectrum_baseline(&self) -> char {
        '·'
    }

    fn spectrum_window_char(&self) -> char {
        '█'
    }

    fn window_bullet(&self) -> &'static str {
        "▸"
    }

    fn header_border(&self) -> char {
        '━'
    }
}

impl TextStyle for LightTheme {
    fn title(&self) -> &'static str {
        "SPECTRUM MONITOR"
    }

    fn subtitle(&self) -> &'static str {
        "FM Receiver • 88–108 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "Found"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Testing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Filtered"
    }

    fn status_signal_text(&self) -> &'static str {
        "Active"
    }

    fn status_playing_text(&self) -> &'static str {
        "Playing"
    }

    fn status_completed_text(&self) -> &'static str {
        "Done"
    }
}

impl Theme for LightTheme {
    fn name(&self) -> &str {
        "minimal-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
