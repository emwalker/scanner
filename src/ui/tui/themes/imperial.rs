//! Imperial theme - Refined opulence of 10,000 years of civilization

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Imperial dark theme - understated luxury
pub struct DarkTheme;

impl ColorScheme for DarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(220, 200, 160) // Champagne gold - refined and warm
    }

    fn secondary(&self) -> Color {
        Color::Rgb(180, 165, 145) // Aged ivory - sophisticated neutral
    }

    fn accent(&self) -> Color {
        Color::Rgb(240, 215, 170) // Pale gold leaf - subtle emphasis
    }

    fn background(&self) -> Color {
        Color::Rgb(18, 16, 14) // Deep ebony - luxurious darkness
    }

    fn foreground(&self) -> Color {
        Color::Rgb(230, 220, 200) // Warm parchment white
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(200, 185, 160) // Soft gold - initial observation
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(220, 200, 160) // Champagne - careful evaluation
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(80, 75, 70) // Subtle shadow - dismissed with grace
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(240, 215, 170) // Pale gold - confirmed significance
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(235, 210, 175) // Luminous ivory - active attention
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(140, 130, 115) // Aged bronze - completed elegantly
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(210, 200, 180) // Pristine ivory - excellent quality
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(200, 185, 160) // Standard gold - acceptable
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(160, 145, 125) // Tarnished bronze - degraded
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(90, 85, 80) // Deep shadow - absent
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(140, 125, 110) // Dusty gold - interference
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(120, 110, 100) // Neutral bronze - undetermined
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(240, 215, 170) // Pale gold leaf
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(230, 210, 175) // Luminous scan indicator
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(140, 130, 115) // Subdued elegance
    }

    fn window_header(&self) -> Color {
        Color::Rgb(210, 195, 165) // Section gold
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }
}

impl SymbolSet for DarkTheme {
    fn symbol_detected(&self) -> &'static str {
        "◇" // Hollow diamond - refined discovery
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◈" // Dotted diamond - careful examination
    }

    fn symbol_rejected(&self) -> &'static str {
        "◦" // Small circle - politely dismissed
    }

    fn symbol_signal(&self) -> &'static str {
        "◆" // Solid diamond - confirmed treasure
    }

    fn symbol_playing(&self) -> &'static str {
        "▹" // Delicate arrow - graceful attention
    }

    fn symbol_completed(&self) -> &'static str {
        "◈" // Dotted diamond - elegantly concluded
    }

    fn progress_empty(&self) -> &'static str {
        "─" // Fine line - refined emptiness
    }

    fn progress_full(&self) -> &'static str {
        "═" // Double line - substantial progress
    }

    fn spectrum_baseline(&self) -> char {
        '·' // Delicate dots
    }

    fn spectrum_window_char(&self) -> char {
        '▬' // Refined block
    }

    fn window_bullet(&self) -> &'static str {
        "◆" // Diamond marker
    }

    fn header_border(&self) -> char {
        '─' // Understated line
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for DarkTheme {
    fn title(&self) -> &'static str {
        "S P E C T R U M   M O N I T O R"
    }

    fn subtitle(&self) -> &'static str {
        "Frequency Analysis  ·  88–108 MHz  ·  FM"
    }

    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Analyzing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Dismissed"
    }

    fn status_signal_text(&self) -> &'static str {
        "Confirmed"
    }

    fn status_playing_text(&self) -> &'static str {
        "Monitoring"
    }

    fn status_completed_text(&self) -> &'static str {
        "Complete"
    }
}

impl Theme for DarkTheme {
    fn name(&self) -> &str {
        "imperial-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Imperial light theme - daylight opulence
pub struct LightTheme;

impl ColorScheme for LightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(120, 100, 70) // Deep bronze - rich authority
    }

    fn secondary(&self) -> Color {
        Color::Rgb(140, 120, 90) // Antique gold - supporting elegance
    }

    fn accent(&self) -> Color {
        Color::Rgb(100, 80, 55) // Dark bronze - refined emphasis
    }

    fn background(&self) -> Color {
        Color::Rgb(250, 248, 244) // Cream silk - luxurious light
    }

    fn foreground(&self) -> Color {
        Color::Rgb(40, 35, 28) // Rich ebony text
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(130, 110, 80) // Warm bronze
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(120, 100, 70) // Deep bronze
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(180, 175, 168) // Pale shadow
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(100, 80, 55) // Dark bronze emphasis
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(110, 90, 65) // Rich bronze
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(150, 140, 125) // Aged gold
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(110, 90, 65) // Rich quality
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(130, 110, 80) // Standard bronze
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(150, 130, 100) // Tarnished
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(170, 165, 158) // Absent gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(145, 125, 100) // Dusty bronze
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(160, 150, 135) // Neutral
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(100, 80, 55) // Dark bronze
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(110, 90, 65) // Rich scan
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(160, 150, 135) // Subdued
    }

    fn window_header(&self) -> Color {
        Color::Rgb(120, 100, 70) // Section bronze
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }
}

impl SymbolSet for LightTheme {
    fn symbol_detected(&self) -> &'static str {
        "◇"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◈"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◦"
    }

    fn symbol_signal(&self) -> &'static str {
        "◆"
    }

    fn symbol_playing(&self) -> &'static str {
        "▹"
    }

    fn symbol_completed(&self) -> &'static str {
        "◈"
    }

    fn progress_empty(&self) -> &'static str {
        "─"
    }

    fn progress_full(&self) -> &'static str {
        "═"
    }

    fn spectrum_baseline(&self) -> char {
        '·'
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

impl TextStyle for LightTheme {
    fn title(&self) -> &'static str {
        "S P E C T R U M   M O N I T O R"
    }

    fn subtitle(&self) -> &'static str {
        "Frequency Analysis  ·  88–108 MHz  ·  FM"
    }

    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Analyzing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Dismissed"
    }

    fn status_signal_text(&self) -> &'static str {
        "Confirmed"
    }

    fn status_playing_text(&self) -> &'static str {
        "Monitoring"
    }

    fn status_completed_text(&self) -> &'static str {
        "Complete"
    }
}

impl Theme for LightTheme {
    fn name(&self) -> &str {
        "imperial-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
