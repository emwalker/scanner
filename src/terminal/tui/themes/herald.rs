//! Herald Monitor theme - Imperial transport ceremonial interface aesthetic
//!
//! Inspired by the spherical Imperial transport shuttle from Denis Villeneuve's Dune films.
//! This theme represents the monitoring equipment used by the Herald's attendants to observe
//! planetary communications upon arrival - dignified, ceremonial, yet precise and functional.
//!
//! Design rationale:
//! - Warm golds and bronzes evoke Imperial ceremonial regalia
//! - Deep blacks suggest the formality of Imperial protocol
//! - Subtle accent colors maintain readability while preserving gravitas
//! - Geometric symbols blend Imperial precision with ceremonial dignity
//! - Formal terminology appropriate for equipment used in official capacity

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Herald Monitor dark theme - ceremonial Imperial equipment aesthetic
pub struct HeraldDarkTheme;

impl ColorScheme for HeraldDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(205, 173, 110) // Ceremonial gold - warm but restrained
    }

    fn secondary(&self) -> Color {
        Color::Rgb(139, 115, 85) // Bronze - supporting Imperial metal
    }

    fn accent(&self) -> Color {
        Color::Rgb(218, 185, 130) // Light gold - subtle emphasis
    }

    fn background(&self) -> Color {
        Color::Rgb(15, 12, 10) // Deep ceremonial black
    }

    fn foreground(&self) -> Color {
        Color::Rgb(220, 200, 170) // Warm parchment - dignified readability
    }

    // Status colors - Imperial monitoring hierarchy
    fn status_detected(&self) -> Color {
        Color::Rgb(195, 165, 120) // Initial detection gold
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(160, 140, 110) // Analysis bronze
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(80, 70, 60) // Filtered dark bronze
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(205, 173, 110) // Signal acquisition gold
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(180, 160, 130) // Active monitoring amber
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(139, 115, 85) // Completion bronze
    }

    // Audio quality colors - subtle ceremonial indicators
    fn quality_good(&self) -> Color {
        Color::Rgb(195, 175, 140) // Excellent clarity - warm gold
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(185, 155, 115) // Acceptable quality - standard gold
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(150, 120, 90) // Degraded signal - darker bronze
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(90, 75, 60) // No transmission - deep bronze
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(130, 105, 80) // Interference - muted bronze
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(100, 85, 70) // Unverified - neutral dark
    }

    // UI element colors - Imperial interface elements
    fn header_accent(&self) -> Color {
        Color::Rgb(218, 185, 130) // Header emphasis - bright gold
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(205, 173, 110) // Scanning window - ceremonial gold
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(120, 100, 80) // Subdued instructions - muted bronze
    }

    fn window_header(&self) -> Color {
        Color::Rgb(195, 165, 120) // Window titles - formal gold
    }
}

impl SymbolSet for HeraldDarkTheme {
    // Status symbols - Imperial precision markers
    fn symbol_detected(&self) -> &'static str {
        "◯" // Open circle - transmission located
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐" // Half circle - under evaluation
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌" // Empty circle - deemed unsuitable
    }

    fn symbol_signal(&self) -> &'static str {
        "◉" // Ringed dot - signal confirmed
    }

    fn symbol_playing(&self) -> &'static str {
        "▸" // Triangle - broadcast monitoring
    }

    fn symbol_completed(&self) -> &'static str {
        "◆" // Diamond - protocol fulfilled
    }

    // Progress bar characters - ceremonial geometric fills
    fn progress_empty(&self) -> &'static str {
        "▱" // Light rectangular block
    }

    fn progress_full(&self) -> &'static str {
        "▰" // Dark rectangular block - formal progression
    }

    // Spectrum visualization - dignified scanning indicators
    fn spectrum_baseline(&self) -> char {
        '·' // Centered dot - baseline monitoring
    }

    fn spectrum_window_char(&self) -> char {
        '═' // Double horizontal line - scanning emphasis
    }

    // Window header decoration - Imperial marker
    fn window_bullet(&self) -> &'static str {
        "⬥" // Diamond bullet - formal designation
    }

    // Header border - ceremonial separator
    fn header_border(&self) -> char {
        '─' // Horizontal line - clean Imperial divide
    }
}

impl TextStyle for HeraldDarkTheme {
    fn title(&self) -> &'static str {
        "HERALD COMMUNICATIONS MONITOR"
    }

    fn subtitle(&self) -> &'static str {
        "Observing Planetary Transmissions • FM Spectrum • 88–108 MHz"
    }

    // Status text - formal Imperial protocol terminology
    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Verifying"
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
        "Logged"
    }
}

impl Theme for HeraldDarkTheme {
    fn name(&self) -> &str {
        "herald-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Herald Monitor light theme - daylight ceremonial interface
pub struct HeraldLightTheme;

impl ColorScheme for HeraldLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(120, 90, 50) // Deep ceremonial bronze
    }

    fn secondary(&self) -> Color {
        Color::Rgb(100, 75, 45) // Darker bronze supporting
    }

    fn accent(&self) -> Color {
        Color::Rgb(140, 105, 60) // Lighter bronze accent
    }

    fn background(&self) -> Color {
        Color::Rgb(245, 240, 230) // Warm parchment background
    }

    fn foreground(&self) -> Color {
        Color::Rgb(40, 35, 25) // Deep ceremonial text
    }

    // Status colors - light theme Imperial hierarchy
    fn status_detected(&self) -> Color {
        Color::Rgb(130, 100, 60) // Detection bronze
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(100, 80, 55) // Analysis deep bronze
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(140, 130, 115) // Filtered neutral
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(120, 90, 50) // Signal gold-bronze
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(110, 85, 50) // Active monitoring
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(100, 75, 45) // Completion dark bronze
    }

    // Audio quality colors - light theme variants
    fn quality_good(&self) -> Color {
        Color::Rgb(90, 110, 70) // Good quality - subtle green-bronze
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(130, 100, 60) // Moderate - standard bronze
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(140, 90, 60) // Poor - warm bronze
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(130, 120, 105) // No audio - neutral gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(135, 95, 65) // Static - muted bronze
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(125, 115, 100) // Unknown - neutral
    }

    // UI element colors - light theme Imperial elements
    fn header_accent(&self) -> Color {
        Color::Rgb(120, 90, 50) // Header bronze
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(110, 85, 50) // Scanning emphasis
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(150, 140, 125) // Dimmed instructions
    }

    fn window_header(&self) -> Color {
        Color::Rgb(130, 100, 60) // Window headers
    }
}

impl SymbolSet for HeraldLightTheme {
    // Same ceremonial symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "◯"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◌"
    }

    fn symbol_signal(&self) -> &'static str {
        "◉"
    }

    fn symbol_playing(&self) -> &'static str {
        "▸"
    }

    fn symbol_completed(&self) -> &'static str {
        "◆"
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
        '═'
    }

    fn window_bullet(&self) -> &'static str {
        "⬥"
    }

    fn header_border(&self) -> char {
        '─'
    }
}

impl TextStyle for HeraldLightTheme {
    fn title(&self) -> &'static str {
        "HERALD COMMUNICATIONS MONITOR"
    }

    fn subtitle(&self) -> &'static str {
        "Observing Planetary Transmissions • FM Spectrum • 88–108 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Verifying"
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
        "Logged"
    }
}

impl Theme for HeraldLightTheme {
    fn name(&self) -> &str {
        "herald-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}