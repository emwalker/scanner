//! Archive theme - Imperial diplomatic protocol and administrative record-keeping

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Archive dark theme - bureaucratic documentation terminal
pub struct ArchiveDarkTheme;

impl ColorScheme for ArchiveDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(145, 125, 105) // Aged parchment brown
    }

    fn secondary(&self) -> Color {
        Color::Rgb(85, 75, 70) // Deep archive ink
    }

    fn accent(&self) -> Color {
        Color::Rgb(165, 140, 115) // Lighter parchment
    }

    fn background(&self) -> Color {
        Color::Rgb(22, 20, 18) // Archive vault black
    }

    fn foreground(&self) -> Color {
        Color::Rgb(190, 175, 160) // Faded ink on parchment
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(155, 135, 110) // New filing brown
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(105, 95, 85) // Under review gray-brown
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(65, 60, 55) // Discarded filing dark
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(145, 125, 105) // Cataloged entry
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(125, 110, 95) // Active reference
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(95, 85, 75) // Archived permanently
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(130, 145, 125) // Verified certification
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(155, 135, 110) // Standard filing
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(125, 105, 85) // Degraded record
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(70, 65, 60) // Empty filing
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(115, 95, 80) // Corrupted archive
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(85, 80, 75) // Unclassified
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(165, 140, 115) // Official document header
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(145, 125, 105) // Scanning catalog
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(100, 90, 80) // Subdued protocol notes
    }

    fn window_header(&self) -> Color {
        Color::Rgb(155, 135, 110) // Section dividers
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(60, 50, 40) // Dark archive brown
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(220, 200, 180) // Light parchment
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(190, 170, 145) // Medium parchment
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(200, 180, 150) // Warm parchment
    }
}

impl SymbolSet for ArchiveDarkTheme {
    fn symbol_detected(&self) -> &'static str {
        "□" // Empty box - awaiting classification
    }

    fn symbol_analyzing(&self) -> &'static str {
        "▢" // Boxed outline - under review
    }

    fn symbol_rejected(&self) -> &'static str {
        "▯" // Hollow box - declassified/removed
    }

    fn symbol_signal(&self) -> &'static str {
        "▣" // Box with center - cataloged entry
    }

    fn symbol_playing(&self) -> &'static str {
        "▶" // Standard play - monitoring record
    }

    fn symbol_completed(&self) -> &'static str {
        "◧" // Box with diagonal - permanently filed
    }

    fn progress_empty(&self) -> &'static str {
        "▭" // Empty filing bar
    }

    fn progress_full(&self) -> &'static str {
        "▬" // Filled filing bar
    }

    fn spectrum_baseline(&self) -> char {
        '━' // Solid baseline - continuous monitoring
    }

    fn spectrum_window_char(&self) -> char {
        '▭' // Scanning marker - active catalog region
    }

    fn window_bullet(&self) -> &'static str {
        "▪" // Small square list marker
    }

    fn header_border(&self) -> char {
        '─' // Simple line document separator
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for ArchiveDarkTheme {}

impl Theme for ArchiveDarkTheme {
    fn name(&self) -> &str {
        "archive-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Archive light theme - daylight archival operations
pub struct ArchiveLightTheme;

impl ColorScheme for ArchiveLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(95, 75, 55) // Dark ink on parchment
    }

    fn secondary(&self) -> Color {
        Color::Rgb(70, 60, 50) // Deep filing gray
    }

    fn accent(&self) -> Color {
        Color::Rgb(115, 90, 65) // Emphasized entries
    }

    fn background(&self) -> Color {
        Color::Rgb(242, 235, 225) // Fresh parchment
    }

    fn foreground(&self) -> Color {
        Color::Rgb(35, 30, 25) // Black administrative ink
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(105, 85, 60) // New entry brown
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(75, 65, 55) // Under classification
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(145, 140, 130) // Dismissed filing
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(95, 75, 55) // Confirmed record
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(85, 70, 55) // Active monitoring
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(75, 65, 55) // Filed permanently
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(70, 95, 70) // Verified - subtle document green
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(105, 85, 60) // Standard
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(115, 85, 60) // Degraded
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(135, 130, 120) // Empty
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(110, 85, 65) // Corrupted
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(125, 120, 110) // Unclassified
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(95, 75, 55) // Document title ink
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(85, 70, 55) // Catalog indicator
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(145, 140, 130) // Dimmed notes
    }

    fn window_header(&self) -> Color {
        Color::Rgb(105, 85, 60) // Section headers
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(139, 115, 85) // Medium brown background
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(30, 25, 20) // Very dark brown text
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(60, 50, 40) // Dark brown
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(80, 70, 55) // Darker brown
    }
}

impl SymbolSet for ArchiveLightTheme {
    fn symbol_detected(&self) -> &'static str {
        "□"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "▢"
    }

    fn symbol_rejected(&self) -> &'static str {
        "▯"
    }

    fn symbol_signal(&self) -> &'static str {
        "▣"
    }

    fn symbol_playing(&self) -> &'static str {
        "▶"
    }

    fn symbol_completed(&self) -> &'static str {
        "◧"
    }

    fn progress_empty(&self) -> &'static str {
        "▭"
    }

    fn progress_full(&self) -> &'static str {
        "▬"
    }

    fn spectrum_baseline(&self) -> char {
        '━'
    }

    fn spectrum_window_char(&self) -> char {
        '▭'
    }

    fn window_bullet(&self) -> &'static str {
        "▪"
    }

    fn header_border(&self) -> char {
        '─'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for ArchiveLightTheme {}

impl Theme for ArchiveLightTheme {
    fn name(&self) -> &str {
        "archive-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
