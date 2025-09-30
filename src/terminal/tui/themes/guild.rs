//! Guild Navigator theme implementations - deep space communication arrays
//!
//! This theme captures the mysterious, spice-influenced aesthetic of Guild
//! Navigator technology as seen in Denis Villeneuve's Dune. The interface
//! represents the communication monitoring systems used by Guild Navigator
//! crews during interstellar transit.
//!
//! Design Philosophy:
//! - Iridescent color palette mixing blues, purples, and oranges with deep blacks
//! - Subtle organic curves contrasting with technical precision
//! - Communication array terminology grounded in radio operations
//! - Mystical yet functional - technology influenced by prescient navigation

use super::{ColorScheme, SymbolSet, TextStyle, Theme};
use ratatui::style::Color;

/// Guild Navigator dark theme - deep space communication arrays
///
/// Color palette inspired by:
/// - Deep space blacks and grays (void of space)
/// - Iridescent spice blues and purples (Navigator consciousness)
/// - Subtle orange highlights (spice gas glow)
/// - Silver-gray technical elements (Guild technology)
pub struct GuildDarkTheme;

impl ColorScheme for GuildDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(140, 160, 200) // Pale iridescent blue
    }

    fn secondary(&self) -> Color {
        Color::Rgb(160, 140, 180) // Soft purple-blue
    }

    fn accent(&self) -> Color {
        Color::Rgb(200, 140, 100) // Muted spice orange
    }

    fn background(&self) -> Color {
        Color::Reset // Deep void (terminal default)
    }

    fn foreground(&self) -> Color {
        Color::Rgb(180, 190, 210) // Soft silver-blue
    }

    // Status colors - iridescent communication states
    fn status_detected(&self) -> Color {
        Color::Rgb(140, 160, 180) // Pale detection blue
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(160, 140, 200) // Prescient purple
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(80, 85, 90) // Deep gray void
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(120, 150, 190) // Clear signal blue
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(180, 130, 160) // Active purple-pink
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(100, 120, 140) // Muted slate blue
    }

    // Audio quality colors - spice-influenced gradient
    fn quality_good(&self) -> Color {
        Color::Rgb(120, 160, 180) // Clear spice blue
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(160, 150, 140) // Neutral gray-blue
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(140, 120, 100) // Dim orange-gray
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(90, 90, 95) // Deep void gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(120, 110, 100) // Static interference gray
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(100, 100, 110) // Uncertain gray-blue
    }

    // UI element colors - mystical navigation interface
    fn header_accent(&self) -> Color {
        Color::Rgb(160, 140, 180) // Soft prescient purple
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(140, 160, 200) // Scanning array blue
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(100, 105, 115) // Dim navigation gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(180, 150, 120) // Muted spice accent
    }
}

impl SymbolSet for GuildDarkTheme {
    // Status symbols - organic curves with technical precision
    fn symbol_detected(&self) -> &'static str {
        "◯" // Open circle - initial detection
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉" // Circled dot - prescient analysis
    }

    fn symbol_rejected(&self) -> &'static str {
        "⊘" // Circle with diagonal - filtered transmission
    }

    fn symbol_signal(&self) -> &'static str {
        "◆" // Diamond - locked signal
    }

    fn symbol_playing(&self) -> &'static str {
        "▷" // Right-pointing triangle - active relay
    }

    fn symbol_completed(&self) -> &'static str {
        "◇" // Open diamond - archived transmission
    }

    // Progress bar characters - flowing organic bars
    fn progress_empty(&self) -> &'static str {
        "▁" // Light bar (prescient path not yet traveled)
    }

    fn progress_full(&self) -> &'static str {
        "▇" // Heavy bar (spice-enhanced completion)
    }

    // Spectrum visualization - organic scanning waves
    fn spectrum_baseline(&self) -> char {
        '⋯' // Horizontal ellipsis (potential frequencies)
    }

    fn spectrum_window_char(&self) -> char {
        '═' // Double horizontal line (active scan window)
    }

    // Window header decoration - Guild insignia style
    fn window_bullet(&self) -> &'static str {
        "◈" // White diamond with X (Guild mark)
    }

    // Header border - flowing energy line
    fn header_border(&self) -> char {
        '═' // Double line (energy barrier)
    }
}

impl TextStyle for GuildDarkTheme {
    fn title(&self) -> &'static str {
        "TRANSMISSION ARRAY"
    }

    fn subtitle(&self) -> &'static str {
        "Guild Navigator • Deep Space Relay • 88–108 MHz Intercept"
    }

    // Status text - Guild communication terminology
    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Analyzing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Rejected"
    }

    fn status_signal_text(&self) -> &'static str {
        "Locked"
    }

    fn status_playing_text(&self) -> &'static str {
        "Relaying"
    }

    fn status_completed_text(&self) -> &'static str {
        "Archived"
    }
}

impl Theme for GuildDarkTheme {
    fn name(&self) -> &str {
        "guild-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Guild Navigator light theme - spice-saturated consciousness
///
/// Color palette inspired by:
/// - Bright spice orange atmosphere
/// - Deep purple shadows (concentrated spice)
/// - Blue-gray technical readouts
/// - Cream backgrounds (pressurized chamber walls)
pub struct GuildLightTheme;

impl ColorScheme for GuildLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(60, 80, 140) // Deep navigation blue
    }

    fn secondary(&self) -> Color {
        Color::Rgb(100, 60, 140) // Deep prescient purple
    }

    fn accent(&self) -> Color {
        Color::Rgb(200, 100, 40) // Bright spice orange
    }

    fn background(&self) -> Color {
        Color::Rgb(240, 235, 225) // Cream chamber walls
    }

    fn foreground(&self) -> Color {
        Color::Rgb(40, 45, 60) // Deep blue-black text
    }

    // Status colors - saturated spice palette
    fn status_detected(&self) -> Color {
        Color::Rgb(80, 100, 160) // Detection blue
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(120, 60, 160) // Prescient purple
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(120, 115, 110) // Neutral gray
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(60, 90, 160) // Strong signal blue
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(140, 70, 140) // Active purple
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(80, 100, 120) // Archived blue-gray
    }

    // Audio quality colors
    fn quality_good(&self) -> Color {
        Color::Rgb(60, 100, 160) // Clear transmission blue
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(160, 120, 80) // Moderate orange-brown
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(140, 100, 60) // Weak signal brown
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(120, 115, 110) // No signal gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(130, 110, 90) // Static interference
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(110, 105, 100) // Unknown gray
    }

    // UI element colors
    fn header_accent(&self) -> Color {
        Color::Rgb(120, 60, 160) // Prescient purple accent
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(60, 90, 160) // Scanning blue
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(120, 115, 110) // Dim text gray
    }

    fn window_header(&self) -> Color {
        Color::Rgb(200, 100, 40) // Spice orange header
    }
}

impl SymbolSet for GuildLightTheme {
    // Same organic symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "◯"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◉"
    }

    fn symbol_rejected(&self) -> &'static str {
        "⊘"
    }

    fn symbol_signal(&self) -> &'static str {
        "◆"
    }

    fn symbol_playing(&self) -> &'static str {
        "▷"
    }

    fn symbol_completed(&self) -> &'static str {
        "◇"
    }

    fn progress_empty(&self) -> &'static str {
        "▁"
    }

    fn progress_full(&self) -> &'static str {
        "▇"
    }

    fn spectrum_baseline(&self) -> char {
        '⋯'
    }

    fn spectrum_window_char(&self) -> char {
        '═'
    }

    fn window_bullet(&self) -> &'static str {
        "◈"
    }

    fn header_border(&self) -> char {
        '═'
    }
}

impl TextStyle for GuildLightTheme {
    fn title(&self) -> &'static str {
        "TRANSMISSION ARRAY"
    }

    fn subtitle(&self) -> &'static str {
        "Guild Navigator • Deep Space Relay • 88–108 MHz Intercept"
    }

    fn status_detected_text(&self) -> &'static str {
        "Detected"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "Analyzing"
    }

    fn status_rejected_text(&self) -> &'static str {
        "Rejected"
    }

    fn status_signal_text(&self) -> &'static str {
        "Locked"
    }

    fn status_playing_text(&self) -> &'static str {
        "Relaying"
    }

    fn status_completed_text(&self) -> &'static str {
        "Archived"
    }
}

impl Theme for GuildLightTheme {
    fn name(&self) -> &str {
        "guild-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}