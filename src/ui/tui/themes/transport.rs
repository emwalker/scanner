//! Transport Monitor theme - analog precision instrumentation aesthetic
//!
//! Inspired by the spherical Imperial Transport from Denis Villeneuve's Dune films,
//! this theme imagines the monitoring equipment as deliberately non-computational
//! analog precision instruments. Post-Butlerian Jihad Imperial technology would
//! have an almost retro-futuristic quality: handcrafted displays, mechanical
//! elegance, and the warm character of analog signal processing.
//!
//! DESIGN RATIONALE:
//!
//! Core Concept: Analog Laboratory Equipment
//! This theme rejects cold digital aesthetics in favor of warm, organic materials
//! reminiscent of vintage laboratory equipment and high-end analog audio gear.
//! Think of communications officers aboard the Transport using equipment that
//! feels more like a 1960s analog oscilloscope than a modern digital display.
//!
//! Color Palette: Warm Phosphor and Aged Materials
//! - Amber phosphor (220, 180, 100) - warm CRT glow, like vintage oscilloscopes
//! - Aged brass (160, 140, 100) - patinated metal controls and bezels
//! - Warm cream (240, 235, 220) - aged instrument panels and backlit displays
//! - Deep charcoal (25, 22, 18) - oxidized metal chassis
//! - Copper glow (200, 150, 90) - active indicator lamps
//! - Verdigris accents (140, 150, 130) - aged brass oxidation for secondary info
//!
//! Why these colors work together:
//! The palette evokes handcrafted precision instruments that have been maintained
//! for generations. Warm ambers and coppers suggest vacuum tubes and filament
//! lighting. The aged brass tones convey mechanical quality and long service.
//! Everything feels tangible, physical, human-operated.
//!
//! Visual Language: Mechanical Precision Indicators
//! - Analog meter symbols (◐, ◑, ◒, ◓) suggesting needle positions on dials
//! - Geometric precision markers (⊡, ⊠, ⊞) like mechanical indicator panels
//! - Horizontal bars (▬, ━) reminiscent of strip chart recorders
//! - Minimal decoration - function follows form in precision equipment
//! - Consistent geometric language suggests calibrated instrumentation
//!
//! Terminology: Technical Laboratory Language
//! - "SIGNAL ANALYSIS STATION" - precise, technical, human-operated
//! - "Precision RF Monitor" - emphasizes analog measurement approach
//! - "Sampled/Measuring/Reading" - language of analog instrumentation
//! - Avoids computational or automated terminology
//! - Professional laboratory vocabulary, not military command
//!
//! Information Architecture Suggestions:
//! 1. De-emphasize digital precision: Round frequencies to .0 instead of .00
//! 2. Add "warmth" through subtle color variations in similar states
//! 3. Use progress bars that feel like analog meters moving smoothly
//! 4. Status indicators suggest mechanical state changes, not binary switches
//! 5. Window headers feel like engraved instrument panel labels
//!
//! Design Differentiation from Other Themes:
//! - vs. Arrakis: Warm vs. cool, refined vs. survival, crafted vs. functional
//! - vs. Imperial: Analog vs. digital, warm vs. cold, organic vs. geometric
//! - vs. Herald: Laboratory vs. ceremonial, working vs. observing
//! - vs. Guild: Precision vs. mystery, measurement vs. navigation
//!
//! The Imperial Transport represents ancient sophistication - technology refined
//! over 10,000 years to a state of deliberate mechanical elegance. This isn't
//! primitive technology; it's the opposite: so advanced it has returned to
//! analog precision as the ultimate expression of reliability and human control.

use ratatui::style::Color;

use super::{ColorScheme, SymbolSet, TextStyle, Theme};

/// Transport dark theme - analog precision instruments with warm phosphor displays
pub struct TransportDarkTheme;

impl ColorScheme for TransportDarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(220, 180, 100) // Amber phosphor - primary readouts
    }

    fn secondary(&self) -> Color {
        Color::Rgb(160, 140, 100) // Aged brass - secondary indicators
    }

    fn accent(&self) -> Color {
        Color::Rgb(200, 150, 90) // Copper glow - active elements
    }

    fn background(&self) -> Color {
        Color::Rgb(25, 22, 18) // Deep oxidized metal
    }

    fn foreground(&self) -> Color {
        Color::Rgb(210, 195, 165) // Warm aged parchment
    }

    // Status colors - analog meter state progression
    fn status_detected(&self) -> Color {
        Color::Rgb(180, 160, 120) // Muted brass - initial reading
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(200, 180, 100) // Warm amber - measurement in progress
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(80, 75, 65) // Dark tarnish - filtered reading
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(220, 180, 100) // Bright phosphor - confirmed measurement
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(190, 140, 80) // Warm copper - active monitoring
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(140, 150, 130) // Verdigris - archived measurement
    }

    // Audio quality colors - analog signal strength indicators
    fn quality_good(&self) -> Color {
        Color::Rgb(160, 170, 140) // Strong verdigris - excellent signal
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(200, 160, 90) // Medium amber - acceptable reading
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(160, 130, 90) // Dull brass - weak signal
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(75, 70, 60) // Dark oxidation - no reading
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(140, 120, 85) // Tarnished brass - interference
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(90, 85, 75) // Neutral dark - uncalibrated
    }

    // UI element colors - warm instrument panel hierarchy
    fn header_accent(&self) -> Color {
        Color::Rgb(210, 170, 95) // Bright brass - panel labels
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(200, 150, 90) // Copper glow - scanning indicator
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(110, 100, 85) // Subdued brass - secondary text
    }

    fn window_header(&self) -> Color {
        Color::Rgb(190, 160, 105) // Polished brass - section markers
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(50, 35, 20) // Dark oxidized metal
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(220, 180, 100) // Amber phosphor
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(200, 150, 90) // Copper glow
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(160, 170, 140) // Verdigris
    }

    fn selected_row_bg(&self) -> Color {
        Color::Rgb(65, 90, 110)
    }

    fn selected_row_fg(&self) -> Color {
        Color::Rgb(230, 235, 240)
    }
}

impl SymbolSet for TransportDarkTheme {
    // Status symbols - analog meter needle positions
    fn symbol_detected(&self) -> &'static str {
        "◐" // Quarter dial - initial detection
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◑" // Half dial - measurement in progress
    }

    fn symbol_rejected(&self) -> &'static str {
        "◯" // Empty dial - reading discarded
    }

    fn symbol_signal(&self) -> &'static str {
        "◉" // Full dial - confirmed reading
    }

    fn symbol_playing(&self) -> &'static str {
        "◒" // Three-quarter dial - active monitoring
    }

    fn symbol_completed(&self) -> &'static str {
        "⊡" // Squared circle - measurement logged
    }

    // Progress bar characters - analog strip chart recorder
    fn progress_empty(&self) -> &'static str {
        "─" // Light trace line
    }

    fn progress_full(&self) -> &'static str {
        "━" // Heavy trace line - smooth analog progression
    }

    // Spectrum visualization - precision frequency markers
    fn spectrum_baseline(&self) -> char {
        '·' // Calibration dots
    }

    fn spectrum_window_char(&self) -> char {
        '▬' // Scanning indicator bar
    }

    // Window header decoration - engraved panel label
    fn window_bullet(&self) -> &'static str {
        "⊞" // Crosshair marker - precision indicator
    }

    // Header border - instrument panel divider
    fn header_border(&self) -> char {
        '─' // Clean mechanical separation
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for TransportDarkTheme {}

impl Theme for TransportDarkTheme {
    fn name(&self) -> &str {
        "transport-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }
}

/// Transport light theme - daylight laboratory with warm natural illumination
pub struct TransportLightTheme;

impl ColorScheme for TransportLightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(120, 90, 50) // Deep bronze - primary readouts in daylight
    }

    fn secondary(&self) -> Color {
        Color::Rgb(100, 80, 55) // Dark aged brass
    }

    fn accent(&self) -> Color {
        Color::Rgb(140, 100, 60) // Warm copper accent
    }

    fn background(&self) -> Color {
        Color::Rgb(240, 235, 220) // Warm cream instrument panel
    }

    fn foreground(&self) -> Color {
        Color::Rgb(35, 30, 25) // Deep warm text
    }

    // Status colors - daylight laboratory indicators
    fn status_detected(&self) -> Color {
        Color::Rgb(130, 105, 70) // Bronze initial reading
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(110, 85, 50) // Deep amber measuring
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(140, 130, 115) // Neutral tarnish
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(120, 90, 50) // Primary bronze confirmed
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(130, 95, 55) // Warm copper active
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(90, 100, 85) // Deep verdigris archived
    }

    // Audio quality colors - daylight signal indicators
    fn quality_good(&self) -> Color {
        Color::Rgb(80, 100, 70) // Forest verdigris - excellent
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(130, 100, 60) // Medium bronze
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(120, 95, 65) // Dull bronze
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(130, 120, 105) // Neutral gray
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(115, 90, 65) // Tarnished
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(120, 110, 95) // Uncertain neutral
    }

    // UI element colors - daylight instrument panel
    fn header_accent(&self) -> Color {
        Color::Rgb(115, 85, 50) // Deep brass labels
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(130, 95, 55) // Copper scanning
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(140, 130, 115) // Subdued text
    }

    fn window_header(&self) -> Color {
        Color::Rgb(125, 100, 65) // Bronze section markers
    }

    fn selection_highlight(&self) -> Color {
        Color::Rgb(255, 200, 0)
    }

    fn active_highlight_bg(&self) -> Color {
        Color::Rgb(120, 90, 50) // Deep bronze
    }

    fn active_highlight_fg(&self) -> Color {
        Color::Rgb(240, 235, 220) // Warm cream
    }

    fn active_highlight_status(&self) -> Color {
        Color::Rgb(140, 100, 60) // Warm copper
    }

    fn active_highlight_quality(&self) -> Color {
        Color::Rgb(80, 100, 70) // Forest verdigris
    }

    fn selected_row_bg(&self) -> Color {
        Color::Rgb(50, 75, 95)
    }

    fn selected_row_fg(&self) -> Color {
        Color::Rgb(255, 255, 255)
    }
}

impl SymbolSet for TransportLightTheme {
    // Same analog instrumentation symbols as dark theme
    fn symbol_detected(&self) -> &'static str {
        "◐"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◑"
    }

    fn symbol_rejected(&self) -> &'static str {
        "◯"
    }

    fn symbol_signal(&self) -> &'static str {
        "◉"
    }

    fn symbol_playing(&self) -> &'static str {
        "◒"
    }

    fn symbol_completed(&self) -> &'static str {
        "⊡"
    }

    fn progress_empty(&self) -> &'static str {
        "─"
    }

    fn progress_full(&self) -> &'static str {
        "━"
    }

    fn spectrum_baseline(&self) -> char {
        '·'
    }

    fn spectrum_window_char(&self) -> char {
        '▬'
    }

    fn window_bullet(&self) -> &'static str {
        "⊞"
    }

    fn header_border(&self) -> char {
        '─'
    }

    fn selection_indicator(&self) -> &'static str {
        "→"
    }
}

impl TextStyle for TransportLightTheme {}

impl Theme for TransportLightTheme {
    fn name(&self) -> &str {
        "transport-light"
    }

    fn is_dark(&self) -> bool {
        false
    }
}
