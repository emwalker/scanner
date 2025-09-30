//! Caladan theme - Philosophical water world aesthetic

use super::{ColorScheme, SymbolSet, TextStyle, Theme, UiVariant};
use ratatui::style::Color;

pub struct DarkTheme;

impl ColorScheme for DarkTheme {
    fn primary(&self) -> Color {
        Color::Rgb(140, 180, 200)
    }

    fn secondary(&self) -> Color {
        Color::Rgb(100, 130, 150)
    }

    fn accent(&self) -> Color {
        Color::Rgb(160, 200, 220)
    }

    fn background(&self) -> Color {
        Color::Rgb(12, 16, 20)
    }

    fn foreground(&self) -> Color {
        Color::Rgb(200, 220, 230)
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(120, 150, 170)
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(140, 180, 200)
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(60, 70, 80)
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(160, 200, 220)
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(150, 190, 210)
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(90, 110, 125)
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(140, 190, 210)
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(120, 150, 170)
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(100, 120, 135)
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(70, 80, 90)
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(90, 110, 125)
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(80, 100, 115)
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(160, 200, 220)
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(140, 180, 200)
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(90, 110, 125)
    }

    fn window_header(&self) -> Color {
        Color::Rgb(130, 170, 190)
    }
}

impl SymbolSet for DarkTheme {
    fn symbol_detected(&self) -> &'static str {
        "○"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐"
    }

    fn symbol_rejected(&self) -> &'static str {
        "·"
    }

    fn symbol_signal(&self) -> &'static str {
        "◉"
    }

    fn symbol_playing(&self) -> &'static str {
        "◉"
    }

    fn symbol_completed(&self) -> &'static str {
        "◯"
    }

    fn progress_empty(&self) -> &'static str {
        "▁"
    }

    fn progress_full(&self) -> &'static str {
        "█"
    }

    fn spectrum_baseline(&self) -> char {
        '≈'
    }

    fn spectrum_window_char(&self) -> char {
        '≋'
    }

    fn window_bullet(&self) -> &'static str {
        "◦"
    }

    fn header_border(&self) -> char {
        '·'
    }
}

impl TextStyle for DarkTheme {
    fn title(&self) -> &'static str {
        "Frequency Scan"
    }

    fn subtitle(&self) -> &'static str {
        "88.0–108.0 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "detecting"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "forming"
    }

    fn status_rejected_text(&self) -> &'static str {
        "static"
    }

    fn status_signal_text(&self) -> &'static str {
        "present"
    }

    fn status_playing_text(&self) -> &'static str {
        "listening"
    }

    fn status_completed_text(&self) -> &'static str {
        "detected"
    }
}

impl Theme for DarkTheme {
    fn name(&self) -> &str {
        "caladan-dark"
    }

    fn is_dark(&self) -> bool {
        true
    }

    fn ui_variant(&self) -> UiVariant {
        UiVariant::Caladan
    }
}

pub struct LightTheme;

impl ColorScheme for LightTheme {
    fn primary(&self) -> Color {
        Color::Rgb(40, 80, 110)
    }

    fn secondary(&self) -> Color {
        Color::Rgb(80, 110, 130)
    }

    fn accent(&self) -> Color {
        Color::Rgb(30, 70, 100)
    }

    fn background(&self) -> Color {
        Color::Rgb(235, 245, 250)
    }

    fn foreground(&self) -> Color {
        Color::Rgb(30, 50, 60)
    }

    fn status_detected(&self) -> Color {
        Color::Rgb(70, 110, 135)
    }

    fn status_analyzing(&self) -> Color {
        Color::Rgb(40, 80, 110)
    }

    fn status_rejected(&self) -> Color {
        Color::Rgb(160, 170, 180)
    }

    fn status_signal(&self) -> Color {
        Color::Rgb(30, 70, 100)
    }

    fn status_playing(&self) -> Color {
        Color::Rgb(35, 75, 105)
    }

    fn status_completed(&self) -> Color {
        Color::Rgb(120, 140, 155)
    }

    fn quality_good(&self) -> Color {
        Color::Rgb(35, 75, 105)
    }

    fn quality_moderate(&self) -> Color {
        Color::Rgb(60, 95, 120)
    }

    fn quality_poor(&self) -> Color {
        Color::Rgb(90, 115, 135)
    }

    fn quality_no_audio(&self) -> Color {
        Color::Rgb(150, 160, 170)
    }

    fn quality_static(&self) -> Color {
        Color::Rgb(110, 130, 145)
    }

    fn quality_unknown(&self) -> Color {
        Color::Rgb(130, 145, 160)
    }

    fn header_accent(&self) -> Color {
        Color::Rgb(30, 70, 100)
    }

    fn spectrum_window(&self) -> Color {
        Color::Rgb(50, 90, 120)
    }

    fn instructions_dim(&self) -> Color {
        Color::Rgb(120, 140, 155)
    }

    fn window_header(&self) -> Color {
        Color::Rgb(50, 90, 120)
    }
}

impl SymbolSet for LightTheme {
    fn symbol_detected(&self) -> &'static str {
        "○"
    }

    fn symbol_analyzing(&self) -> &'static str {
        "◐"
    }

    fn symbol_rejected(&self) -> &'static str {
        "·"
    }

    fn symbol_signal(&self) -> &'static str {
        "◉"
    }

    fn symbol_playing(&self) -> &'static str {
        "◉"
    }

    fn symbol_completed(&self) -> &'static str {
        "◯"
    }

    fn progress_empty(&self) -> &'static str {
        "▁"
    }

    fn progress_full(&self) -> &'static str {
        "█"
    }

    fn spectrum_baseline(&self) -> char {
        '≈'
    }

    fn spectrum_window_char(&self) -> char {
        '≋'
    }

    fn window_bullet(&self) -> &'static str {
        "◦"
    }

    fn header_border(&self) -> char {
        '·'
    }
}

impl TextStyle for LightTheme {
    fn title(&self) -> &'static str {
        "Frequency Scan"
    }

    fn subtitle(&self) -> &'static str {
        "88.0–108.0 MHz"
    }

    fn status_detected_text(&self) -> &'static str {
        "detecting"
    }

    fn status_analyzing_text(&self) -> &'static str {
        "forming"
    }

    fn status_rejected_text(&self) -> &'static str {
        "static"
    }

    fn status_signal_text(&self) -> &'static str {
        "present"
    }

    fn status_playing_text(&self) -> &'static str {
        "listening"
    }

    fn status_completed_text(&self) -> &'static str {
        "detected"
    }
}

impl Theme for LightTheme {
    fn name(&self) -> &str {
        "caladan-light"
    }

    fn is_dark(&self) -> bool {
        false
    }

    fn ui_variant(&self) -> UiVariant {
        UiVariant::Caladan
    }
}
