use crate::ui::tui::themes::Theme;
use ratatui::style::Color;

pub(super) struct MockTheme;

impl crate::ui::tui::themes::ColorScheme for MockTheme {
    fn primary(&self) -> Color {
        Color::Blue
    }
    fn secondary(&self) -> Color {
        Color::Gray
    }
    fn accent(&self) -> Color {
        Color::Cyan
    }
    fn background(&self) -> Color {
        Color::Black
    }
    fn foreground(&self) -> Color {
        Color::White
    }
    fn status_detected(&self) -> Color {
        Color::Yellow
    }
    fn status_analyzing(&self) -> Color {
        Color::Yellow
    }
    fn status_rejected(&self) -> Color {
        Color::Gray
    }
    fn status_signal(&self) -> Color {
        Color::Green
    }
    fn status_playing(&self) -> Color {
        Color::Green
    }
    fn status_completed(&self) -> Color {
        Color::Blue
    }
    fn quality_good(&self) -> Color {
        Color::Green
    }
    fn quality_moderate(&self) -> Color {
        Color::Yellow
    }
    fn quality_poor(&self) -> Color {
        Color::Red
    }
    fn quality_no_audio(&self) -> Color {
        Color::Gray
    }
    fn quality_static(&self) -> Color {
        Color::DarkGray
    }
    fn quality_unknown(&self) -> Color {
        Color::Gray
    }
    fn header_accent(&self) -> Color {
        Color::Cyan
    }
    fn spectrum_window(&self) -> Color {
        Color::Cyan
    }
    fn instructions_dim(&self) -> Color {
        Color::DarkGray
    }
    fn window_header(&self) -> Color {
        Color::Blue
    }
    fn selection_highlight(&self) -> Color {
        Color::Cyan
    }
    fn active_highlight_bg(&self) -> Color {
        Color::DarkGray
    }
    fn active_highlight_fg(&self) -> Color {
        Color::White
    }
    fn active_highlight_status(&self) -> Color {
        Color::LightBlue
    }
    fn active_highlight_quality(&self) -> Color {
        Color::LightGreen
    }
}

impl crate::ui::tui::themes::SymbolSet for MockTheme {
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
        "░"
    }
    fn progress_full(&self) -> &'static str {
        "█"
    }
    fn spectrum_baseline(&self) -> char {
        '≈'
    }
    fn spectrum_window_char(&self) -> char {
        '▬'
    }
    fn window_bullet(&self) -> &'static str {
        "•"
    }
    fn header_border(&self) -> char {
        '─'
    }
    fn selection_indicator(&self) -> &'static str {
        ">"
    }
}

impl crate::ui::tui::themes::TextStyle for MockTheme {
    fn title(&self) -> &'static str {
        "SCANNER"
    }
    fn subtitle(&self) -> &'static str {
        "FM Band Monitor"
    }
    fn status_detected_text(&self) -> &'static str {
        "detected"
    }
    fn status_analyzing_text(&self) -> &'static str {
        "analyzing"
    }
    fn status_rejected_text(&self) -> &'static str {
        "rejected"
    }
    fn status_signal_text(&self) -> &'static str {
        "signal"
    }
    fn status_playing_text(&self) -> &'static str {
        "playing"
    }
    fn status_completed_text(&self) -> &'static str {
        "completed"
    }
}

impl Theme for MockTheme {
    fn name(&self) -> &str {
        "mock"
    }
    fn is_dark(&self) -> bool {
        true
    }
}

mod frequency_labels;
mod pause_animation;
mod wave_animation;
