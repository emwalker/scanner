use crate::ui::tui::themes::Theme;
use ratatui::{
    style::Style,
    text::{Line, Span},
};

fn calculate_wave_offset(col: usize, animation_time: f32) -> f32 {
    let horizontal_phase = col as f32 * 0.3 + animation_time * 1.2;
    let amplitude_modulation = (horizontal_phase * 0.5).sin() * 0.3 + 0.7;

    let wave1 = horizontal_phase.sin() * 0.35 * amplitude_modulation;
    let wave2 = (col as f32 * 0.17 + animation_time * 0.7).sin() * 0.25;
    let wave3 = (col as f32 * 0.43 + animation_time * 0.9).sin() * 0.15;

    (wave1 + wave2 + wave3) * 0.5 + 0.5
}

pub(super) fn render_full_spectrum_row(
    width: usize,
    fm_start: f64,
    fm_range: f64,
    window_start: Option<f64>,
    window_width: f64,
    theme: &dyn Theme,
    animation_time: f32,
) -> Line<'static> {
    let mut spans = Vec::new();

    for col in 0..width {
        let pos = col as f64 / width as f64;
        let freq = fm_start + (fm_range * pos);

        let in_window = if let Some(ws) = window_start {
            freq >= ws && freq <= ws + window_width
        } else {
            false
        };

        let offset = calculate_wave_offset(col, animation_time);

        let ch = if in_window {
            '▬'
        } else if offset > 0.65 {
            '≋'
        } else if offset > 0.45 {
            '≈'
        } else if offset > 0.25 {
            '~'
        } else {
            '·'
        };

        let color = if in_window {
            theme.spectrum_window()
        } else {
            theme.secondary()
        };

        spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
    }

    Line::from(spans)
}
