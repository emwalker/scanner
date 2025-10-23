//! Caladan wave-based spectrum visualization

mod frequency_labels;
mod wave_animation;

use crate::ui::tui::{
    model::{FocusState, Model},
    themes::Theme,
};
use frequency_labels::{render_frequency_labels, render_window_frequency_labels};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, Paragraph},
};
use wave_animation::render_full_spectrum_row;

pub fn render_spectrum(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::Instant;

    struct AnimationState {
        last_update: Instant,
        accumulated_time: f32,
        was_paused: bool,
    }

    static ANIMATION_STATE: OnceLock<Mutex<AnimationState>> = OnceLock::new();
    let state = ANIMATION_STATE.get_or_init(|| {
        Mutex::new(AnimationState {
            last_update: Instant::now(),
            accumulated_time: 0.0,
            was_paused: false,
        })
    });

    let animation_time = if let Ok(mut state) = state.try_lock() {
        let now = Instant::now();
        let is_paused = model.is_globally_paused();

        if !is_paused {
            let delta = now.duration_since(state.last_update).as_secs_f32();
            state.accumulated_time += delta;
        }

        state.last_update = now;
        state.was_paused = is_paused;
        state.accumulated_time
    } else {
        0.0
    };

    if area.height < 4 {
        return;
    }

    let fm_start = 88.0e6;
    let fm_end = 108.0e6;
    let fm_range = fm_end - fm_start;
    let window_width = 2.4e6;

    // Use selected candidate's center frequency if in interactive mode, otherwise current window
    let window_start = if model.is_interactive() {
        model
            .selected_candidate_info()
            .map(|info| info.metadata.center_frequency_hz - window_width / 2.0)
    } else {
        model.windows.get(&model.current_window).map(|w| {
            if !w.candidates.is_empty() {
                let center = w.candidates.iter().map(|c| c.frequency_hz).sum::<f64>()
                    / w.candidates.len() as f64;
                center - window_width / 2.0
            } else {
                fm_start
            }
        })
    };

    let bracket_color = Color::Rgb(160, 200, 220);
    let has_focus = matches!(model.focus_state, FocusState::Spectrum);

    let border_style = if has_focus {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::DIM)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .border_set(ratatui::symbols::border::Set {
            top_left: "╭",
            top_right: "╮",
            bottom_left: "╰",
            bottom_right: "╯",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: "─",
            horizontal_bottom: "─",
        })
        .padding(ratatui::widgets::Padding::horizontal(1));

    let inner = block.inner(area);
    let content_width = inner.width as usize;

    // Render the 3 rows (freq labels, spectrum, window freq labels)
    let lines = vec![
        render_frequency_labels(content_width, fm_start, fm_range, theme),
        render_full_spectrum_row(
            content_width,
            fm_start,
            fm_range,
            window_start,
            window_width,
            theme,
            animation_time,
        ),
        render_window_frequency_labels(content_width, window_start, window_width, theme),
    ];

    let paragraph = Paragraph::new(lines);
    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests;
