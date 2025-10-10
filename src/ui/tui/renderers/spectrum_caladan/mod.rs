//! Caladan wave-based spectrum visualization

mod frequency_labels;
mod wave_animation;
mod window_detail;

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
use window_detail::render_window_detail_row;

pub fn render_spectrum(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    use std::sync::OnceLock;
    use std::time::Instant;

    static ANIMATION_START: OnceLock<Instant> = OnceLock::new();
    let start = ANIMATION_START.get_or_init(Instant::now);
    let animation_time = start.elapsed().as_secs_f32();

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

    // Split the inner area to separate the bottom row for the window detail box
    let layout = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(3), // Top 3 rows (freq labels, spectrum, window freq labels)
            ratatui::layout::Constraint::Length(3), // Bottom box (1 content + 2 borders)
        ])
        .split(inner);

    let top_area = layout[0];
    let window_detail_area = layout[1];

    // Render top 3 rows
    let top_lines = vec![
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

    let top_paragraph = Paragraph::new(top_lines);
    f.render_widget(block, area);
    f.render_widget(top_paragraph, top_area);

    // Create a subtle box for the window detail row with dim bracket color
    let window_detail_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(
            Style::default()
                .fg(bracket_color)
                .add_modifier(Modifier::DIM),
        )
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

    let window_detail_inner = window_detail_block.inner(window_detail_area);
    let window_detail_width = window_detail_inner.width as usize;

    let window_detail_line = render_window_detail_row(
        window_detail_width,
        window_start,
        window_width,
        model,
        theme,
    );

    let window_detail_paragraph = Paragraph::new(vec![window_detail_line]);

    f.render_widget(window_detail_block, window_detail_area);
    f.render_widget(window_detail_paragraph, window_detail_inner);
}

#[cfg(test)]
mod tests;
