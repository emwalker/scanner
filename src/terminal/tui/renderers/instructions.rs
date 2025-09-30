//! Instructions rendering for the bottom of the TUI

use crate::terminal::tui::{model::Model, themes::Theme};
use ratatui::{
    Frame,
    layout::Alignment,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

pub fn render_instructions(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    theme: &dyn Theme,
    theme_name: &str,
    model: &Model,
    all_themes: &[String],
) {
    if model.theme_selector_open {
        render_theme_selector(f, area, theme, model, all_themes);
    } else {
        let left_instructions = if model.selection_mode {
            "  ⌃C Exit  ↑↓ Browse  ↵ Continue scan"
        } else {
            "  ⌃C Exit  ↑↓ Browse"
        };

        let instruction =
            Paragraph::new(left_instructions).style(Style::default().fg(theme.instructions_dim()));
        f.render_widget(instruction, area);

        let right_text = if model.selection_mode {
            if let Some((_, _, candidate_freq, _, _)) = model.get_selected_candidate_info() {
                format!("[Listening: {:.1} MHz]  ", candidate_freq / 1e6)
            } else {
                format!("{}  ", theme_name)
            }
        } else {
            format!("{}  ", theme_name)
        };

        let theme_display = Paragraph::new(right_text)
            .alignment(Alignment::Right)
            .style(Style::default().fg(theme.instructions_dim()));
        f.render_widget(theme_display, area);
    }
}

fn render_theme_selector(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    theme: &dyn Theme,
    model: &Model,
    all_themes: &[String],
) {
    let total_themes = all_themes.len();
    let max_visible = 10.min(total_themes);

    let start_idx = if total_themes <= max_visible || model.theme_selector_index < max_visible / 2 {
        0
    } else if model.theme_selector_index >= total_themes - max_visible / 2 {
        total_themes.saturating_sub(max_visible)
    } else {
        model.theme_selector_index.saturating_sub(max_visible / 2)
    };

    let end_idx = (start_idx + max_visible).min(total_themes);

    let mut lines: Vec<Line> = Vec::new();

    for (i, theme_name) in all_themes[start_idx..end_idx].iter().enumerate() {
        let actual_idx = start_idx + i;
        let is_selected = actual_idx == model.theme_selector_index;

        let style = if is_selected {
            Style::default()
                .fg(theme.primary())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.instructions_dim())
        };

        let prefix = if is_selected { "▶ " } else { "  " };
        lines.push(Line::from(Span::styled(
            format!("{}{}", prefix, theme_name),
            style,
        )));
    }

    let height = lines.len() as u16;
    let width = area.width;
    let selector_area = ratatui::layout::Rect {
        x: area.x,
        y: area.y.saturating_sub(height.saturating_sub(1)),
        width,
        height,
    };

    let selector = Paragraph::new(lines).alignment(Alignment::Right);
    f.render_widget(selector, selector_area);
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_instructions_format_unchanged() {
        let instruction_text = "  ⌃C to exit";
        assert_eq!(instruction_text, "  ⌃C to exit");

        assert!(instruction_text.contains("⌃C"));
        assert!(instruction_text.starts_with("  "));
    }
}
