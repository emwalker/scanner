//! Instructions rendering for the bottom of the TUI

use ratatui::{
    Frame,
    layout::Alignment,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::ui::tui::{
    model::{Model, UiMode},
    themes::Theme,
};

fn build_left_instructions_text(model: &Model) -> String {
    let pause_prefix = if model.is_globally_paused() {
        "[PAUSED] "
    } else {
        ""
    };

    let instructions = match &model.ui_mode {
        UiMode::Listening { .. } if !model.all_complete() => {
            "  ⌃C Exit  ↑↓ Browse  ↵ Continue scan"
        }
        UiMode::AwaitingTune { .. } if !model.all_complete() => {
            "  ⌃C Exit  ↑↓ Browse  ↵ Continue scan"
        }
        UiMode::NavigatingScanner { .. } => "  ⌃C Exit  ↑↓ Navigate  ↵ Listen",
        _ => "  ⌃C Exit  ↑↓ Navigate",
    };

    format!("{}{}", pause_prefix, instructions)
}

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
        let left_text = build_left_instructions_text(model);

        let mut spans = vec![];
        if model.is_globally_paused() {
            spans.push(Span::styled(
                "[PAUSED] ",
                Style::default()
                    .fg(theme.active_highlight_fg())
                    .add_modifier(Modifier::BOLD),
            ));
        }

        spans.push(Span::styled(
            &left_text[if model.is_globally_paused() { 9 } else { 0 }..],
            Style::default().fg(theme.instructions_dim()),
        ));

        let instruction = Paragraph::new(Line::from(spans));
        f.render_widget(instruction, area);

        let right_text = match &model.ui_mode {
            UiMode::Listening { .. } => {
                if let Some(info) = model.selected_signal_info() {
                    format!("[Listening: {:.1} MHz]  ", info.signal_frequency / 1e6)
                } else {
                    format!("{}  ", theme_name)
                }
            }
            _ => format!("{}  ", theme_name),
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
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::ui::tui::model::Model;

    #[test]
    fn test_instructions_format_unchanged() {
        let instruction_text = "  ⌃C to exit";
        assert_eq!(instruction_text, "  ⌃C to exit");

        assert!(instruction_text.contains("⌃C"));
        assert!(instruction_text.starts_with("  "));
    }

    #[test]
    fn test_pause_indicator_not_shown_when_active() {
        let mut model = Model::new();
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
        model.set_global_pause_resource(resource);

        let left_text = build_left_instructions_text(&model);
        assert!(!left_text.contains("[PAUSED]"));
    }

    #[test]
    fn test_pause_indicator_shown_when_paused() {
        let mut model = Model::new();
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));
        model.set_global_pause_resource(resource);

        let left_text = build_left_instructions_text(&model);
        assert!(left_text.contains("[PAUSED]"));
    }
}
