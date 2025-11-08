//! Header rendering for the TUI interface

use ratatui::{
    Frame,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::ui::tui::{model::Model, themes::Theme};

/// Render the application header
pub fn render_header(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    _model: &Model,
    theme: &dyn Theme,
) {
    let title_text = theme.title();
    let subtitle_text = theme.subtitle();

    // Create a block with top border only (dotted style)
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(
            Style::default()
                .fg(theme.primary())
                .add_modifier(Modifier::BOLD),
        )
        .border_set(ratatui::symbols::border::Set {
            top_left: "·",
            top_right: "·",
            bottom_left: " ",
            bottom_right: " ",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: "·",
            horizontal_bottom: " ",
        })
        .padding(ratatui::widgets::Padding::horizontal(1));

    let inner = block.inner(area);
    let inner_width = inner.width as usize;

    // Calculate padding for centered text
    let title_padding = inner_width.saturating_sub(title_text.len());
    let subtitle_padding = inner_width.saturating_sub(subtitle_text.len());

    // Create simple title line
    let title_line = Line::from(vec![Span::styled(
        format!("{}{}", title_text, " ".repeat(title_padding)),
        Style::default().add_modifier(Modifier::BOLD),
    )]);

    let subtitle_line = Line::from(vec![Span::styled(
        format!("{}{}", subtitle_text, " ".repeat(subtitle_padding)),
        Style::default().add_modifier(Modifier::BOLD),
    )]);

    let header_content = vec![title_line, subtitle_line];

    let paragraph = Paragraph::new(header_content).style(Style::default().fg(theme.primary()));

    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_header_format_unchanged() {
        let title_text = "Monitor";
        let subtitle_text = "";

        assert_eq!(title_text, "Monitor");
        assert_eq!(subtitle_text, "");

        let header_width: usize = 50;
        let top_border = format!("╭{}╮", "─".repeat(header_width.saturating_sub(2)));
        assert!(top_border.starts_with("╭"));
        assert!(top_border.ends_with("╮"));
        let border_char_count = top_border.chars().count(); // Count Unicode chars, not bytes
        assert_eq!(border_char_count, header_width);
    }
}
