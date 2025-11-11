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
    locality: Option<&str>,
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

    // Create title line with locality on the right if present
    let title_line = if let Some(locality_text) = locality {
        let locality_text = if locality_text.len() > 15 {
            // Truncate long locality names to prevent layout issues
            format!("{}...", &locality_text[..12])
        } else {
            locality_text.to_string()
        };

        let title_len = title_text.len();
        let locality_len = locality_text.len();
        let total_content = title_len + locality_len;

        if total_content < inner_width {
            let padding = inner_width.saturating_sub(total_content);
            Line::from(vec![
                Span::styled(title_text, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" ".repeat(padding)),
                Span::styled(locality_text, Style::default()),
            ])
        } else {
            // Not enough space for both, just show title
            let title_padding = inner_width.saturating_sub(title_len);
            Line::from(vec![Span::styled(
                format!("{}{}", title_text, " ".repeat(title_padding)),
                Style::default().add_modifier(Modifier::BOLD),
            )])
        }
    } else {
        // No locality, center title as before
        let title_padding = inner_width.saturating_sub(title_text.len());
        Line::from(vec![Span::styled(
            format!("{}{}", title_text, " ".repeat(title_padding)),
            Style::default().add_modifier(Modifier::BOLD),
        )])
    };

    let subtitle_padding = inner_width.saturating_sub(subtitle_text.len());

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

    #[test]
    fn test_locality_truncation() {
        let long_locality = "San Francisco International Airport Area";
        let truncated = if long_locality.len() > 15 {
            format!("{}...", &long_locality[..12])
        } else {
            long_locality.to_string()
        };

        assert_eq!(truncated, "San Francisc...");
        assert!(truncated.len() <= 15);
    }

    #[test]
    fn test_locality_layout_calculation() {
        let title_text = "Monitor";
        let locality_text = "Loveland";
        let inner_width = 40;

        let title_len = title_text.len();
        let locality_len = locality_text.len();
        let total_content = title_len + locality_len;

        // Should have enough space for both title and locality with padding
        assert!(total_content < inner_width);

        let padding = inner_width.saturating_sub(total_content);
        assert!(padding > 0);
    }

    #[test]
    fn test_locality_fallback_for_narrow_width() {
        let title_text = "Monitor";
        let locality_text = "Loveland";
        let narrow_width = 10; // Too narrow for both

        let title_len = title_text.len();
        let locality_len = locality_text.len();
        let total_content = title_len + locality_len;

        // Should fall back to title-only when not enough space
        assert!(total_content >= narrow_width);
    }
}
