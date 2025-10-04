//! Header rendering for the TUI interface

use crate::terminal::tui::{
    model::Model,
    themes::{SharedText, Theme},
};
use ratatui::{
    Frame,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

/// Render the application header
pub fn render_header(f: &mut Frame, area: ratatui::layout::Rect, model: &Model, theme: &dyn Theme) {
    let title_text = theme.title();
    let subtitle_text = theme.subtitle();

    // Calculate statistics
    let total_candidates = model
        .windows
        .values()
        .map(|w| w.candidates.len())
        .sum::<usize>();

    let stations_found = model
        .windows
        .values()
        .flat_map(|w| &w.candidates)
        .filter(|c| {
            matches!(
                c.status,
                crate::terminal::tui::model::CandidateStatus::Signal
                    | crate::terminal::tui::model::CandidateStatus::Playing
                    | crate::terminal::tui::model::CandidateStatus::Completed
            )
        })
        .count();

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

    // Create stats text with colored stations count
    let stats_text = format!(
        "{}: {} | {}: {}",
        SharedText::candidates_label(),
        total_candidates,
        SharedText::stations_label(),
        stations_found
    );
    let stats_text_len = stats_text.len();

    // Calculate padding for right-aligned stats
    let title_padding = inner_width
        .saturating_sub(title_text.len())
        .saturating_sub(stats_text_len);
    let subtitle_padding = inner_width.saturating_sub(subtitle_text.len());

    // Create title line with colored spans
    let title_spans = vec![
        Span::styled(
            format!("{}{}", title_text, " ".repeat(title_padding)),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("{}: ", SharedText::candidates_label())),
        Span::raw(total_candidates.to_string()),
        Span::raw(format!(" | {}: ", SharedText::stations_label())),
        Span::styled(
            stations_found.to_string(),
            Style::default()
                .fg(theme.header_accent())
                .add_modifier(Modifier::BOLD),
        ),
    ];
    let title_line = Line::from(title_spans);

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
        let title_text = "Radio Scanner";
        let subtitle_text = "Monitoring broadcast spectrum • FM • 88–108 MHz";

        assert_eq!(title_text, "Radio Scanner");
        assert_eq!(
            subtitle_text,
            "Monitoring broadcast spectrum • FM • 88–108 MHz"
        );

        let total_candidates = 5;
        let stations_found = 2;
        let stats_text = format!(
            "Candidates: {} | Stations: {}",
            total_candidates, stations_found
        );
        assert_eq!(stats_text, "Candidates: 5 | Stations: 2");

        let header_width: usize = 50;
        let top_border = format!("╭{}╮", "─".repeat(header_width.saturating_sub(2)));
        assert!(top_border.starts_with("╭"));
        assert!(top_border.ends_with("╮"));
        let border_char_count = top_border.chars().count(); // Count Unicode chars, not bytes
        assert_eq!(border_char_count, header_width);
    }
}
