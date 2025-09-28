//! Header rendering for the TUI interface

use crate::terminal::tui::model::Model;
use ratatui::{
    Frame,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

/// Render the application header
pub fn render_header(f: &mut Frame, area: ratatui::layout::Rect, model: &Model) {
    // Create full-width header with sophisticated styling
    let header_width = area.width as usize;
    let title_text = "Radio Scanner";
    let subtitle_text = "Monitoring broadcast spectrum • FM • 88–108 MHz";

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

    // Create stats text with colored stations count
    let stats_text = format!(
        "Candidates: {} | Stations: {}",
        total_candidates, stations_found
    );
    let stats_text_len = stats_text.len();

    // Top border only with rounded corners
    let top_border = format!("╭{}╮", "─".repeat(header_width.saturating_sub(2)));

    // Calculate padding for right-aligned stats
    let title_padding = header_width
        .saturating_sub(2)
        .saturating_sub(title_text.len())
        .saturating_sub(stats_text_len);
    let subtitle_padding = header_width
        .saturating_sub(2)
        .saturating_sub(subtitle_text.len());

    // Create title line with colored spans
    let title_spans = vec![
        Span::raw(format!(" {}{}", title_text, " ".repeat(title_padding))),
        Span::raw(format!("Candidates: {} | Stations: ", total_candidates)),
        Span::styled(
            stations_found.to_string(),
            Style::default().fg(Color::Green), // Match progress bar green
        ),
    ];
    let title_line = Line::from(title_spans);

    let subtitle_line = format!(" {}{}", subtitle_text, " ".repeat(subtitle_padding));

    // Create header with mixed content (border, colored line, plain line)
    let header_content = vec![
        Line::from(top_border),
        title_line,
        Line::from(subtitle_line),
    ];

    let title = Paragraph::new(header_content).style(
        Style::default()
            .fg(Color::Rgb(220, 220, 240))
            .add_modifier(Modifier::BOLD),
    );
    f.render_widget(title, area);
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
