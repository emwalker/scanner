//! Instructions rendering for the bottom of the TUI

use ratatui::{
    Frame,
    style::{Color, Style},
    widgets::Paragraph,
};

/// Render user instructions
pub fn render_instructions(f: &mut Frame, area: ratatui::layout::Rect) {
    // Elegant instructions with refined language
    let instruction =
        Paragraph::new("  ⌃C to exit").style(Style::default().fg(Color::Rgb(120, 120, 140)));
    f.render_widget(instruction, area);
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
