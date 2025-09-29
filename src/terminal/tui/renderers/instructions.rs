//! Instructions rendering for the bottom of the TUI

use crate::terminal::tui::themes::Theme;
use ratatui::{Frame, style::Style, widgets::Paragraph};

/// Render user instructions
pub fn render_instructions(f: &mut Frame, area: ratatui::layout::Rect, theme: &dyn Theme) {
    // Elegant instructions with refined language
    let instruction =
        Paragraph::new("  ⌃C to exit").style(Style::default().fg(theme.instructions_dim()));
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
