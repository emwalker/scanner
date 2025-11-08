use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

const MAX_NOTES_LENGTH: usize = 100;

#[derive(Debug, Clone)]
pub struct NotesInput {
    pub input: String,
    pub cursor_position: usize,
    pub is_active: bool,
}

impl Default for NotesInput {
    fn default() -> Self {
        Self::new()
    }
}

impl NotesInput {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor_position: 0,
            is_active: false,
        }
    }

    pub fn with_content(content: &str) -> Self {
        let content = content.to_string();
        let cursor_position = content.len();
        Self {
            input: content,
            cursor_position,
            is_active: false,
        }
    }

    pub fn activate(&mut self) {
        self.is_active = true;
    }

    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn content(&self) -> &str {
        &self.input
    }

    pub fn set_content(&mut self, content: String) {
        self.input = content.chars().take(MAX_NOTES_LENGTH).collect();
        self.cursor_position = self.cursor_position.min(self.input.len());
    }

    pub fn clear(&mut self) {
        self.input.clear();
        self.cursor_position = 0;
    }

    /// Handle character input
    pub fn handle_char(&mut self, c: char) {
        if self.input.len() < MAX_NOTES_LENGTH {
            self.input.insert(self.cursor_position, c);
            self.move_cursor_right();
        }
    }

    /// Handle backspace
    pub fn handle_backspace(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.input.remove(self.cursor_position);
        }
    }

    /// Handle delete key
    pub fn handle_delete(&mut self) {
        if self.cursor_position < self.input.len() {
            self.input.remove(self.cursor_position);
        }
    }

    /// Move cursor left
    pub fn move_cursor_left(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
        }
    }

    /// Move cursor right
    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < self.input.len() {
            self.cursor_position += 1;
        }
    }

    /// Move cursor to start of input
    pub fn move_cursor_home(&mut self) {
        self.cursor_position = 0;
    }

    /// Move cursor to end of input
    pub fn move_cursor_end(&mut self) {
        self.cursor_position = self.input.len();
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        // Clear the background
        f.render_widget(Clear, area);

        // Create border style based on whether widget is active
        let border_style = if self.is_active {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };

        // Create the block with title showing character count
        let char_count = self.input.len();
        let title = format!(
            "Edit Notes ({}/{}) - ESC to cancel, Enter to save",
            char_count, MAX_NOTES_LENGTH
        );

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border_style);

        // Create text with cursor
        let text = if self.is_active && !self.input.is_empty() {
            // Split text at cursor position to insert cursor
            let before_cursor = &self.input[..self.cursor_position];
            let cursor_char = self.input.chars().nth(self.cursor_position).unwrap_or(' ');
            let after_cursor = &self.input[self.cursor_position + cursor_char.len_utf8()..];

            let mut spans = Vec::new();
            if !before_cursor.is_empty() {
                spans.push(Span::raw(before_cursor));
            }

            // Cursor character with highlight
            spans.push(Span::styled(
                cursor_char.to_string(),
                Style::default().bg(Color::Yellow).fg(Color::Black),
            ));

            if !after_cursor.is_empty() {
                spans.push(Span::raw(after_cursor));
            }

            vec![Line::from(spans)]
        } else if self.is_active {
            // Empty input, show just cursor
            vec![Line::from(vec![Span::styled(
                " ",
                Style::default().bg(Color::Yellow),
            )])]
        } else {
            // Not active, show text normally
            vec![Line::from(Span::raw(&self.input))]
        };

        let paragraph = Paragraph::new(text)
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true });

        f.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notes_input_new() {
        let input = NotesInput::new();
        assert_eq!(input.content(), "");
        assert_eq!(input.cursor_position, 0);
        assert!(!input.is_active());
    }

    #[test]
    fn test_notes_input_with_content() {
        let input = NotesInput::with_content("Hello world");
        assert_eq!(input.content(), "Hello world");
        assert_eq!(input.cursor_position, 11);
        assert!(!input.is_active());
    }

    #[test]
    fn test_handle_char() {
        let mut input = NotesInput::new();
        input.handle_char('H');
        input.handle_char('i');
        assert_eq!(input.content(), "Hi");
        assert_eq!(input.cursor_position, 2);
    }

    #[test]
    fn test_handle_backspace() {
        let mut input = NotesInput::with_content("Hello");
        input.handle_backspace();
        assert_eq!(input.content(), "Hell");
        assert_eq!(input.cursor_position, 4);
    }

    #[test]
    fn test_cursor_movement() {
        let mut input = NotesInput::with_content("Hello");

        input.move_cursor_left();
        assert_eq!(input.cursor_position, 4);

        input.move_cursor_home();
        assert_eq!(input.cursor_position, 0);

        input.move_cursor_end();
        assert_eq!(input.cursor_position, 5);
    }

    #[test]
    fn test_max_length_limit() {
        let mut input = NotesInput::new();
        let long_text = "a".repeat(MAX_NOTES_LENGTH + 10);

        for c in long_text.chars() {
            input.handle_char(c);
        }

        assert_eq!(input.content().len(), MAX_NOTES_LENGTH);
    }

    #[test]
    fn test_activate_deactivate() {
        let mut input = NotesInput::new();
        assert!(!input.is_active());

        input.activate();
        assert!(input.is_active());

        input.deactivate();
        assert!(!input.is_active());
    }
}
