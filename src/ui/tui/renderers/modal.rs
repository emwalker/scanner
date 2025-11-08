//! Modal rendering for signal detail views

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Clear, Paragraph},
};

use crate::ui::tui::{model::state::Model, themes::Theme};

/// Get modal foreground color from theme - background left transparent
fn modal_foreground(theme: &dyn Theme) -> Color {
    // Use theme foreground, let background be transparent/natural
    theme.foreground()
}

/// Render signal detail modal on top of existing UI
pub fn render_signal_detail_modal(f: &mut Frame, model: &Model, theme: &dyn Theme) {
    if let Some(modal) = &model.signal_detail_modal {
        // Calculate modal area (centered, 60% width, 40% height)
        let area = centered_rect(60, 40, f.area());

        // Clear the background where modal will appear
        f.render_widget(Clear, area);

        // Render modal background with border using theme colors
        let modal_fg = modal_foreground(theme);
        let block = Block::default()
            .title("Signal Details")
            .borders(Borders::ALL)
            .style(
                Style::default()
                    .fg(theme.primary())
                    .add_modifier(Modifier::BOLD),
            );
        f.render_widget(block, area);

        // Create inner content area with padding
        let inner_area = area.inner(Margin {
            horizontal: 2,
            vertical: 1,
        });

        // Find the signal to display its information
        // Use frequency-based lookup instead of SignalId to support both scan and persistent
        // signals
        let scan_signal = model
            .windows
            .values()
            .flat_map(|window| &window.signals)
            .find(|signal| (signal.frequency_hz - modal.frequency_hz).abs() < 1000.0); // 1kHz tolerance

        let persistent_signal = model
            .persistent_signals
            .iter()
            .find(|signal| (signal.frequency_hz - modal.frequency_hz).abs() < 1000.0); // 1kHz tolerance

        // Prepare modal content based on which type of signal was found
        let content = if let Some(signal) = scan_signal {
            // Scan signal found - use SignalProgress fields
            format!(
                "Signal ID: {}\nFrequency: {:.1} MHz\nStatus: {:?}\nAudio Quality: {:?}\nSignal \
                 Strength: {:?}\n\nNotes:\n{}",
                signal.signal_id,
                signal.frequency_hz / 1_000_000.0,
                signal.status,
                signal
                    .audio_quality
                    .as_ref()
                    .map(|q| format!("{:?}", q))
                    .unwrap_or_else(|| "Unknown".to_string()),
                signal
                    .signal_strength
                    .as_ref()
                    .map(|s| format!("{:.2}", s))
                    .unwrap_or_else(|| "Unknown".to_string()),
                modal.notes_input
            )
        } else if let Some(signal) = persistent_signal {
            // Persistent signal found - use PersistedSignal fields
            format!(
                "Signal ID: {}\nFrequency: {:.1} MHz\nModulation: {}\nDetection Count: {}\nSignal \
                 Strength: {:.2}\nFirst Detected: {}\nLast Detected: {}\n\nNotes:\n{}",
                modal.signal_id,
                signal.frequency_hz / 1_000_000.0,
                signal.modulation,
                signal.detection_count,
                signal.signal_strength,
                signal.first_detected.format("%Y-%m-%d %H:%M"),
                signal.last_detected.format("%Y-%m-%d %H:%M"),
                modal.notes_input
            )
        } else {
            format!("Signal not found: {}", modal.signal_id)
        };

        // Render content using natural background
        let paragraph = Paragraph::new(content)
            .style(Style::default().fg(modal_fg))
            .alignment(Alignment::Left);
        f.render_widget(paragraph, inner_area);

        // Add instructions at the bottom
        let instruction_area = Rect {
            x: inner_area.x,
            y: inner_area.y + inner_area.height.saturating_sub(2),
            width: inner_area.width,
            height: 2,
        };

        let instructions = Paragraph::new("Press ESC to close • ENTER to save notes")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center);
        f.render_widget(instructions, instruction_area);
    }
}

/// Helper function to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
