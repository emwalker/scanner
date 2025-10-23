//! Layout calculations and constraints for TUI rendering

use ratatui::layout::{Constraint, Direction, Layout as RatatuiLayout, Rect};

/// Main TUI layout with Activities at full width and Tuners/Progress side-by-side
pub struct Layout {
    pub header: Rect,
    pub activities: Rect,
    pub tuners: Rect,
    pub scan_progress: Rect,
    pub instructions: Rect,
}

impl Layout {
    pub fn new(area: Rect, tuner_count: usize, signal_count: usize) -> Self {
        let vertical_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // Header
                Constraint::Min(0),    // Content (activities + tuners/progress)
                Constraint::Length(1), // Instructions
            ])
            .split(area);

        let tuners_height = (tuner_count + 3).max(3) as u16;
        let scan_height = (signal_count + 3).max(3) as u16;

        let content_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Activities (header + border + 1 row)
                Constraint::Min(0),    /* Tuners and Progress (side-by-side with independent
                                        * heights) */
            ])
            .split(vertical_chunks[1]);

        let bottom_chunks = RatatuiLayout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Tuners column
                Constraint::Percentage(50), // Scan progress column
            ])
            .split(content_chunks[1]);

        // Create independent vertical layouts for each column
        let tuners_column = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(tuners_height), Constraint::Min(0)])
            .split(bottom_chunks[0]);

        let scan_column = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(scan_height), Constraint::Min(0)])
            .split(bottom_chunks[1]);

        Self {
            header: vertical_chunks[0],
            activities: content_chunks[0],
            tuners: tuners_column[0],
            scan_progress: scan_column[0],
            instructions: vertical_chunks[2],
        }
    }
}
