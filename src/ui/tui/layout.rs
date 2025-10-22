//! Layout calculations and constraints for TUI rendering

use ratatui::layout::Layout as RatatuiLayout;
use ratatui::layout::{Constraint, Direction, Rect};

/// Main TUI layout with horizontal split for tuner list
pub struct Layout {
    pub header: Rect,
    pub spectrum: Rect,
    pub progress: Rect,
    pub tuners: Rect,
    pub instructions: Rect,
}

impl Layout {
    pub fn new(area: Rect) -> Self {
        let vertical_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Length(8), // Spectrum (full width)
                Constraint::Min(0),    // Content area (split horizontally for progress + tuners)
                Constraint::Length(1), // Instructions
            ])
            .split(area);

        let terminal_width = area.width as usize;
        let wide_threshold = 100;

        let content_chunks = if terminal_width >= wide_threshold {
            RatatuiLayout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50), // Left: progress
                    Constraint::Length(2),      // Margin
                    Constraint::Percentage(50), // Right: tuner list
                ])
                .split(vertical_chunks[2])
        } else {
            RatatuiLayout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(60), // Left: progress
                    Constraint::Percentage(40), // Right: tuner list
                ])
                .split(vertical_chunks[2])
        };

        let (progress_area, tuners_area) = if terminal_width >= wide_threshold {
            (content_chunks[0], content_chunks[2])
        } else {
            (content_chunks[0], content_chunks[1])
        };

        Self {
            header: vertical_chunks[0],
            spectrum: vertical_chunks[1],
            progress: progress_area,
            tuners: tuners_area,
            instructions: vertical_chunks[3],
        }
    }
}
