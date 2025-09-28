//! Layout calculations and constraints for TUI rendering

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Minimum number of terminal lines to reserve for title, instructions, etc.
const RESERVED_TERMINAL_LINES: usize = 5;

/// Main layout structure for the TUI
pub struct TuiLayout {
    pub header: Rect,
    pub header_spacing: Rect,
    pub spectrum: Rect,
    pub separator: Rect,
    pub progress: Rect,
    pub instructions: Rect,
}

impl TuiLayout {
    /// Create the main layout for the TUI
    pub fn new(area: Rect) -> Self {
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title (top border only)
                Constraint::Length(1), // Spacing after header
                Constraint::Length(4), // Spectrum visualization (increased height for 3 lines)
                Constraint::Length(1), // Separator line
                Constraint::Min(0),    // Progress area
                Constraint::Length(1), // Instructions
            ])
            .split(area);

        Self {
            header: main_chunks[0],
            header_spacing: main_chunks[1],
            spectrum: main_chunks[2],
            separator: main_chunks[3],
            progress: main_chunks[4],
            instructions: main_chunks[5],
        }
    }

    /// Calculate constraints for progress bars within the progress area
    pub fn progress_constraints(&self, total_lines: usize) -> Vec<Constraint> {
        (0..total_lines).map(|_| Constraint::Length(1)).collect()
    }

    /// Calculate maximum number of progress bars that can fit
    pub fn max_progress_bars(&self) -> usize {
        let available_height = self.progress.height as usize;
        available_height.saturating_sub(RESERVED_TERMINAL_LINES)
    }
}

/// Layout for spectrum visualization
pub struct SpectrumLayout {
    pub frequencies: Rect,
    pub spectrum_bar: Rect,
    pub markers: Rect,
    pub extra: Rect,
}

impl SpectrumLayout {
    /// Create spectrum layout from the given area
    pub fn new(area: Rect) -> Self {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Top frequencies
                Constraint::Length(1), // Spectrum bar
                Constraint::Length(1), // Frequency markers below
                Constraint::Min(0),    // Extra space if available
            ])
            .split(area);

        Self {
            frequencies: layout[0],
            spectrum_bar: layout[1],
            markers: layout[2],
            extra: layout[3],
        }
    }
}
