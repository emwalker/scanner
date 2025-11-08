//! Layout calculations and constraints for TUI rendering

use ratatui::layout::{Constraint, Direction, Layout as RatatuiLayout, Rect};

/// Main TUI layout with Activities at full width and Tuners/Progress side-by-side
pub struct Layout {
    pub header: Rect,
    pub activities: Rect,
    pub tuners: Rect,
    pub signals_table: Rect,
    pub scan_progress: Rect,
    pub instructions: Rect,
}

impl Layout {
    /// Create a new layout with dynamic table heights
    ///
    /// Tables are sized to fit their content exactly:
    /// - Tuners table: height = (tuner_count + 3), capped at 50% of available height
    /// - Signals table: height = (confirmed_signals_count + 3), no cap
    /// - Scan progress table: height = (total_signals_count + 3), no cap
    /// - Remaining space in left column is left empty (not allocated to tables)
    pub fn new(
        area: Rect,
        tuner_count: usize,
        confirmed_signals_count: usize,
        total_signals_count: usize,
    ) -> Self {
        let vertical_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // Header
                Constraint::Min(0),    // Content (activities + tuners/progress)
                Constraint::Length(1), // Instructions
            ])
            .split(area);

        // Calculate dynamic heights for tables
        let tuners_height_uncapped = (tuner_count + 3).max(3) as u16;
        let signals_height = (confirmed_signals_count + 3).max(3) as u16;
        let scan_height = (total_signals_count + 3).max(3) as u16;

        let content_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Activities (header + border + 1 row)
                Constraint::Min(0),    /* Tuners and Progress (side-by-side with independent
                                        * heights) */
            ])
            .split(vertical_chunks[1]);

        // Calculate available height for bottom section
        let available_for_bottom = content_chunks[1].height;
        let left_column_height = available_for_bottom / 2; // 50% for left column
        let max_tuners_height = left_column_height / 2; // 50% of left column for tuners

        // Apply 50% cap to tuners height, but maintain minimum of 3
        let tuners_height = tuners_height_uncapped.min(max_tuners_height).max(3);

        let bottom_chunks = RatatuiLayout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Left column (tuners + signals table)
                Constraint::Percentage(50), // Scan progress column
            ])
            .split(content_chunks[1]);

        // Split left column into tuners and signals table with dynamic heights
        let left_column_chunks = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(tuners_height),  // Dynamic tuners height
                Constraint::Length(signals_height), // Dynamic signals height
                Constraint::Min(0),                 // Remaining space
            ])
            .split(bottom_chunks[0]);

        let scan_column = RatatuiLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(scan_height), Constraint::Min(0)])
            .split(bottom_chunks[1]);

        Self {
            header: vertical_chunks[0],
            activities: content_chunks[0],
            tuners: left_column_chunks[0],
            signals_table: left_column_chunks[1],
            scan_progress: scan_column[0],
            instructions: vertical_chunks[2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_tuners_height_calculation() {
        let area = Rect::new(0, 0, 100, 50);

        // Test with 0 tuners - should be minimum 3
        let layout = Layout::new(area, 0, 5, 5);
        assert_eq!(
            layout.tuners.height, 3,
            "Empty tuners table should have minimum height of 3"
        );

        // Test with 1 tuner - should be 1 + 3 = 4
        let layout = Layout::new(area, 1, 5, 5);
        assert_eq!(
            layout.tuners.height, 4,
            "1 tuner should result in height of 4 (1 + 3 for borders/header)"
        );

        // Test with 3 tuners - should be 3 + 3 = 6
        let layout = Layout::new(area, 3, 5, 5);
        assert_eq!(
            layout.tuners.height, 6,
            "3 tuners should result in height of 6"
        );

        // Test with 5 tuners - should be 5 + 3 = 8
        let layout = Layout::new(area, 5, 5, 5);
        assert_eq!(
            layout.tuners.height, 8,
            "5 tuners should result in height of 8"
        );
    }

    #[test]
    fn test_tuners_height_50_percent_maximum_cap() {
        // Create area where 50% would be less than needed
        let area = Rect::new(0, 0, 100, 20); // Total height 20
        // Available height for bottom section: 20 - 2 (header) - 4 (activities) - 1 (instructions)
        // = 13 Left column gets 50% of bottom = ~6-7 units
        // 50% of left column = ~3 units maximum for tuners

        // Test with many tuners that would exceed 50% cap
        let layout = Layout::new(area, 10, 5, 5); // 10 tuners would want 13 height, but should be capped
        let available_for_bottom = 20 - 2 - 4 - 1; // 13
        let left_column_height = available_for_bottom / 2; // ~6-7
        let max_tuners_height = left_column_height / 2; // ~3

        assert!(
            layout.tuners.height <= max_tuners_height,
            "Tuners height should be capped at 50% of available height"
        );
    }

    #[test]
    fn test_dynamic_signals_height_calculation() {
        let area = Rect::new(0, 0, 100, 50);

        // Test with 0 signals - should be minimum 3
        let layout = Layout::new(area, 2, 0, 0);
        assert_eq!(
            layout.signals_table.height, 3,
            "Empty signals table should have minimum height of 3"
        );

        // Test with 1 signal - should be 1 + 3 = 4
        let layout = Layout::new(area, 2, 1, 1);
        assert_eq!(
            layout.signals_table.height, 4,
            "1 signal should result in height of 4"
        );

        // Test with 5 signals - should be 5 + 3 = 8
        let layout = Layout::new(area, 2, 5, 5);
        assert_eq!(
            layout.signals_table.height, 8,
            "5 signals should result in height of 8"
        );

        // Test with 20 signals - should be 20 + 3 = 23
        let layout = Layout::new(area, 2, 20, 20);
        assert_eq!(
            layout.signals_table.height, 23,
            "20 signals should result in height of 23"
        );
    }

    #[test]
    fn test_dynamic_heights_leave_remaining_space() {
        let area = Rect::new(0, 0, 100, 50);

        // Test with small counts that don't fill available space
        let layout = Layout::new(area, 2, 3, 3); // Small counts

        let tuners_height = layout.tuners.height;
        let signals_height = layout.signals_table.height;
        let total_used = tuners_height + signals_height;

        // Calculate available space for left column
        let available_for_bottom = 50 - 2 - 4 - 1; // 43
        let left_column_height = available_for_bottom / 2; // ~21

        assert!(
            total_used < left_column_height,
            "Small table content should not fill entire left column, leaving remaining space"
        );
    }

    #[test]
    fn test_edge_case_very_small_area() {
        // Test with area too small for normal operation
        // Height 15 gives us: 15 - 2 (header) - 4 (activities) - 1 (instructions) = 8 for bottom
        // Left column gets 4, so we have some space to work with
        let area = Rect::new(0, 0, 100, 15);

        let layout = Layout::new(area, 3, 5, 5);

        // With severely constrained space, layout engine may not honor our exact requests,
        // but we should still get reasonable allocations
        assert!(
            layout.tuners.height > 0,
            "Tuners should get some height, got: {}",
            layout.tuners.height
        );
        assert!(
            layout.signals_table.height > 0,
            "Signals should get some height, got: {}",
            layout.signals_table.height
        );

        // The total allocated should not exceed available space
        let total = layout.tuners.height + layout.signals_table.height;
        assert!(total <= 8, "Total height should not exceed available space");
    }

    #[test]
    fn test_signals_table_height_matches_confirmed_signals_not_all_signals() {
        // This test reproduces the issue where Signals table has blank rows
        // because we count all displayable signals but only show confirmed ones
        let area = Rect::new(0, 0, 100, 50);

        // Scenario: 1 confirmed signal out of 10 total signals
        // Layout should now use confirmed signal count (1) for Signals table height
        // and total signal count (10) for Scan Progress table height
        let layout = Layout::new(area, 2, 1, 10); // 2 tuners, 1 confirmed signal, 10 total signals

        // After fix: signals table height should match confirmed signals count
        // 1 confirmed signal should result in height of 4 (1 + 3 for borders/header)
        assert_eq!(
            layout.signals_table.height, 4,
            "Signals table height should match confirmed signals count (1+3=4), got: {}",
            layout.signals_table.height
        );

        // Scan progress should still use total signals count
        assert_eq!(
            layout.scan_progress.height, 13,
            "Scan progress height should match total signals count (10+3=13), got: {}",
            layout.scan_progress.height
        );
    }
}
