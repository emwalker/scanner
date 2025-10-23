use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    widgets::{Cell, Row, Table},
};

use super::table_styles::{self, AlwaysVisibleFilter, TableRenderer2, VisibilityContext};
use crate::ui::tui::{
    model::{state::Model, types::FocusedTable},
    themes::Theme,
};

pub fn render_activities(f: &mut Frame, area: Rect, model: &mut Model, theme: &dyn Theme) {
    let has_focus = table_styles::check_focus(&model.focus_state, FocusedTable::Activities);
    let block = table_styles::create_table_block("Activities", has_focus, theme);

    let header = Row::new(vec![
        Cell::from("Task"),
        Cell::from(""),
        Cell::from("Position"),
        Cell::from("Tuner"),
        Cell::from("Status"),
    ])
    .style(table_styles::header_style());

    let viewport_height = area.height.saturating_sub(3) as usize;
    model.activities_scroll.viewport_height = viewport_height;

    let visibility_context = VisibilityContext::new(None, None);

    // Build a lookup map for tuner labels
    let tuner_labels: std::collections::HashMap<&crate::hardware::pool::TunerId, &str> = model
        .tuners
        .iter()
        .map(|tuner_info| (&tuner_info.id, tuner_info.label.as_str()))
        .collect();

    // Update assigned_tuner labels using the tuner info from model
    for task in &mut model.tasks {
        if let Some(tuner_id) = &task.assigned_tuner_id
            && let Some(label) = tuner_labels.get(tuner_id)
        {
            task.assigned_tuner = Some(label.to_string());
        }
    }

    let (rows, scrollbar_state) = {
        let mut renderer = TableRenderer2::new(
            &model.tasks,
            FocusedTable::Activities,
            AlwaysVisibleFilter,
            &mut model.activities_scroll,
        );

        renderer.render(&model.focus_state, theme, visibility_context)
    };

    let table = Table::new(rows, [
        Constraint::Length(8),  // Task
        Constraint::Length(20), // Range (no header label)
        Constraint::Length(24), // Position (spectrum bar)
        Constraint::Min(20),    // Tuner (expands to fill available space)
        Constraint::Length(22), // Status
    ])
    .header(header)
    .block(block)
    .column_spacing(2);

    f.render_widget(table, area);
    table_styles::render_scrollbar(f, area, &scrollbar_state, theme);
}

pub fn render_window_cell_content(
    full_range_hz: &(f64, f64),
    current_window_hz: &Option<(f64, f64)>,
    width: usize,
) -> String {
    render_spectrum_bar(full_range_hz, current_window_hz, width)
}

fn render_spectrum_bar(
    full_range: &(f64, f64),
    window_range: &Option<(f64, f64)>,
    width: usize,
) -> String {
    let (full_start, full_end) = full_range;
    let full_span = full_end - full_start;

    let mut bar = vec!['░'; width];

    if let Some((win_start, win_end)) = window_range {
        // Clamp window to the visible range
        let clamped_start = win_start.max(*full_start);
        let clamped_end = win_end.min(*full_end);

        if clamped_end > clamped_start {
            // Calculate the center position of the window
            let center = (clamped_start + clamped_end) / 2.0;
            let center_pos = (center - full_start) / full_span * width as f64;

            // Calculate a fixed width for the window based on its actual size
            let window_span = clamped_end - clamped_start;
            let window_width_chars = (window_span / full_span * width as f64).max(1.0);

            // Round the width to ensure consistency
            let fixed_width = window_width_chars.round() as usize;

            // Center the fixed-width bar around the center position
            let start_pos = ((center_pos - fixed_width as f64 / 2.0).round() as usize)
                .min(width.saturating_sub(fixed_width));
            let end_pos = (start_pos + fixed_width).min(width);

            tracing::info!(
                "Spectrum bar: full_range={:.2e}-{:.2e}, window={:.2e}-{:.2e}, \
                 clamped={:.2e}-{:.2e}, width={}, start_pos={}, end_pos={}, fixed_width={}",
                full_start,
                full_end,
                win_start,
                win_end,
                clamped_start,
                clamped_end,
                width,
                start_pos,
                end_pos,
                fixed_width
            );

            for item in bar.iter_mut().take(end_pos).skip(start_pos) {
                *item = '█';
            }
        }
    } else {
        tracing::info!("Spectrum bar: No window range provided");
    }

    bar.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_bar_width_stability_across_sliding_window() {
        // Simulate the actual FM scan parameters
        let full_range = (88.0e6, 108.0e6); // 88-108 MHz
        let step_size = 0.5e6; // 0.5 MHz window
        let width = 24; // UI constraint from line 61

        // Simulate 40 sequential window positions as the scan progresses
        let mut widths = Vec::new();
        let num_windows = 40;

        for i in 0..num_windows {
            let center = full_range.0 + (i as f64 * step_size);
            let window = (center - step_size / 2.0, center + step_size / 2.0);

            let bar = render_spectrum_bar(&full_range, &Some(window), width);
            let rendered_width = bar.chars().filter(|&c| c == '█').count();
            widths.push(rendered_width);
        }

        // The window size is constant (0.5 MHz), so the rendered width should be constant
        // Currently this fails because widths alternate between 1 and 2 characters
        let first_width = widths[0];
        for (i, &w) in widths.iter().enumerate() {
            assert_eq!(
                w, first_width,
                "Window {} has width {} but expected {} (widths: {:?})",
                i, w, first_width, widths
            );
        }
    }

    #[test]
    fn test_spectrum_bar_empty_window() {
        let full_range = (88.0e6, 108.0e6);
        let width = 24;

        let bar = render_spectrum_bar(&full_range, &None, width);

        assert_eq!(bar.chars().filter(|&c| c == '░').count(), width);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 0);
    }

    #[test]
    fn test_spectrum_bar_full_range_window() {
        let full_range = (88.0e6, 108.0e6);
        let width = 24;

        let bar = render_spectrum_bar(&full_range, &Some(full_range), width);

        assert_eq!(bar.chars().filter(|&c| c == '█').count(), width);
    }

    #[test]
    fn test_spectrum_bar_window_outside_range() {
        let full_range = (88.0e6, 108.0e6);
        let width = 24;

        // Window completely outside range
        let window = (110.0e6, 111.0e6);
        let bar = render_spectrum_bar(&full_range, &Some(window), width);

        // Should render as empty (all background)
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 0);
    }

    #[test]
    fn test_spectrum_bar_window_partial_overlap() {
        let full_range = (88.0e6, 108.0e6);
        let width = 24;

        // Window overlaps start of range
        let window = (87.0e6, 89.0e6);
        let bar = render_spectrum_bar(&full_range, &Some(window), width);

        // Should have some filled characters (clamped to valid range)
        let filled = bar.chars().filter(|&c| c == '█').count();
        assert!(
            filled > 0,
            "Expected some filled characters for partial overlap"
        );
    }
}
