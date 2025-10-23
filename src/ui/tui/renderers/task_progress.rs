use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Rect},
    text::{Line, Span, Text},
    widgets::{Cell, Paragraph, Row, Table},
};

use super::table_styles::{self, AlwaysVisibleFilter};
use crate::ui::tui::{
    model::{
        AnalysisStatus, Model, PlaybackState,
        types::{FocusedTable, SignalRow},
    },
    themes::Theme,
};

impl table_styles::TableRow for SignalRow {
    fn build_cells(&self, theme: &dyn Theme) -> Vec<Cell<'static>> {
        let window_label = format!("{:03}", self.window_id + 1);
        let frequency = format_frequency_hz(self.frequency_hz);

        let status_text = match self.status {
            AnalysisStatus::Detected => theme.status_detected_text(),
            AnalysisStatus::Analyzing => theme.status_analyzing_text(),
            AnalysisStatus::Rejected => theme.status_rejected_text(),
            AnalysisStatus::Signal => theme.status_signal_text(),
            AnalysisStatus::Error => theme.status_error_text(),
        };

        let analysis_content = vec![Span::raw(status_text)];

        let audio_quality = if let Some(quality) = &self.audio_quality {
            use crate::audio::quality::AudioQuality;
            let quality_text = match quality {
                AudioQuality::Good => theme.quality_good_text(),
                AudioQuality::Moderate => theme.quality_moderate_text(),
                AudioQuality::Poor => theme.quality_poor_text(),
                AudioQuality::NoAudio => theme.quality_no_audio_text(),
                AudioQuality::Static => theme.quality_static_text(),
                AudioQuality::Unknown => theme.quality_unknown_text(),
            };

            let mut chars = quality_text.chars();
            let capitalized = match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            };

            Cell::from(capitalized)
        } else {
            Cell::from("")
        };

        let activity_cell = if self.playback_state == PlaybackState::Playing {
            Cell::from(theme.status_playing_text())
        } else {
            Cell::from("")
        };

        vec![
            Cell::from(window_label),
            Cell::from(Text::from(frequency).alignment(Alignment::Right)),
            Cell::from(ratatui::text::Line::from(analysis_content)),
            audio_quality,
            activity_cell,
        ]
    }
}

pub fn render_task_progress(
    f: &mut Frame,
    area: Rect,
    model: &mut Model,
    theme: &dyn Theme,
    task_id: &crate::ecs::TaskId,
) {
    let title = model
        .tasks
        .iter()
        .find(|t| &t.task_id == task_id)
        .map(|t| t.label.clone());

    if let Some(title) = title {
        // For now, all tasks are scans - in future, dispatch based on task type
        render_scan_table(f, area, model, theme, &title);
    } else {
        render_no_progress_message(f, area, theme, "Progress", "Task not found");
    }
}

fn render_scan_table(f: &mut Frame, area: Rect, model: &mut Model, theme: &dyn Theme, title: &str) {
    let has_focus = table_styles::check_focus(&model.focus_state, FocusedTable::ScanProgress);

    let block = table_styles::create_table_block(title, has_focus, theme);

    let header = Row::new(vec![
        Cell::from("Window"),
        Cell::from(Text::from("Frequency").alignment(Alignment::Right)),
        Cell::from("Analysis"),
        Cell::from("Audio"),
        Cell::from("Activity"),
    ])
    .style(table_styles::header_style());

    let signals = model.build_signal_rows();

    let viewport_height = area.height.saturating_sub(3) as usize;
    model.scan_progress_scroll.viewport_height = viewport_height;

    let visibility_context = table_styles::VisibilityContext::new(None, None);

    let (rows, scrollbar_state) = {
        let mut renderer = table_styles::TableRenderer2::new(
            &signals,
            FocusedTable::ScanProgress,
            AlwaysVisibleFilter,
            &mut model.scan_progress_scroll,
        );
        renderer.render(&model.focus_state, theme, visibility_context)
    };

    let table = Table::new(rows, [
        Constraint::Length(8),       // Window
        Constraint::Length(11),      // Frequency
        Constraint::Length(9),       // Analysis (exact fit for "Analyzing")
        Constraint::Length(8),       // Audio (exact fit for "Moderate", "No audio")
        Constraint::Percentage(100), // Activity (takes remaining width)
    ])
    .header(header)
    .block(block)
    .column_spacing(2);

    f.render_widget(table, area);
    table_styles::render_scrollbar(f, area, &scrollbar_state, theme);
}

pub fn render_no_progress_message(
    f: &mut Frame,
    area: Rect,
    theme: &dyn Theme,
    title: &str,
    message: &str,
) {
    let block = table_styles::create_table_block(title, false, theme);
    let message_line = Line::from(message);

    let paragraph = Paragraph::new(message_line)
        .alignment(Alignment::Center)
        .block(block);

    f.render_widget(paragraph, area);
}

fn format_frequency_hz(freq_hz: f64) -> String {
    let freq_int = freq_hz as u64;
    let freq_str = freq_int.to_string();
    let len = freq_str.len();

    let mut result = String::new();
    for (i, ch) in freq_str.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push('.');
        }
        result.push(ch);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::renderers::table_styles::RowGroup;

    #[test]
    fn test_signal_row_group_id() {
        let row = SignalRow {
            window_id: 5,
            frequency_hz: 88.9e6,
            status: AnalysisStatus::Signal,
            playback_state: PlaybackState::NotPlaying,
            audio_quality: None,
            is_window_complete: false,
            completion: 0.5,
        };

        assert_eq!(row.group_id(), 5);
    }
}
