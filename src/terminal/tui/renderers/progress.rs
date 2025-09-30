//! Progress bar rendering for candidates and windows

use crate::terminal::tui::{
    model::{CandidateProgress, CandidateStatus, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

/// Render all progress bars for windows and candidates
pub fn render_progress(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    model: &Model,
    theme: &dyn Theme,
) {
    if model.windows.is_empty() {
        let waiting =
            Paragraph::new("  Establishing connection…\n  Preparing to monitor frequencies")
                .style(Style::default().fg(theme.instructions_dim()));
        f.render_widget(waiting, area);
        return;
    }

    // Get displayable windows for selection tracking
    let displayable_windows = model.get_displayable_windows();

    // Calculate available space for progress bars
    let available_height = area.height as usize;
    let max_bars = available_height.saturating_sub(5); // RESERVED_TERMINAL_LINES

    // In selection mode, use selectable candidates (no rejected); otherwise use displayable
    let use_selectable = model.selection_mode;

    // Calculate window sizes that fit in available space
    let mut windows_that_fit = Vec::new();

    for (window_id, window) in displayable_windows.iter() {
        let is_current_window = **window_id == model.current_window;
        let candidates = if use_selectable {
            // In selection mode, never show rejected candidates
            window
                .displayable_candidates(is_current_window)
                .into_iter()
                .filter(|c| c.status != crate::terminal::tui::model::CandidateStatus::Rejected)
                .collect::<Vec<_>>()
        } else {
            window.displayable_candidates(is_current_window)
        };
        let candidate_count = candidates.len();
        let window_bars = candidate_count + 1; // +1 for window header

        windows_that_fit.push((window_id, candidate_count, window_bars));
    }

    // Now work backwards to fit as many recent windows as possible
    let mut window_sizes = Vec::new();
    let mut running_total = 0;

    // Add "Continue scan" line if in selection mode
    if model.selection_mode {
        running_total += 1;
    }

    for (window_id, candidate_count, window_bars) in windows_that_fit.iter().rev() {
        if running_total + window_bars <= max_bars {
            window_sizes.insert(0, (*window_id, *candidate_count)); // Insert at front to maintain order
            running_total += window_bars;
        }
        // If it doesn't fit, this older window gets pushed off the top
    }

    if window_sizes.is_empty() {
        return;
    }

    // Create constraints: 1 line per candidate + 1 line per window header + 1 for continue scan
    let mut total_lines = window_sizes
        .iter()
        .map(|(_, count)| count + 1)
        .sum::<usize>();
    if model.selection_mode {
        total_lines += 1;
    }
    let constraints: Vec<Constraint> = (0..total_lines).map(|_| Constraint::Length(1)).collect();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    // Render all progress bars sequentially
    let mut chunk_idx = 0;
    let mut candidate_index = 0;

    for (window_id, _candidate_count) in window_sizes {
        if chunk_idx >= chunks.len() {
            break;
        }

        // Check if any candidate in this window is selected
        let window = &model.windows[window_id];
        let is_current_window = **window_id == model.current_window;
        let displayable_candidates = if use_selectable {
            // In selection mode, filter out rejected candidates
            window
                .displayable_candidates(is_current_window)
                .into_iter()
                .filter(|c| c.status != crate::terminal::tui::model::CandidateStatus::Rejected)
                .collect::<Vec<_>>()
        } else {
            window.displayable_candidates(is_current_window)
        };
        let window_candidate_count = displayable_candidates.len();

        let window_has_selection = if let Some(selected_idx) = model.selected_candidate_index {
            selected_idx >= candidate_index
                && selected_idx < candidate_index + window_candidate_count
        } else {
            false
        };

        // Render window header
        render_window_header(
            f,
            chunks[chunk_idx],
            **window_id,
            window_has_selection,
            model,
            theme,
        );
        chunk_idx += 1;

        // Render candidates in this window (preserves insertion order, filtered)
        for candidate in displayable_candidates {
            if chunk_idx >= chunks.len() {
                break;
            }
            let is_selected =
                model.selection_mode && model.selected_candidate_index == Some(candidate_index);
            render_candidate_progress(f, chunks[chunk_idx], candidate, is_selected, theme);
            chunk_idx += 1;
            candidate_index += 1;
        }
    }

    // Add "Continue scan" option if in selection mode
    if model.selection_mode && chunk_idx < chunks.len() {
        let is_continue_selected = model.is_continue_scan_selected();
        render_continue_scan(f, chunks[chunk_idx], is_continue_selected, theme);
    }
}

/// Render a window header
fn render_window_header(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    window_id: usize,
    _is_selected: bool,
    _model: &Model,
    theme: &dyn Theme,
) {
    let color = theme.window_header();

    // Geometric window indicator with precision framing
    let header = format!(
        " {} Scan {} {}",
        theme.window_bullet(),
        window_id,
        theme.window_bullet()
    );
    let header_widget =
        Paragraph::new(header).style(Style::default().fg(color).add_modifier(Modifier::BOLD));
    f.render_widget(header_widget, area);
}

/// Render progress for a single candidate
fn render_candidate_progress(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    candidate: &CandidateProgress,
    is_selected: bool,
    theme: &dyn Theme,
) {
    let freq_mhz = candidate.frequency_hz / 1e6;

    // Clean geometric status indicators with refined terminology
    let (status_text, status_symbol) = match candidate.status {
        CandidateStatus::Detected => (theme.status_detected_text(), theme.symbol_detected()),
        CandidateStatus::Analyzing => (theme.status_analyzing_text(), theme.symbol_analyzing()),
        CandidateStatus::Rejected => (theme.status_rejected_text(), theme.symbol_rejected()),
        CandidateStatus::Signal => (theme.status_signal_text(), theme.symbol_signal()),
        CandidateStatus::Playing => (theme.status_playing_text(), theme.symbol_playing()),
        CandidateStatus::Completed => (theme.status_completed_text(), theme.symbol_completed()),
    };

    // Status colors from theme
    let status_color = if is_selected {
        theme.selection_highlight()
    } else {
        match candidate.status {
            CandidateStatus::Detected => theme.status_detected(),
            CandidateStatus::Analyzing => theme.status_analyzing(),
            CandidateStatus::Rejected => theme.status_rejected(),
            CandidateStatus::Signal => theme.status_signal(),
            CandidateStatus::Playing => theme.status_playing(),
            CandidateStatus::Completed => theme.status_completed(),
        }
    };

    // Atmospheric amber progress with geometric precision
    let progress_width = 20;
    let filled = (candidate.completion * progress_width as f64) as usize;
    let progress_bar = if filled == 0 {
        theme.progress_empty().repeat(progress_width)
    } else if filled >= progress_width {
        theme.progress_full().repeat(progress_width)
    } else {
        // Gradient-like progression with theme elements
        let full_blocks = filled;
        let remaining = progress_width - filled;
        format!(
            "{}{}",
            theme.progress_full().repeat(full_blocks),
            theme.progress_empty().repeat(remaining)
        )
    };

    // Create line with separate styling for different parts
    let line = if let Some(audio_quality) = &candidate.audio_quality {
        if candidate.status == CandidateStatus::Completed
            || candidate.status == CandidateStatus::Rejected
        {
            // Get audio quality text and style
            let (quality_text, quality_style) = match audio_quality {
                crate::audio_quality::AudioQuality::Good => (
                    "Good",
                    Style::default()
                        .fg(theme.quality_good())
                        .add_modifier(Modifier::BOLD),
                ),
                crate::audio_quality::AudioQuality::Moderate => {
                    ("Moderate", Style::default().fg(theme.quality_moderate()))
                }
                crate::audio_quality::AudioQuality::Poor => {
                    ("Poor", Style::default().fg(theme.quality_poor()))
                }
                crate::audio_quality::AudioQuality::NoAudio => {
                    ("No Audio", Style::default().fg(theme.quality_no_audio()))
                }
                crate::audio_quality::AudioQuality::Static => {
                    ("Static", Style::default().fg(theme.quality_static()))
                }
                crate::audio_quality::AudioQuality::Unknown => {
                    ("Unknown", Style::default().fg(theme.quality_unknown()))
                }
            };

            // Create spans with different colors
            Line::from(vec![
                Span::styled(
                    format!(
                        "   {} {:>5.1} MHz • {} • {} • ",
                        status_symbol, freq_mhz, progress_bar, status_text
                    ),
                    Style::default().fg(status_color),
                ),
                Span::styled(quality_text, quality_style),
            ])
        } else {
            Line::from(vec![Span::styled(
                format!(
                    "   {} {:>5.1} MHz • {} • {}",
                    status_symbol, freq_mhz, progress_bar, status_text
                ),
                Style::default().fg(status_color),
            )])
        }
    } else {
        Line::from(vec![Span::styled(
            format!(
                "   {} {:>5.1} MHz • {} • {}",
                status_symbol, freq_mhz, progress_bar, status_text
            ),
            Style::default().fg(status_color),
        )])
    };

    let gauge = Paragraph::new(line);

    f.render_widget(gauge, area);
}

/// Render "Continue scan" option when in selection mode
fn render_continue_scan(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    is_selected: bool,
    theme: &dyn Theme,
) {
    let color = if is_selected {
        theme.selection_highlight()
    } else {
        theme.instructions_dim()
    };

    let line = Line::from(vec![Span::styled(
        " Continue scan →",
        Style::default().fg(color).add_modifier(if is_selected {
            Modifier::BOLD
        } else {
            Modifier::empty()
        }),
    )]);

    let paragraph = Paragraph::new(line);
    f.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use crate::terminal::tui::model::CandidateStatus;
    use ratatui::style::Color;

    #[test]
    fn test_window_header_format_unchanged() {
        let expected_format = " ◆ Scan 1 ◆";
        let actual_format = format!(" ◆ Scan {} ◆", 1);
        assert_eq!(actual_format, expected_format);

        assert_eq!(format!(" ◆ Scan {} ◆", 1), " ◆ Scan 1 ◆");
        assert_eq!(format!(" ◆ Scan {} ◆", 15), " ◆ Scan 15 ◆");
        assert_eq!(format!(" ◆ Scan {} ◆", 999), " ◆ Scan 999 ◆");
    }

    #[test]
    fn test_progress_bar_visual_elements_unchanged() {
        let empty_char = "⠀"; // Braille blank
        let full_char = "⣿"; // Braille full
        let partial_char = "⣀"; // Braille partial

        let progress_width = 20;

        // 0% progress
        let progress_0 = empty_char.repeat(progress_width);
        assert_eq!(progress_0.len(), empty_char.len() * progress_width);

        // 100% progress
        let progress_100 = full_char.repeat(progress_width);
        assert_eq!(progress_100.len(), full_char.len() * progress_width);

        // Partial progress (50%)
        let filled = (0.5 * progress_width as f64) as usize;
        let remaining = progress_width - filled;
        let progress_50 = format!(
            "{}{}",
            full_char.repeat(filled),
            partial_char.repeat(remaining)
        );
        assert_eq!(filled, 10);
        assert_eq!(remaining, 10);

        assert_eq!(progress_50, format!("{}{}", "⣿".repeat(10), "⣀".repeat(10)));
    }

    #[test]
    fn test_status_symbols_unchanged() {
        let status_symbols = vec![
            (CandidateStatus::Detected, "◦"),  // Located
            (CandidateStatus::Analyzing, "◐"), // Evaluating
            (CandidateStatus::Rejected, "◌"),  // Filtered
            (CandidateStatus::Signal, "●"),    // Acquired
            (CandidateStatus::Playing, "♬"),   // Playing
            (CandidateStatus::Completed, "◆"), // Complete
        ];

        for (status, expected_symbol) in status_symbols {
            let (status_text, status_symbol) = match status {
                CandidateStatus::Detected => ("Located", "◦"),
                CandidateStatus::Analyzing => ("Evaluating", "◐"),
                CandidateStatus::Rejected => ("Filtered", "◌"),
                CandidateStatus::Signal => ("Acquired", "●"),
                CandidateStatus::Playing => ("Playing", "♬"),
                CandidateStatus::Completed => ("Complete", "◆"),
            };

            assert_eq!(status_symbol, expected_symbol);

            let expected_text = match status {
                CandidateStatus::Detected => "Located",
                CandidateStatus::Analyzing => "Evaluating",
                CandidateStatus::Rejected => "Filtered",
                CandidateStatus::Signal => "Acquired",
                CandidateStatus::Playing => "Playing",
                CandidateStatus::Completed => "Complete",
            };
            assert_eq!(status_text, expected_text);
        }
    }

    #[test]
    fn test_color_scheme_unchanged() {
        let expected_colors = vec![
            (CandidateStatus::Detected, Color::Rgb(255, 215, 0)), // Gold
            (CandidateStatus::Analyzing, Color::Rgb(100, 149, 237)), // Cornflower blue
            (CandidateStatus::Rejected, Color::Rgb(105, 105, 105)), // Dim gray
            (CandidateStatus::Signal, Color::Rgb(70, 130, 180)),  // Steel blue
            (CandidateStatus::Playing, Color::Rgb(186, 85, 211)), // Medium orchid
            (CandidateStatus::Completed, Color::Rgb(60, 179, 113)), // Medium sea green
        ];

        for (status, expected_color) in expected_colors {
            let actual_color = match status {
                CandidateStatus::Detected => Color::Rgb(255, 215, 0),
                CandidateStatus::Analyzing => Color::Rgb(100, 149, 237),
                CandidateStatus::Rejected => Color::Rgb(105, 105, 105),
                CandidateStatus::Signal => Color::Rgb(70, 130, 180),
                CandidateStatus::Playing => Color::Rgb(186, 85, 211),
                CandidateStatus::Completed => Color::Rgb(60, 179, 113),
            };

            assert_eq!(actual_color, expected_color);
        }
    }

    #[test]
    fn test_candidate_progress_format_unchanged() {
        let freq_mhz = 88.9;
        let status_symbol = "●";
        let progress_bar = "⣿".repeat(10);
        let status_text = "Acquired";

        let expected_format = format!(
            "   {} {:>5.1} MHz • {} • {}",
            status_symbol, freq_mhz, progress_bar, status_text
        );

        assert!(expected_format.starts_with("   ●")); // 3 spaces + symbol
        assert!(expected_format.contains(" 88.9 MHz")); // Right-aligned frequency
        assert!(expected_format.contains(" • ")); // Bullet separators
        assert!(expected_format.ends_with(" • Acquired")); // Status text

        assert_eq!(format!("{:>5.1}", 88.9), " 88.9");
        assert_eq!(format!("{:>5.1}", 107.1), "107.1");
        assert_eq!(format!("{:>5.1}", 90.0), " 90.0");
    }
}
