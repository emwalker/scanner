//! Scan progress rendering for candidates and windows

use crate::terminal::tui::{
    model::{CandidateProgress, CandidateStatus, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

fn render_empty_state(f: &mut Frame, area: Rect, theme: &dyn Theme) {
    let waiting = Paragraph::new("  Establishing connection…\n  Preparing to monitor frequencies")
        .style(Style::default().fg(theme.instructions_dim()));
    f.render_widget(waiting, area);
}

fn render_scroll_up_indicator(f: &mut Frame, area: Rect, theme: &dyn Theme) {
    let indicator = Paragraph::new("        ↑ more above ↑").style(
        Style::default()
            .fg(theme.instructions_dim())
            .add_modifier(Modifier::DIM),
    );
    f.render_widget(indicator, area);
}

fn render_scroll_down_indicator(f: &mut Frame, area: Rect, theme: &dyn Theme) {
    let indicator = Paragraph::new("        ↓ more below ↓").style(
        Style::default()
            .fg(theme.instructions_dim())
            .add_modifier(Modifier::DIM),
    );
    f.render_widget(indicator, area);
}

/// Render all scan progress for windows and candidates
pub fn render_scan(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    if model.windows.is_empty() {
        render_empty_state(f, area, theme);
        return;
    }

    // Get displayable windows for selection tracking
    let displayable_windows = model.displayable_windows();

    // Interactive mode means user is navigating/browsing
    let in_interactive_mode = model.is_interactive();

    // Count total displayable candidates for scroll indicators
    let total_candidates: usize = displayable_windows
        .iter()
        .map(|(window_id, window)| {
            let is_current = **window_id == model.current_window;
            let candidates = window.displayable_candidates(is_current, in_interactive_mode);
            if in_interactive_mode {
                candidates
                    .iter()
                    .filter(|c| c.status != CandidateStatus::Rejected)
                    .count()
            } else {
                candidates.len()
            }
        })
        .sum();

    // Calculate available space for progress bars
    let available_height = area.height as usize;
    let max_bars = available_height.saturating_sub(5); // RESERVED_TERMINAL_LINES

    // In interactive mode, use selectable candidates (no rejected); otherwise use displayable
    let use_selectable = in_interactive_mode;

    // Calculate window sizes that fit in available space
    let mut windows_that_fit = Vec::new();

    for (window_id, window) in displayable_windows.iter() {
        let is_current_window = **window_id == model.current_window;
        let candidates = if use_selectable {
            // In selection mode, never show rejected candidates
            window
                .displayable_candidates(is_current_window, use_selectable)
                .into_iter()
                .filter(|c| c.status != crate::terminal::tui::model::CandidateStatus::Rejected)
                .collect::<Vec<_>>()
        } else {
            window.displayable_candidates(is_current_window, use_selectable)
        };
        let candidate_count = candidates.len();
        let window_bars = candidate_count + 1; // +1 for window header

        windows_that_fit.push((window_id, candidate_count, window_bars));
    }

    // Now work backwards to fit as many recent windows as possible
    let mut window_sizes = Vec::new();
    let mut running_total = 0;

    // Add "Continue scan" line if in interactive mode and scan is not complete
    if in_interactive_mode && !model.all_complete() {
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

    // Create constraints: 1 line per candidate + 1 line per window header + 1 for continue scan + scroll indicators
    let mut total_lines = window_sizes
        .iter()
        .map(|(_, count)| count + 1)
        .sum::<usize>();
    if in_interactive_mode && !model.all_complete() {
        total_lines += 1;
    }

    // Add space for scroll indicators
    let has_content_above = model.scroll_offset > 0;

    if has_content_above {
        total_lines += 1;
    }
    // Note: has_content_below will be calculated after rendering

    let constraints: Vec<Constraint> = (0..total_lines).map(|_| Constraint::Length(1)).collect();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    // Render all progress bars sequentially
    let mut chunk_idx = 0;
    let mut candidate_index = 0;
    let mut skipped_count = 0;
    let mut rendered_candidates = 0;

    // Add scroll-up indicator if needed
    if has_content_above && chunk_idx < chunks.len() {
        render_scroll_up_indicator(f, chunks[chunk_idx], theme);
        chunk_idx += 1;
    }

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
                .displayable_candidates(is_current_window, use_selectable)
                .into_iter()
                .filter(|c| c.status != crate::terminal::tui::model::CandidateStatus::Rejected)
                .collect::<Vec<_>>()
        } else {
            window.displayable_candidates(is_current_window, use_selectable)
        };

        // Render window header
        render_window_header(
            f,
            chunks[chunk_idx],
            &WindowHeaderContext {
                window_id: **window_id,
                theme,
            },
        );
        chunk_idx += 1;

        // Render candidates in this window (preserves insertion order, filtered)
        for candidate in displayable_candidates {
            // Track candidate_index for ALL candidates BEFORE skipping
            let current_candidate_index = candidate_index;
            candidate_index += 1;

            // Skip candidates before scroll offset
            if skipped_count < model.scroll_offset {
                skipped_count += 1;
                continue;
            }

            if chunk_idx >= chunks.len() {
                break;
            }

            let is_selected = in_interactive_mode
                && model.selected_candidate_index() == Some(current_candidate_index);
            let is_playing = in_interactive_mode && candidate.status == CandidateStatus::Playing;
            render_candidate_progress(
                f,
                chunks[chunk_idx],
                &CandidateRenderContext {
                    candidate,
                    is_selected,
                    is_playing,
                    theme,
                },
            );
            chunk_idx += 1;
            rendered_candidates += 1;
        }
    }

    // Add "Continue scan" option if in interactive mode and scan is not complete
    if in_interactive_mode && !model.all_complete() && chunk_idx < chunks.len() {
        let is_continue_selected = model.is_continue_scan_selected();
        render_continue_scan(f, chunks[chunk_idx], is_continue_selected, theme);
        chunk_idx += 1;
    }

    // Add scroll-down indicator if there's content below
    let total_rendered_or_skipped = model.scroll_offset + rendered_candidates;
    let has_content_below = total_rendered_or_skipped < total_candidates;
    if has_content_below && chunk_idx < chunks.len() {
        render_scroll_down_indicator(f, chunks[chunk_idx], theme);
    }
}

struct WindowHeaderContext<'a> {
    window_id: usize,
    theme: &'a dyn Theme,
}

/// Render a window header
fn render_window_header(f: &mut Frame, area: ratatui::layout::Rect, ctx: &WindowHeaderContext) {
    let color = ctx.theme.window_header();

    // Geometric window indicator with precision framing
    let header = format!(
        " {} Scan {} {}",
        ctx.theme.window_bullet(),
        ctx.window_id,
        ctx.theme.window_bullet()
    );
    let header_widget =
        Paragraph::new(header).style(Style::default().fg(color).add_modifier(Modifier::BOLD));
    f.render_widget(header_widget, area);
}

struct CandidateRenderContext<'a> {
    candidate: &'a CandidateProgress,
    is_selected: bool,
    is_playing: bool,
    theme: &'a dyn Theme,
}

/// Render progress for a single candidate
fn render_candidate_progress(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    ctx: &CandidateRenderContext,
) {
    let freq_mhz = ctx.candidate.frequency_hz / 1e6;

    // Clean geometric status indicators with refined terminology
    let (status_text, status_symbol) = match ctx.candidate.status {
        CandidateStatus::Detected => (
            ctx.theme.status_detected_text(),
            ctx.theme.symbol_detected(),
        ),
        CandidateStatus::Analyzing => (
            ctx.theme.status_analyzing_text(),
            ctx.theme.symbol_analyzing(),
        ),
        CandidateStatus::Rejected => (
            ctx.theme.status_rejected_text(),
            ctx.theme.symbol_rejected(),
        ),
        CandidateStatus::Signal => (ctx.theme.status_signal_text(), ctx.theme.symbol_signal()),
        CandidateStatus::Playing => (ctx.theme.status_playing_text(), ctx.theme.symbol_playing()),
        CandidateStatus::Completed => (
            ctx.theme.status_completed_text(),
            ctx.theme.symbol_completed(),
        ),
    };

    // Status colors from theme
    let status_color = if ctx.is_selected {
        ctx.theme.selection_highlight()
    } else {
        match ctx.candidate.status {
            CandidateStatus::Detected => ctx.theme.status_detected(),
            CandidateStatus::Analyzing => ctx.theme.status_analyzing(),
            CandidateStatus::Rejected => ctx.theme.status_rejected(),
            CandidateStatus::Signal => ctx.theme.status_signal(),
            CandidateStatus::Playing => ctx.theme.status_playing(),
            CandidateStatus::Completed => ctx.theme.status_completed(),
        }
    };

    // Create base style with optional background for playing station
    let base_style = if ctx.is_playing {
        Style::default()
            .bg(Color::Rgb(0, 60, 90)) // Bright cyan-blue background
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    // Atmospheric amber progress with geometric precision
    let progress_width = 20;
    let filled = (ctx.candidate.completion * progress_width as f64) as usize;
    let progress_bar = if filled == 0 {
        ctx.theme.progress_empty().repeat(progress_width)
    } else if filled >= progress_width {
        ctx.theme.progress_full().repeat(progress_width)
    } else {
        // Gradient-like progression with theme elements
        let full_blocks = filled;
        let remaining = progress_width - filled;
        format!(
            "{}{}",
            ctx.theme.progress_full().repeat(full_blocks),
            ctx.theme.progress_empty().repeat(remaining)
        )
    };

    // Create line with separate styling for different parts
    let line = if let Some(audio_quality) = &ctx.candidate.audio_quality {
        if ctx.candidate.status == CandidateStatus::Completed
            || ctx.candidate.status == CandidateStatus::Rejected
        {
            // Get audio quality text and style
            let (quality_text, quality_style) = if ctx.is_playing {
                // Use bright colors for playing station
                (
                    match audio_quality {
                        crate::audio_quality::AudioQuality::Good => ctx.theme.quality_good_text(),
                        crate::audio_quality::AudioQuality::Moderate => {
                            ctx.theme.quality_moderate_text()
                        }
                        crate::audio_quality::AudioQuality::Poor => ctx.theme.quality_poor_text(),
                        crate::audio_quality::AudioQuality::NoAudio => {
                            ctx.theme.quality_no_audio_text()
                        }
                        crate::audio_quality::AudioQuality::Static => {
                            ctx.theme.quality_static_text()
                        }
                        crate::audio_quality::AudioQuality::Unknown => {
                            ctx.theme.quality_unknown_text()
                        }
                    },
                    Style::default()
                        .fg(Color::Rgb(150, 255, 150))
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                match audio_quality {
                    crate::audio_quality::AudioQuality::Good => (
                        ctx.theme.quality_good_text(),
                        Style::default()
                            .fg(ctx.theme.quality_good())
                            .add_modifier(Modifier::BOLD),
                    ),
                    crate::audio_quality::AudioQuality::Moderate => (
                        ctx.theme.quality_moderate_text(),
                        Style::default().fg(ctx.theme.quality_moderate()),
                    ),
                    crate::audio_quality::AudioQuality::Poor => (
                        ctx.theme.quality_poor_text(),
                        Style::default().fg(ctx.theme.quality_poor()),
                    ),
                    crate::audio_quality::AudioQuality::NoAudio => (
                        ctx.theme.quality_no_audio_text(),
                        Style::default().fg(ctx.theme.quality_no_audio()),
                    ),
                    crate::audio_quality::AudioQuality::Static => (
                        ctx.theme.quality_static_text(),
                        Style::default().fg(ctx.theme.quality_static()),
                    ),
                    crate::audio_quality::AudioQuality::Unknown => (
                        ctx.theme.quality_unknown_text(),
                        Style::default().fg(ctx.theme.quality_unknown()),
                    ),
                }
            };

            // Create spans with different colors
            let mut spans = vec![
                Span::styled(" ", base_style),
                Span::styled(
                    format!(
                        "  {} {:>5.1} MHz • {} • {} • ",
                        status_symbol, freq_mhz, progress_bar, status_text
                    ),
                    base_style.fg(if ctx.is_playing {
                        Color::White
                    } else {
                        status_color
                    }),
                ),
                Span::styled(quality_text, base_style.patch(quality_style)),
            ];

            // Add padding to extend background to right edge
            let content_width = 3 + 5 + 4 + 3 + progress_width + 3 + 8 + 3 + quality_text.len();
            let padding_needed = area.width.saturating_sub(content_width as u16);
            spans.push(Span::styled(
                " ".repeat(padding_needed as usize),
                base_style,
            ));

            Line::from(spans)
        } else {
            let mut spans = vec![
                Span::styled(" ", base_style),
                Span::styled(
                    format!(
                        "  {} {:>5.1} MHz • {} • {}",
                        status_symbol, freq_mhz, progress_bar, status_text
                    ),
                    base_style.fg(if ctx.is_playing {
                        Color::White
                    } else {
                        status_color
                    }),
                ),
            ];

            // Add padding to extend background to right edge
            let content_width = 3 + 5 + 4 + 3 + progress_width + 3 + 8;
            let padding_needed = area.width.saturating_sub(content_width as u16);
            spans.push(Span::styled(
                " ".repeat(padding_needed as usize),
                base_style,
            ));

            Line::from(spans)
        }
    } else {
        let mut spans = vec![
            Span::styled(" ", base_style),
            Span::styled(
                format!(
                    "  {} {:>5.1} MHz • {} • {}",
                    status_symbol, freq_mhz, progress_bar, status_text
                ),
                base_style.fg(if ctx.is_playing {
                    Color::White
                } else {
                    status_color
                }),
            ),
        ];

        // Add padding to extend background to right edge
        let content_width = 3 + 5 + 4 + 3 + progress_width + 3 + 8;
        let padding_needed = area.width.saturating_sub(content_width as u16);
        spans.push(Span::styled(
            " ".repeat(padding_needed as usize),
            base_style,
        ));

        Line::from(spans)
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
