//! Caladan organic progress display with inline mini-graphs

use crate::terminal::tui::{
    model::{CandidateStatus, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

fn wrap_with_bracket(
    line: Line<'static>,
    left: char,
    right: char,
    _theme: &dyn Theme,
    in_selection_mode: bool,
) -> Line<'static> {
    let bracket_color = Color::Rgb(160, 200, 220); // Light cornflower blue
    let bracket_style = if in_selection_mode {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::DIM)
    };
    let mut spans = vec![
        Span::styled(" ".to_string(), bracket_style),
        Span::styled(left.to_string(), bracket_style),
        Span::styled(" ".to_string(), bracket_style),
    ];
    spans.extend(line.spans);
    spans.push(Span::styled(" ".to_string(), bracket_style));
    spans.push(Span::styled(right.to_string(), bracket_style));
    spans.push(Span::styled(" ".to_string(), bracket_style));
    Line::from(spans)
}

pub fn render_progress(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    if area.height < 2 {
        return;
    }

    // Count total displayable candidates for scroll indicators
    let displayable_windows = model.get_displayable_windows();
    let total_candidates: usize = displayable_windows
        .iter()
        .map(|(window_id, window)| {
            let is_current = **window_id == model.current_window;
            window
                .displayable_candidates(is_current, model.selection_mode)
                .len()
        })
        .sum();

    let mut lines = Vec::new();
    let max_lines = (area.height - 2) as usize;
    let mut line_count = 0;

    // Add scroll-up indicator if there's content above
    let has_content_above = model.scroll_offset > 0;
    if has_content_above {
        lines.push(Line::from(Span::styled(
            "       ↑ more above ↑",
            Style::default().fg(theme.instructions_dim()),
        )));
        line_count += 1;
    }

    // Start candidate_index at scroll_offset since we're skipping that many selectable candidates
    let mut candidate_index = 0;
    let mut skipped_count = 0;
    let mut rendered_candidates = 0;

    for (window_id, window) in displayable_windows.iter() {
        let is_current = **window_id == model.current_window;
        let displayable = window.displayable_candidates(is_current, model.selection_mode);

        for candidate in displayable {
            // Track candidate_index for ALL candidates (to match selection logic)
            let current_candidate_index = if candidate.status != CandidateStatus::Rejected {
                let idx = candidate_index;
                candidate_index += 1;
                idx
            } else {
                usize::MAX // Rejected candidates don't have an index
            };

            // Skip candidates before scroll offset
            if skipped_count < model.scroll_offset {
                skipped_count += 1;
                continue;
            }

            if line_count >= max_lines {
                break;
            }

            // Check if this candidate is selected
            let is_selected = model.selection_mode
                && candidate.status != CandidateStatus::Rejected
                && model.selected_candidate_index == Some(current_candidate_index);

            rendered_candidates += 1;

            let status_symbol = match candidate.status {
                CandidateStatus::Detected => "○",
                CandidateStatus::Analyzing => "◐",
                CandidateStatus::Rejected => "·",
                CandidateStatus::Signal => "◉",
                CandidateStatus::Playing => "◉",
                CandidateStatus::Completed => "◯",
            };

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

            let status_text = match candidate.status {
                CandidateStatus::Detected => "detecting",
                CandidateStatus::Analyzing => "forming",
                CandidateStatus::Rejected => "static",
                CandidateStatus::Signal => "present",
                CandidateStatus::Playing => "listening",
                CandidateStatus::Completed => "detected",
            };

            let freq_mhz = candidate.frequency_hz / 1e6;
            let progress_pct = (candidate.completion * 100.0) as u8;
            let mini_graph = create_mini_graph(progress_pct);

            let selection_prefix = if is_selected {
                theme.selection_indicator()
            } else {
                " "
            };

            let mut spans = vec![
                Span::styled(
                    format!("{} {} ", selection_prefix, status_symbol),
                    Style::default().fg(status_color),
                ),
                Span::styled(
                    format!("{:>5.1} MHz  ", freq_mhz),
                    Style::default()
                        .fg(if is_selected {
                            theme.selection_highlight()
                        } else {
                            theme.primary()
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{:<11}", status_text),
                    Style::default().fg(theme.foreground()),
                ),
                Span::raw("  "),
                Span::styled(mini_graph, Style::default().fg(status_color)),
            ];

            if let Some(quality) = &candidate.audio_quality {
                use crate::audio_quality::AudioQuality;
                let (quality_text, quality_color) = match quality {
                    AudioQuality::Good => ("good", theme.quality_good()),
                    AudioQuality::Moderate => ("moderate", theme.quality_moderate()),
                    AudioQuality::Poor => ("poor", theme.quality_poor()),
                    AudioQuality::NoAudio => ("no-audio", theme.quality_no_audio()),
                    AudioQuality::Static => ("static", theme.quality_static()),
                    AudioQuality::Unknown => ("unknown", theme.quality_unknown()),
                };
                spans.push(Span::raw("  "));
                spans.push(Span::styled(
                    format!("·{:<8}", quality_text), // Right-pad quality text to 9 chars (8 + ·)
                    Style::default().fg(quality_color),
                ));
            } else {
                // Add padding when no quality to maintain alignment
                spans.push(Span::raw("           ")); // 11 spaces (2 + 9)
            }

            lines.push(Line::from(spans));
            line_count += 1;
        }

        if line_count >= max_lines {
            break;
        }
    }

    // Add "Continue scan" option if in selection mode and scan is not complete
    if model.selection_mode && !model.all_complete() && line_count < max_lines {
        let is_continue_selected = model.is_continue_scan_selected();

        let color = if is_continue_selected {
            theme.selection_highlight()
        } else {
            theme.instructions_dim()
        };

        // Pad to match the width of candidate lines (47 chars total)
        // Candidate width: 4 (prefix+symbol) + 10 (freq) + 11 (status) + 2 (space) + 8 (graph) + 11 (quality) = 46
        // Need 47 to align properly with the bracket wrapper
        lines.push(Line::from(vec![Span::styled(
            format!("{:<47}", "Continue scan →"),
            Style::default()
                .fg(color)
                .add_modifier(if is_continue_selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        )]));
    }

    // Add scroll-down indicator if there's content below
    // We've rendered: scroll_offset (skipped) + rendered_candidates (shown) out of total_candidates
    let total_rendered_or_skipped = model.scroll_offset + rendered_candidates;
    let has_content_below = total_rendered_or_skipped < total_candidates;
    if has_content_below && line_count < max_lines {
        lines.push(Line::from(Span::styled(
            "       ↓ more below ↓",
            Style::default().fg(theme.instructions_dim()),
        )));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            " awaiting signals...",
            Style::default().fg(theme.instructions_dim()),
        )));
    }

    // Wrap all lines with brackets
    let line_count = lines.len();
    let wrapped_lines: Vec<Line> = lines
        .into_iter()
        .enumerate()
        .map(|(idx, line)| {
            let (left, right) = if idx == 0 {
                ('╭', '╮') // Top corners
            } else if idx == line_count - 1 {
                ('╰', '╯') // Bottom corners
            } else {
                ('│', '│') // Sides
            };
            wrap_with_bracket(line, left, right, theme, model.selection_mode)
        })
        .collect();

    let final_lines = wrapped_lines;

    let block = Block::default();
    let paragraph = Paragraph::new(final_lines).block(block);
    f.render_widget(paragraph, area);
}

fn create_mini_graph(progress: u8) -> String {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let width = 8;
    let filled = (progress as usize * width) / 100;

    (0..width)
        .map(|i| {
            if i < filled {
                chars[7]
            } else if i == filled && !(progress as usize).is_multiple_of(100 / width) {
                let partial = ((progress as usize % (100 / width)) * chars.len()) / (100 / width);
                chars[partial.min(chars.len() - 1)]
            } else {
                chars[0]
            }
        })
        .collect()
}
