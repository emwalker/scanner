//! Caladan organic scan display with inline mini-graphs

use crate::terminal::tui::{
    model::{CandidateStatus, FocusState, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

pub fn render_scan(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    if area.height < 4 {
        return;
    }

    // Interactive mode means user is navigating/browsing
    let in_interactive_mode = model.is_interactive();

    // Count total displayable candidates for scroll indicators
    let displayable_windows = model.displayable_windows();
    let total_candidates: usize = displayable_windows
        .iter()
        .map(|(window_id, window)| {
            let is_current = **window_id == model.current_window;
            window
                .displayable_candidates(is_current, in_interactive_mode)
                .len()
        })
        .sum();

    // Calculate minimum width needed for content without wrapping
    // Format: " ◉ 107.1 MHz  detecting    ▁▁▁▁▁▁▁▁  ·moderate"
    // Breakdown: 1 (selection) + 1 (space) + 1 (symbol) + 1 (space) +
    //            5 (freq) + 4 (space+MHz) + 2 (space) + 11 (status) + 2 (space) +
    //            8 (graph) + 2 (space) + 1 (dot) + 8 (quality) = ~47 chars
    // Add 2 for padding (1 on each side) + 2 for borders = 51 total width needed
    let min_content_width = 47;
    let total_min_width = min_content_width + 4; // +2 padding +2 borders

    // Responsive width: use half of terminal width in wide terminals, full width in narrow ones
    let terminal_width = area.width as usize;
    let wide_threshold = 100; // Terminals wider than this are considered "wide"

    let progress_width = if terminal_width >= wide_threshold {
        // Wide terminal: use half width with margin on the right
        let half_width = terminal_width / 2;
        half_width.max(total_min_width)
    } else {
        // Narrow terminal: use full width
        terminal_width.max(total_min_width)
    };

    // Constrain to calculated width
    let constrained_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(progress_width as u16),
            Constraint::Min(0),
        ])
        .split(area)[0];

    let bracket_color = Color::Rgb(160, 200, 220);
    let has_focus = matches!(model.focus_state, FocusState::Scan);

    let border_style = if has_focus {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::DIM)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .border_set(ratatui::symbols::border::Set {
            top_left: "╭",
            top_right: "╮",
            bottom_left: "╰",
            bottom_right: "╯",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: "─",
            horizontal_bottom: "─",
        })
        .padding(ratatui::widgets::Padding::horizontal(1));

    let inner = block.inner(constrained_area);

    let mut lines = Vec::new();
    let max_lines = inner.height as usize;
    let mut line_count = 0;

    // Add scroll-up indicator if there's content above
    let has_content_above = model.scroll_offset > 0;
    if has_content_above {
        lines.push(Line::from(Span::styled(
            "↑ more above ↑",
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
        let displayable = window.displayable_candidates(is_current, in_interactive_mode);

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

            // Check if this candidate is selected (for arrow key navigation)
            let is_selected = in_interactive_mode
                && candidate.status != CandidateStatus::Rejected
                && model.selected_candidate_index() == Some(current_candidate_index);

            // Determine which station should be highlighted
            // In AwaitingTune/Listening modes, highlight the station being tuned/played, not arrow selection
            let should_highlight = match &model.ui_mode {
                crate::terminal::tui::model::UiMode::AwaitingTune { tuning_index, .. } => {
                    current_candidate_index == *tuning_index
                }
                crate::terminal::tui::model::UiMode::Listening { playing_index, .. } => {
                    current_candidate_index == *playing_index
                }
                _ => false,
            };

            rendered_candidates += 1;

            lines.push(render_candidate_line(
                candidate,
                is_selected,
                should_highlight,
                theme,
                inner.width,
            ));
            line_count += 1;
        }

        if line_count >= max_lines {
            break;
        }
    }

    // Add "Continue scan" option if in interactive mode and scan is not complete
    if in_interactive_mode && !model.all_complete() && line_count < max_lines {
        let is_continue_selected = model.is_continue_scan_selected();

        let color = if is_continue_selected {
            theme.selection_highlight()
        } else {
            theme.instructions_dim()
        };

        lines.push(Line::from(vec![Span::styled(
            "Continue scan →",
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
            "↓ more below ↓",
            Style::default().fg(theme.instructions_dim()),
        )));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "awaiting signals...",
            Style::default().fg(theme.instructions_dim()),
        )));
    }

    let paragraph = Paragraph::new(lines);

    f.render_widget(block, constrained_area);
    f.render_widget(paragraph, inner);
}

fn render_candidate_line(
    candidate: &crate::terminal::tui::model::CandidateProgress,
    is_selected: bool,
    should_highlight: bool,
    theme: &dyn Theme,
    inner_width: u16,
) -> Line<'static> {
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
        CandidateStatus::Detected => theme.status_detected_text(),
        CandidateStatus::Analyzing => theme.status_analyzing_text(),
        CandidateStatus::Rejected => theme.status_rejected_text(),
        CandidateStatus::Signal => theme.status_signal_text(),
        CandidateStatus::Playing => theme.status_playing_text(),
        CandidateStatus::Completed => theme.status_completed_text(),
    };

    let freq_mhz = candidate.frequency_hz / 1e6;
    let progress_pct = (candidate.completion * 100.0) as u8;
    let mini_graph = create_mini_graph(progress_pct);

    let selection_prefix = if should_highlight {
        "▶"
    } else if is_selected {
        theme.selection_indicator()
    } else {
        " "
    };

    let base_style = if should_highlight {
        Style::default()
            .bg(Color::Rgb(0, 60, 90))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    let mut spans = vec![Span::styled(" ", base_style)];

    spans.push(Span::styled(
        format!("{} {} ", selection_prefix, status_symbol),
        base_style.fg(if should_highlight {
            Color::White
        } else {
            status_color
        }),
    ));
    spans.push(Span::styled(
        format!("{:>5.1} MHz  ", freq_mhz),
        base_style
            .fg(if should_highlight {
                Color::White
            } else if is_selected {
                theme.selection_highlight()
            } else {
                theme.primary()
            })
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        format!("{:<8}", status_text),
        base_style.fg(if should_highlight {
            Color::Rgb(200, 220, 255)
        } else {
            theme.foreground()
        }),
    ));
    spans.push(Span::styled("  ", base_style));
    spans.push(Span::styled(
        mini_graph,
        base_style.fg(if should_highlight {
            Color::White
        } else {
            status_color
        }),
    ));

    if let Some(quality) = &candidate.audio_quality {
        use crate::audio_quality::AudioQuality;
        let (quality_text, quality_color) = match quality {
            AudioQuality::Good => (theme.quality_good_text(), theme.quality_good()),
            AudioQuality::Moderate => (theme.quality_moderate_text(), theme.quality_moderate()),
            AudioQuality::Poor => (theme.quality_poor_text(), theme.quality_poor()),
            AudioQuality::NoAudio => (theme.quality_no_audio_text(), theme.quality_no_audio()),
            AudioQuality::Static => (theme.quality_static_text(), theme.quality_static()),
            AudioQuality::Unknown => (theme.quality_unknown_text(), theme.quality_unknown()),
        };
        spans.push(Span::styled("  ", base_style));
        spans.push(Span::styled(
            format!("·{:<8}", quality_text),
            base_style.fg(if should_highlight {
                Color::Rgb(150, 255, 150)
            } else {
                quality_color
            }),
        ));
    } else {
        spans.push(Span::styled("           ", base_style));
    }

    let content_width = 3 + 5 + 4 + 2 + 8 + 2 + 5 + 2 + 9;
    let padding_needed = inner_width.saturating_sub(content_width as u16);
    spans.push(Span::styled(
        " ".repeat(padding_needed as usize),
        base_style,
    ));

    Line::from(spans)
}

fn create_mini_graph(progress: u8) -> String {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let width = 5;
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
