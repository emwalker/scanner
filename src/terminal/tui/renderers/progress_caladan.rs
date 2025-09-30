//! Caladan organic progress display with inline mini-graphs

use crate::terminal::tui::{
    model::{CandidateStatus, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

pub fn render_progress(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    if area.height < 2 {
        return;
    }

    let mut lines = Vec::new();
    let max_lines = (area.height - 2) as usize;
    let mut line_count = 0;

    // Get displayable windows for selection tracking
    let displayable_windows = model.get_displayable_windows();
    let mut candidate_index = 0;

    for (window_id, window) in displayable_windows.iter() {
        let is_current = **window_id == model.current_window;
        let displayable = window.displayable_candidates(is_current);

        for candidate in displayable {
            if line_count >= max_lines {
                break;
            }

            // Only count non-rejected candidates for selection index
            // (matches get_selectable_candidates logic)
            let is_selected = model.selection_mode
                && candidate.status != CandidateStatus::Rejected
                && model.selected_candidate_index == Some(candidate_index);

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
                    format!(" {} {} ", selection_prefix, status_symbol),
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
                    format!("·{}", quality_text),
                    Style::default().fg(quality_color),
                ));
            }

            lines.push(Line::from(spans));
            line_count += 1;

            // Only increment candidate_index for selectable (non-rejected) candidates
            if candidate.status != CandidateStatus::Rejected {
                candidate_index += 1;
            }
        }

        if line_count >= max_lines {
            break;
        }
    }

    // Add "Continue scan" option if in selection mode
    if model.selection_mode && line_count < max_lines {
        let is_continue_selected = model.is_continue_scan_selected();

        let color = if is_continue_selected {
            theme.selection_highlight()
        } else {
            theme.instructions_dim()
        };

        lines.push(Line::from(vec![Span::styled(
            " Continue scan →",
            Style::default()
                .fg(color)
                .add_modifier(if is_continue_selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        )]));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  awaiting signals...",
            Style::default().fg(theme.instructions_dim()),
        )));
    }

    let block = Block::default();
    let paragraph = Paragraph::new(lines).block(block);
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
