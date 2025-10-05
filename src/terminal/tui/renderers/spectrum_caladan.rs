//! Caladan wave-based spectrum visualization

use crate::terminal::tui::{
    model::{FocusState, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

pub fn render_spectrum(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    use std::sync::OnceLock;
    use std::time::Instant;

    static ANIMATION_START: OnceLock<Instant> = OnceLock::new();
    let start = ANIMATION_START.get_or_init(Instant::now);
    let animation_time = start.elapsed().as_secs_f32();

    if area.height < 4 {
        return;
    }

    let fm_start = 88.0e6;
    let fm_end = 108.0e6;
    let fm_range = fm_end - fm_start;
    let window_width = 2.4e6;

    // Use selected candidate's center frequency if in selection mode, otherwise current window
    let window_start = if model.selection_mode {
        model
            .selected_candidate_info()
            .map(|(_, center_freq, _, _, _)| center_freq - window_width / 2.0)
    } else {
        model.windows.get(&model.current_window).map(|w| {
            if !w.candidates.is_empty() {
                let center = w.candidates.iter().map(|c| c.frequency_hz).sum::<f64>()
                    / w.candidates.len() as f64;
                center - window_width / 2.0
            } else {
                fm_start
            }
        })
    };

    let bracket_color = Color::Rgb(160, 200, 220);
    let has_focus = matches!(model.focus_state, FocusState::Spectrum);

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

    let inner = block.inner(area);
    let content_width = inner.width as usize;

    // Split the inner area to separate the bottom row for the window detail box
    let layout = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(3), // Top 3 rows (freq labels, spectrum, window freq labels)
            ratatui::layout::Constraint::Length(3), // Bottom box (1 content + 2 borders)
        ])
        .split(inner);

    let top_area = layout[0];
    let window_detail_area = layout[1];

    // Render top 3 rows
    let top_lines = vec![
        render_frequency_labels(content_width, fm_start, fm_range, theme),
        render_full_spectrum_row(
            content_width,
            fm_start,
            fm_range,
            window_start,
            window_width,
            theme,
            animation_time,
        ),
        render_window_frequency_labels(content_width, window_start, window_width, theme),
    ];

    let top_paragraph = Paragraph::new(top_lines);
    f.render_widget(block, area);
    f.render_widget(top_paragraph, top_area);

    // Create a subtle box for the window detail row with dim bracket color
    let window_detail_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(
            Style::default()
                .fg(bracket_color)
                .add_modifier(Modifier::DIM),
        )
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

    let window_detail_inner = window_detail_block.inner(window_detail_area);
    let window_detail_width = window_detail_inner.width as usize;

    let window_detail_line = render_window_detail_row(
        window_detail_width,
        window_start,
        window_width,
        model,
        theme,
    );

    let window_detail_paragraph = Paragraph::new(vec![window_detail_line]);

    f.render_widget(window_detail_block, window_detail_area);
    f.render_widget(window_detail_paragraph, window_detail_inner);
}

fn render_frequency_labels(
    width: usize,
    fm_start: f64,
    fm_range: f64,
    theme: &dyn Theme,
) -> Line<'static> {
    let mut chars = vec![' '; width];
    let mut is_label = vec![false; width];
    let label_positions = [0.0, 0.25, 0.5, 0.75];

    let mut labels_placed = vec![false; label_positions.len()];

    for col in 0..width {
        let pos = col as f64 / width as f64;

        for (label_idx, &label_pos) in label_positions.iter().enumerate() {
            if !labels_placed[label_idx] && (pos - label_pos).abs() < 0.01 {
                let freq = fm_start + (fm_range * label_pos);
                let freq_mhz_raw = freq / 1_000_000.0;
                let freq_mhz = (freq_mhz_raw * 10.0).round() / 10.0;
                let label = format!("{:.1}", freq_mhz);

                if col + label.len() <= width {
                    for (i, ch) in label.chars().enumerate() {
                        let char_pos = col + i;
                        if char_pos < width {
                            chars[char_pos] = ch;
                            is_label[char_pos] = true;
                        }
                    }
                    labels_placed[label_idx] = true;
                }
                break;
            }
        }
    }

    let fm_end = fm_start + fm_range;
    let end_freq_mhz_raw = fm_end / 1_000_000.0;
    let end_freq_mhz = (end_freq_mhz_raw * 10.0).round() / 10.0;
    let end_label = format!("{:.1}", end_freq_mhz);
    let end_pos = width.saturating_sub(end_label.len());
    for (i, ch) in end_label.chars().enumerate() {
        let char_pos = end_pos + i;
        if char_pos < width {
            chars[char_pos] = ch;
            is_label[char_pos] = true;
        }
    }

    let spans: Vec<Span> = chars
        .iter()
        .zip(is_label.iter())
        .map(|(&ch, &labeled)| {
            if labeled {
                Span::styled(
                    ch.to_string(),
                    Style::default().fg(theme.instructions_dim()),
                )
            } else {
                Span::raw(ch.to_string())
            }
        })
        .collect();

    Line::from(spans)
}

fn render_window_frequency_labels(
    width: usize,
    window_start: Option<f64>,
    window_width: f64,
    theme: &dyn Theme,
) -> Line<'static> {
    let mut result = vec![' '; width];

    if let Some(ws) = window_start {
        let fm_start = 88.0e6;
        let fm_range = 20.0e6;

        let window_start_pos = ((ws - fm_start) / fm_range * width as f64) as usize;
        let window_end_pos = (((ws + window_width) - fm_start) / fm_range * width as f64) as usize;

        let start_mhz_raw = ws / 1_000_000.0;
        let start_mhz = (start_mhz_raw * 10.0).round() / 10.0;
        let end_mhz_raw = (ws + window_width) / 1_000_000.0;
        let end_mhz = (end_mhz_raw * 10.0).round() / 10.0;

        let start_label = format!("{:.1}", start_mhz);
        let end_label = format!("{:.1}", end_mhz);

        // Center the start label on the left edge, shifted one column right
        let start_label_offset = window_start_pos.saturating_sub(start_label.len() / 2) + 1;
        for (i, ch) in start_label.chars().enumerate() {
            let pos = start_label_offset + i;
            if pos < width {
                result[pos] = ch;
            }
        }

        // Center the end label on the right edge, shifted one column right
        let end_label_offset = window_end_pos.saturating_sub(end_label.len() / 2) + 1;
        for (i, ch) in end_label.chars().enumerate() {
            let pos = end_label_offset + i;
            if pos < width {
                result[pos] = ch;
            }
        }
    }

    let spans: Vec<Span> = result
        .iter()
        .map(|&ch| {
            if ch == ' ' {
                Span::raw(" ")
            } else {
                Span::styled(ch.to_string(), Style::default().fg(theme.primary()))
            }
        })
        .collect();

    Line::from(spans)
}

fn render_full_spectrum_row(
    width: usize,
    fm_start: f64,
    fm_range: f64,
    window_start: Option<f64>,
    window_width: f64,
    theme: &dyn Theme,
    animation_time: f32,
) -> Line<'static> {
    let mut spans = Vec::new();

    for col in 0..width {
        let pos = col as f64 / width as f64;
        let freq = fm_start + (fm_range * pos);

        let in_window = if let Some(ws) = window_start {
            freq >= ws && freq <= ws + window_width
        } else {
            false
        };

        // Slower speeds (1.2, 0.7, 0.9)
        // Add vertical undulation: vary amplitude based on position along the wave
        let horizontal_phase = col as f32 * 0.3 + animation_time * 1.2;
        let amplitude_modulation = (horizontal_phase * 0.5).sin() * 0.3 + 0.7; // Undulates between 0.4 and 1.0

        let wave1 = horizontal_phase.sin() * 0.35 * amplitude_modulation;
        let wave2 = (col as f32 * 0.17 + animation_time * 0.7).sin() * 0.25;
        let wave3 = (col as f32 * 0.43 + animation_time * 0.9).sin() * 0.15;
        let offset = (wave1 + wave2 + wave3) * 0.5 + 0.5;

        let ch = if in_window {
            '▬'
        } else if offset > 0.65 {
            '≋'
        } else if offset > 0.45 {
            '≈'
        } else if offset > 0.25 {
            '~'
        } else {
            '·'
        };

        let color = if in_window {
            theme.spectrum_window()
        } else {
            theme.secondary()
        };

        spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
    }

    Line::from(spans)
}

fn render_window_detail_row(
    width: usize,
    window_start: Option<f64>,
    window_width: f64,
    model: &Model,
    theme: &dyn Theme,
) -> Line<'static> {
    let mut chars = vec![' '; width];
    let mut colors: Vec<Option<ratatui::style::Color>> = vec![None; width];

    if let Some(ws) = window_start
        && let Some(current_window) = model.windows.get(&model.current_window)
    {
        let mut stations: Vec<_> = current_window.candidates.iter().collect();

        stations.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());

        for candidate in stations {
            let freq_in_window = candidate.frequency_hz - ws;
            if freq_in_window >= 0.0 && freq_in_window <= window_width {
                let pos = (freq_in_window / window_width * width as f64) as usize;
                let freq_mhz = candidate.frequency_hz / 1e6;
                let label = format!("{:.1}", freq_mhz);
                let rejected =
                    candidate.status == crate::terminal::tui::model::CandidateStatus::Rejected;

                let quality_char = match &candidate.audio_quality {
                    Some(crate::audio_quality::AudioQuality::Good) => '●',
                    Some(crate::audio_quality::AudioQuality::Moderate) => '◐',
                    Some(crate::audio_quality::AudioQuality::Poor) => '◦',
                    Some(crate::audio_quality::AudioQuality::Static) => '⋯',
                    Some(crate::audio_quality::AudioQuality::NoAudio) => '·',
                    _ => '○',
                };

                let quality_color = if rejected {
                    theme.status_rejected()
                } else {
                    match &candidate.audio_quality {
                        Some(crate::audio_quality::AudioQuality::Good) => theme.quality_good(),
                        Some(crate::audio_quality::AudioQuality::Moderate) => {
                            theme.quality_moderate()
                        }
                        Some(crate::audio_quality::AudioQuality::Poor) => theme.quality_poor(),
                        Some(crate::audio_quality::AudioQuality::Static) => theme.quality_static(),
                        Some(crate::audio_quality::AudioQuality::NoAudio) => {
                            theme.quality_no_audio()
                        }
                        _ => theme.status_detected(),
                    }
                };

                let station_width = 1 + label.len();
                if pos + station_width <= width {
                    if pos < width {
                        chars[pos] = quality_char;
                        colors[pos] = Some(quality_color);
                    }
                    for (i, ch) in label.chars().enumerate() {
                        let char_pos = pos + 1 + i;
                        if char_pos < width {
                            chars[char_pos] = ch;
                            colors[char_pos] = Some(quality_color);
                        }
                    }
                }
            }
        }
    }

    let spans: Vec<Span> = chars
        .iter()
        .zip(colors.iter())
        .map(|(&ch, &color)| {
            if let Some(c) = color {
                Span::styled(ch.to_string(), Style::default().fg(c))
            } else {
                Span::raw(ch.to_string())
            }
        })
        .collect();

    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_quality::AudioQuality;
    use crate::terminal::tui::{
        model::{CandidateProgress, CandidateStatus, Model, WindowProgress},
        themes::Theme,
    };
    use ratatui::style::Color;

    struct MockTheme;

    impl crate::terminal::tui::themes::ColorScheme for MockTheme {
        fn primary(&self) -> Color {
            Color::Blue
        }
        fn secondary(&self) -> Color {
            Color::Gray
        }
        fn accent(&self) -> Color {
            Color::Cyan
        }
        fn background(&self) -> Color {
            Color::Black
        }
        fn foreground(&self) -> Color {
            Color::White
        }
        fn status_detected(&self) -> Color {
            Color::Yellow
        }
        fn status_analyzing(&self) -> Color {
            Color::Yellow
        }
        fn status_rejected(&self) -> Color {
            Color::Gray
        }
        fn status_signal(&self) -> Color {
            Color::Green
        }
        fn status_playing(&self) -> Color {
            Color::Green
        }
        fn status_completed(&self) -> Color {
            Color::Blue
        }
        fn quality_good(&self) -> Color {
            Color::Green
        }
        fn quality_moderate(&self) -> Color {
            Color::Yellow
        }
        fn quality_poor(&self) -> Color {
            Color::Red
        }
        fn quality_no_audio(&self) -> Color {
            Color::Gray
        }
        fn quality_static(&self) -> Color {
            Color::DarkGray
        }
        fn quality_unknown(&self) -> Color {
            Color::Gray
        }
        fn header_accent(&self) -> Color {
            Color::Cyan
        }
        fn spectrum_window(&self) -> Color {
            Color::Cyan
        }
        fn instructions_dim(&self) -> Color {
            Color::DarkGray
        }
        fn window_header(&self) -> Color {
            Color::Blue
        }
        fn selection_highlight(&self) -> Color {
            Color::Cyan
        }
    }

    impl crate::terminal::tui::themes::SymbolSet for MockTheme {
        fn symbol_detected(&self) -> &'static str {
            "○"
        }
        fn symbol_analyzing(&self) -> &'static str {
            "◐"
        }
        fn symbol_rejected(&self) -> &'static str {
            "·"
        }
        fn symbol_signal(&self) -> &'static str {
            "◉"
        }
        fn symbol_playing(&self) -> &'static str {
            "◉"
        }
        fn symbol_completed(&self) -> &'static str {
            "◯"
        }
        fn progress_empty(&self) -> &'static str {
            "░"
        }
        fn progress_full(&self) -> &'static str {
            "█"
        }
        fn spectrum_baseline(&self) -> char {
            '≈'
        }
        fn spectrum_window_char(&self) -> char {
            '▬'
        }
        fn window_bullet(&self) -> &'static str {
            "•"
        }
        fn header_border(&self) -> char {
            '─'
        }
        fn selection_indicator(&self) -> &'static str {
            ">"
        }
    }

    impl crate::terminal::tui::themes::TextStyle for MockTheme {
        fn title(&self) -> &'static str {
            "SCANNER"
        }
        fn subtitle(&self) -> &'static str {
            "FM Band Monitor"
        }
        fn status_detected_text(&self) -> &'static str {
            "detected"
        }
        fn status_analyzing_text(&self) -> &'static str {
            "analyzing"
        }
        fn status_rejected_text(&self) -> &'static str {
            "rejected"
        }
        fn status_signal_text(&self) -> &'static str {
            "signal"
        }
        fn status_playing_text(&self) -> &'static str {
            "playing"
        }
        fn status_completed_text(&self) -> &'static str {
            "completed"
        }
    }

    impl Theme for MockTheme {
        fn name(&self) -> &str {
            "mock"
        }
        fn is_dark(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_frequency_labels_exact_character_count() {
        let theme = MockTheme;
        let width = 100;
        let line = render_frequency_labels(width, 88.0e6, 20.0e6, &theme);

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Frequency labels must produce exactly {width} characters, got {char_count}"
        );
    }

    #[test]
    fn test_frequency_labels_correct_mhz_values() {
        let theme = MockTheme;
        let width = 100;
        let line = render_frequency_labels(width, 88.0e6, 20.0e6, &theme);

        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

        assert!(content.contains("88.0"), "Should contain 88.0 MHz");
        assert!(content.contains("108.0"), "Should contain 108.0 MHz");
        assert!(
            !content.contains("11103") && !content.contains("9998"),
            "Should not contain overlapping digits like 11103 or 9998"
        );
    }

    #[test]
    fn test_frequency_labels_no_overlapping() {
        let theme = MockTheme;
        let width = 100;
        let line = render_frequency_labels(width, 88.0e6, 20.0e6, &theme);

        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

        let label_88_count = content.matches("88.0").count();
        let label_108_count = content.matches("108.0").count();

        assert_eq!(label_88_count, 1, "88.0 should appear exactly once");
        assert_eq!(label_108_count, 1, "108.0 should appear exactly once");
    }

    #[test]
    fn test_window_frequency_labels_exact_character_count() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let line = render_window_frequency_labels(width, window_start, window_width, &theme);

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Window labels must produce exactly {width} characters, got {char_count}"
        );
    }

    #[test]
    fn test_window_frequency_labels_correct_values() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let line = render_window_frequency_labels(width, window_start, window_width, &theme);
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

        assert!(
            content.contains("89.5"),
            "Should contain start frequency 89.5 MHz"
        );
        assert!(
            content.contains("91.9"),
            "Should contain end frequency 91.9 MHz"
        );
    }

    #[test]
    fn test_full_spectrum_row_exact_character_count() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let line = render_full_spectrum_row(
            width,
            88.0e6,
            20.0e6,
            window_start,
            window_width,
            &theme,
            0.0,
        );

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Full spectrum row must produce exactly {width} characters, got {char_count}"
        );
    }

    #[test]
    fn test_window_detail_row_exact_character_count() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;
        let model = Model::new();

        let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Window detail row must produce exactly {width} characters, got {char_count}"
        );
    }

    #[test]
    fn test_window_detail_row_with_stations() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let mut model = Model::new();
        let window = WindowProgress {
            window_id: 1,
            candidates: vec![
                CandidateProgress {
                    frequency_hz: 89.9e6,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: 89.9e6,
                        window_id: 1,
                    },
                    status: CandidateStatus::Signal,
                    completion: 0.8,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.5),
                    last_update: std::time::Instant::now(),
                },
                CandidateProgress {
                    frequency_hz: 90.5e6,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: 90.5e6,
                        window_id: 1,
                    },
                    status: CandidateStatus::Signal,
                    completion: 0.6,
                    audio_quality: Some(AudioQuality::Moderate),
                    signal_strength: Some(0.4),
                    last_update: std::time::Instant::now(),
                },
            ],
            is_complete: false,
            candidate_lookup: Default::default(),
        };
        model.windows.insert(1, window);
        model.current_window = 1;

        let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Window detail row with stations must produce exactly {width} characters"
        );

        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            content.contains("89.9") || content.contains("90.5"),
            "Should contain at least one station frequency"
        );
    }

    #[test]
    fn test_window_detail_row_no_overflow() {
        let theme = MockTheme;
        let width = 50;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let mut model = Model::new();
        let window = WindowProgress {
            window_id: 1,
            candidates: vec![CandidateProgress {
                frequency_hz: 91.8e6,
                metadata: crate::window::WindowMetadata {
                    center_frequency_hz: 91.8e6,
                    window_id: 1,
                },
                status: CandidateStatus::Signal,
                completion: 1.0,
                audio_quality: Some(AudioQuality::Good),
                signal_strength: Some(0.5),
                last_update: std::time::Instant::now(),
            }],
            is_complete: false,
            candidate_lookup: Default::default(),
        };
        model.windows.insert(1, window);
        model.current_window = 1;

        let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

        let char_count: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(
            char_count, width,
            "Station near edge must not overflow, got {char_count} chars for width {width}"
        );
    }

    #[test]
    fn test_rejected_stations_shown() {
        let theme = MockTheme;
        let width = 100;
        let window_start = Some(89.5e6);
        let window_width = 2.4e6;

        let mut model = Model::new();
        let window = WindowProgress {
            window_id: 1,
            candidates: vec![
                CandidateProgress {
                    frequency_hz: 89.9e6,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: 89.9e6,
                        window_id: 1,
                    },
                    status: CandidateStatus::Signal,
                    completion: 0.8,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.5),
                    last_update: std::time::Instant::now(),
                },
                CandidateProgress {
                    frequency_hz: 90.5e6,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: 90.5e6,
                        window_id: 1,
                    },
                    status: CandidateStatus::Rejected,
                    completion: 1.0,
                    audio_quality: Some(AudioQuality::NoAudio),
                    signal_strength: Some(0.1),
                    last_update: std::time::Instant::now(),
                },
            ],
            is_complete: false,
            candidate_lookup: Default::default(),
        };
        model.windows.insert(1, window);
        model.current_window = 1;

        let line = render_window_detail_row(width, window_start, window_width, &model, &theme);
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

        assert!(
            content.contains("89.9"),
            "Should contain non-rejected station 89.9"
        );
        assert!(
            content.contains("90.5"),
            "Should contain rejected station 90.5"
        );
    }
}
