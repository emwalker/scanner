use crate::ui::tui::themes::Theme;
use ratatui::{
    style::Style,
    text::{Line, Span},
};

pub(super) fn place_frequency_label(
    chars: &mut [char],
    is_label: &mut [bool],
    label: &str,
    start_pos: usize,
) -> bool {
    let width = chars.len();

    if start_pos + label.len() > width {
        return false;
    }

    for (i, ch) in label.chars().enumerate() {
        let char_pos = start_pos + i;
        if char_pos < width {
            chars[char_pos] = ch;
            is_label[char_pos] = true;
        }
    }

    true
}

pub(super) fn render_frequency_labels(
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

                if place_frequency_label(&mut chars, &mut is_label, &label, col) {
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
    place_frequency_label(&mut chars, &mut is_label, &end_label, end_pos);

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

pub(super) fn render_window_frequency_labels(
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
