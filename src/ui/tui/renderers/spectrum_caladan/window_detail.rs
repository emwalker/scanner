use crate::ui::tui::{model::Model, themes::Theme};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

pub(super) fn render_window_detail_row(
    width: usize,
    window_start: Option<f64>,
    window_width: f64,
    model: &Model,
    theme: &dyn Theme,
) -> Line<'static> {
    let mut chars = vec![' '; width];
    let mut colors: Vec<Option<ratatui::style::Color>> = vec![None; width];

    if let Some(ws) = window_start {
        let mut stations: Vec<_> = model.spectrum_stations.iter().collect();

        stations.sort_by(|a, b| {
            a.frequency_hz
                .partial_cmp(&b.frequency_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for station in stations {
            let freq_in_window = station.frequency_hz - ws;
            if freq_in_window >= 0.0 && freq_in_window <= window_width {
                let pos = (freq_in_window / window_width * width as f64) as usize;
                let freq_mhz = station.frequency_hz / 1e6;
                let label = format!("{:.1}", freq_mhz);

                let char = quality_char(&station.audio_quality);
                let color = quality_color(&station.audio_quality, station.is_active, theme);

                place_station_marker(&mut chars, &mut colors, pos, char, &label, color);
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

fn quality_char(quality: &Option<crate::audio::quality::AudioQuality>) -> char {
    match quality {
        Some(crate::audio::quality::AudioQuality::Good) => '●',
        Some(crate::audio::quality::AudioQuality::Moderate) => '◐',
        Some(crate::audio::quality::AudioQuality::Poor) => '◦',
        Some(crate::audio::quality::AudioQuality::Static) => '⋯',
        Some(crate::audio::quality::AudioQuality::NoAudio) => '·',
        _ => '○',
    }
}

fn quality_color(
    quality: &Option<crate::audio::quality::AudioQuality>,
    is_active: bool,
    theme: &dyn Theme,
) -> Color {
    if is_active {
        return theme.status_playing();
    }
    match quality {
        Some(crate::audio::quality::AudioQuality::Good) => theme.quality_good(),
        Some(crate::audio::quality::AudioQuality::Moderate) => theme.quality_moderate(),
        Some(crate::audio::quality::AudioQuality::Poor) => theme.quality_poor(),
        Some(crate::audio::quality::AudioQuality::Static) => theme.quality_static(),
        Some(crate::audio::quality::AudioQuality::NoAudio) => theme.quality_no_audio(),
        _ => theme.status_detected(),
    }
}

fn place_station_marker(
    chars: &mut [char],
    colors: &mut [Option<Color>],
    pos: usize,
    quality_char: char,
    label: &str,
    color: Color,
) {
    let width = chars.len();
    let station_width = 1 + label.len();

    if pos + station_width > width {
        return;
    }

    if pos < width {
        chars[pos] = quality_char;
        colors[pos] = Some(color);
    }

    for (i, ch) in label.chars().enumerate() {
        let char_pos = pos + 1 + i;
        if char_pos < width {
            chars[char_pos] = ch;
            colors[char_pos] = Some(color);
        }
    }
}
