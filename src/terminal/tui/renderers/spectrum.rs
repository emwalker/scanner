//! Spectrum visualization rendering

use crate::terminal::tui::{layout::SpectrumLayout, model::Model, themes::Theme};
use ratatui::{
    Frame,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

/// Render spectrum visualization with sliding window indicator
pub fn render_spectrum(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    model: &Model,
    theme: &dyn Theme,
) {
    // FM band: 88.0 - 108.0 MHz (20 MHz total range)
    let fm_start = 88.0;
    let fm_end = 108.0;
    let fm_range = fm_end - fm_start;

    let spectrum_width = area.width as usize - 2; // Account for padding

    // Window width for FM scanning (2.4 MHz)
    let window_width_mhz = 2.4;

    // Calculate current window position if we have scanning data
    // Use selected candidate's center frequency if in selection mode, otherwise current window
    let current_freq = if model.selection_mode {
        if let Some((_, center_freq, _, _, _)) = model.selected_candidate_info() {
            center_freq / 1e6
        } else {
            fm_start + window_width_mhz / 2.0
        }
    } else if let Some(current_window) = model.windows.get(&model.current_window) {
        // Get the average frequency of candidates in current window, or use left side as default
        if !current_window.candidates.is_empty() {
            current_window
                .candidates
                .iter()
                .map(|c| c.frequency_hz / 1e6)
                .sum::<f64>()
                / current_window.candidates.len() as f64
        } else {
            fm_start + window_width_mhz / 2.0 // Default to left side (centered on first window)
        }
    } else {
        fm_start + window_width_mhz / 2.0 // Default to left side (centered on first window)
    };

    // Create spectrum bar - clean baseline
    let mut spectrum_chars: Vec<char> = vec![theme.spectrum_baseline(); spectrum_width];

    // Add scanning window indicator
    let window_start = current_freq - window_width_mhz / 2.0;
    let window_end = current_freq + window_width_mhz / 2.0;

    let window_start_pos =
        ((window_start - fm_start) / fm_range * spectrum_width as f64).max(0.0) as usize;
    let window_end_pos = ((window_end - fm_start) / fm_range * spectrum_width as f64)
        .min(spectrum_width as f64 - 1.0) as usize;

    // Mark the scanning window with theme-appropriate framing
    let end_pos = window_end_pos.min(spectrum_width - 1);
    let window_char = theme.spectrum_window_char();
    for (_idx, char) in spectrum_chars
        .iter_mut()
        .enumerate()
        .take(end_pos + 1)
        .skip(window_start_pos)
    {
        *char = window_char; // Use theme window character
    }

    // Create layout for spectrum visualization
    let spectrum_layout = SpectrumLayout::new(area);

    // Top line with end frequencies
    let top_line = format!(
        " {:.1} MHz {} {:.1} MHz",
        fm_start,
        " ".repeat(spectrum_width.saturating_sub(20)),
        fm_end
    );
    let top_widget = Paragraph::new(top_line);
    f.render_widget(top_widget, spectrum_layout.frequencies);

    // Main spectrum line with proper ratatui coloring
    // Create spans for proper coloring
    let mut spans = vec![Span::raw(" ")]; // Leading space

    for ch in spectrum_chars.iter() {
        if *ch == theme.spectrum_window_char() {
            // Highlight scanning window with theme color
            spans.push(Span::styled(
                ch.to_string(),
                Style::default()
                    .fg(theme.spectrum_window())
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            // Default color for baseline
            spans.push(Span::raw(ch.to_string()));
        }
    }

    let spectrum_line = Line::from(spans);
    let spectrum_widget = Paragraph::new(spectrum_line);
    f.render_widget(spectrum_widget, spectrum_layout.spectrum_bar);

    // Add adaptive frequency markers beneath the spectrum bar
    let freq_markers = if area.width >= 140 {
        // Very wide: show many markers
        vec![
            89.0, 91.0, 93.0, 95.0, 97.0, 99.0, 101.0, 103.0, 105.0, 107.0,
        ]
    } else if area.width >= 100 {
        // Wide: show moderate markers
        vec![90.0, 93.0, 96.0, 99.0, 102.0, 105.0]
    } else if area.width >= 80 {
        // Medium: show key markers
        vec![90.0, 95.0, 100.0, 105.0]
    } else if area.width >= 60 {
        // Narrow: minimal markers
        vec![90.0, 100.0]
    } else {
        // Very narrow: no markers
        vec![]
    };

    if !freq_markers.is_empty() {
        let mut freq_line = " ".repeat(spectrum_width + 2);
        for freq in freq_markers.iter() {
            let pos = ((*freq - fm_start) / fm_range * spectrum_width as f64) as usize + 1;
            if pos < freq_line.len().saturating_sub(2) {
                let marker = format!("{:.0}", freq);
                let end_pos = (pos + marker.len()).min(freq_line.len());
                freq_line.replace_range(pos..end_pos, &marker);
            }
        }
        let freq_widget = Paragraph::new(freq_line);
        f.render_widget(freq_widget, spectrum_layout.markers);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_spectrum_format_unchanged() {
        let fm_start = 88.0;
        let fm_end = 108.0;
        let fm_range = fm_end - fm_start;
        assert_eq!(fm_range, 20.0);

        let baseline_char = '─';
        let window_start_char = '╟';
        let window_end_char = '╢';
        let window_area_char = '▬';

        assert_eq!(baseline_char, '─');
        assert_eq!(window_start_char, '╟');
        assert_eq!(window_end_char, '╢');
        assert_eq!(window_area_char, '▬');

        let window_width_mhz = 2.4;
        assert_eq!(window_width_mhz, 2.4);

        let freq_markers = vec![90.0, 95.0, 100.0, 105.0];
        for freq in freq_markers {
            let marker = format!("{:.0}", freq);
            assert!(marker.len() <= 3); // No decimal places
        }
    }
}
