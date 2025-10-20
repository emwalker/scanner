use super::MockTheme;
use crate::audio::quality::AudioQuality;
use crate::ui::tui::{
    colors::ACTIVE_STATE_GREEN,
    model::{Model, types::SpectrumStation},
    renderers::spectrum_caladan::window_detail::render_window_detail_row,
};

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

    model.spectrum_stations = vec![
        SpectrumStation {
            frequency_hz: 89.9e6,
            signal_strength: 0.5,
            audio_quality: Some(AudioQuality::Good),
            is_active: false,
        },
        SpectrumStation {
            frequency_hz: 90.5e6,
            signal_strength: 0.4,
            audio_quality: Some(AudioQuality::Moderate),
            is_active: false,
        },
    ];

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

    model.spectrum_stations = vec![SpectrumStation {
        frequency_hz: 91.8e6,
        signal_strength: 0.5,
        audio_quality: Some(AudioQuality::Good),
        is_active: false,
    }];

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

    model.spectrum_stations = vec![
        SpectrumStation {
            frequency_hz: 89.9e6,
            signal_strength: 0.5,
            audio_quality: Some(AudioQuality::Good),
            is_active: false,
        },
        SpectrumStation {
            frequency_hz: 90.5e6,
            signal_strength: 0.1,
            audio_quality: Some(AudioQuality::NoAudio),
            is_active: false,
        },
    ];

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

#[test]
fn test_active_station_uses_active_state_green() {
    let theme = MockTheme;
    let width = 100;
    let window_start = Some(89.5e6);
    let window_width = 2.4e6;

    let mut model = Model::new();

    model.spectrum_stations = vec![SpectrumStation {
        frequency_hz: 89.9e6,
        signal_strength: 0.5,
        audio_quality: Some(AudioQuality::Good),
        is_active: true,
    }];

    let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

    let active_spans: Vec<_> = line
        .spans
        .iter()
        .filter(|span| span.content.trim() != "")
        .filter(|span| span.style.fg == Some(ACTIVE_STATE_GREEN))
        .collect();

    assert!(
        !active_spans.is_empty(),
        "Active station should use ACTIVE_STATE_GREEN color (RGB 150, 255, 150)"
    );
}

#[test]
fn test_inactive_station_does_not_use_active_state_green() {
    let theme = MockTheme;
    let width = 100;
    let window_start = Some(89.5e6);
    let window_width = 2.4e6;

    let mut model = Model::new();

    model.spectrum_stations = vec![SpectrumStation {
        frequency_hz: 89.9e6,
        signal_strength: 0.5,
        audio_quality: Some(AudioQuality::Good),
        is_active: false,
    }];

    let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

    let green_spans: Vec<_> = line
        .spans
        .iter()
        .filter(|span| span.content.trim() != "")
        .filter(|span| span.style.fg == Some(ACTIVE_STATE_GREEN))
        .collect();

    assert!(
        green_spans.is_empty(),
        "Inactive station should not use ACTIVE_STATE_GREEN color"
    );
}

#[test]
fn test_active_and_inactive_stations_different_colors() {
    let theme = MockTheme;
    let width = 100;
    let window_start = Some(89.5e6);
    let window_width = 2.4e6;

    let mut model = Model::new();

    model.spectrum_stations = vec![
        SpectrumStation {
            frequency_hz: 89.9e6,
            signal_strength: 0.5,
            audio_quality: Some(AudioQuality::Good),
            is_active: true,
        },
        SpectrumStation {
            frequency_hz: 90.5e6,
            signal_strength: 0.5,
            audio_quality: Some(AudioQuality::Good),
            is_active: false,
        },
    ];

    let line = render_window_detail_row(width, window_start, window_width, &model, &theme);

    let all_colors: Vec<_> = line
        .spans
        .iter()
        .filter(|span| span.content.trim() != "")
        .filter_map(|span| span.style.fg)
        .collect();

    assert!(
        all_colors.contains(&ACTIVE_STATE_GREEN),
        "Should contain ACTIVE_STATE_GREEN for active station"
    );

    let unique_colors: std::collections::HashSet<_> = all_colors.iter().collect();
    assert!(
        unique_colors.len() > 1,
        "Active and inactive stations should use different colors"
    );
}

#[test]
fn test_active_state_green_constant_value() {
    use ratatui::style::Color;

    assert_eq!(
        ACTIVE_STATE_GREEN,
        Color::Rgb(150, 255, 150),
        "ACTIVE_STATE_GREEN must be RGB(150, 255, 150) to match tuner Scanning/Listening color"
    );
}
