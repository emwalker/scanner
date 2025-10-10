use super::MockTheme;
use crate::audio::quality::AudioQuality;
use crate::ui::tui::{
    model::{CandidateProgress, CandidateStatus, Model, WindowProgress},
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
    let window = WindowProgress {
        window_id: 1,
        candidates: vec![
            CandidateProgress {
                candidate_id: "89.9-1".to_string(),
                frequency_hz: 89.9e6,
                metadata: crate::scanning::window::WindowMetadata {
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
                candidate_id: "90.5-1".to_string(),
                frequency_hz: 90.5e6,
                metadata: crate::scanning::window::WindowMetadata {
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
            candidate_id: "91.8-1".to_string(),
            frequency_hz: 91.8e6,
            metadata: crate::scanning::window::WindowMetadata {
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
                candidate_id: "89.9-1".to_string(),
                frequency_hz: 89.9e6,
                metadata: crate::scanning::window::WindowMetadata {
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
                candidate_id: "90.5-1".to_string(),
                frequency_hz: 90.5e6,
                metadata: crate::scanning::window::WindowMetadata {
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
