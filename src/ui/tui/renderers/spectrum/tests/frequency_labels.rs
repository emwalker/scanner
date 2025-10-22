use super::MockTheme;
use crate::ui::tui::renderers::spectrum::frequency_labels::{
    render_frequency_labels, render_window_frequency_labels,
};

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
