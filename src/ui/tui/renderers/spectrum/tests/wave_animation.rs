use super::MockTheme;
use crate::ui::tui::renderers::spectrum::wave_animation::render_full_spectrum_row;

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
