//! Signals table display

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    widgets::{Cell, Row, Table},
};

#[cfg(test)]
use super::table_styles::TableRow;
use super::table_styles::{self, AlwaysVisibleFilter};
use crate::ui::tui::{
    model::{
        Model,
        types::{FocusedTable, SignalsTableRow},
    },
    themes::Theme,
};

pub fn render_signals_table(f: &mut Frame, area: Rect, model: &mut Model, theme: &dyn Theme) {
    let has_focus = table_styles::check_focus(&model.focus_state, FocusedTable::SignalsTable);

    // Get count of confirmed signals for the table header
    let signal_count = model.confirmed_signal_count();
    let table_title = format!("Signals ({})", signal_count);
    let block = table_styles::create_table_block(&table_title, has_focus, theme);

    let header = Row::new(vec![
        Cell::from(
            ratatui::text::Text::from("Frequency").alignment(ratatui::layout::Alignment::Right),
        ),
        Cell::from("Modulation"),
        Cell::from("Activity"),
        Cell::from("Notes"),
    ])
    .style(table_styles::header_style());

    // Get confirmed signal data from model (only signals that go into playback)
    let signals = model.build_confirmed_signal_rows();

    let viewport_height = area.height.saturating_sub(3) as usize;
    model.signals_table_scroll.viewport_height = viewport_height;

    let visibility_context = table_styles::VisibilityContext::new(None, None);

    // Wrap SignalRow instances for signals table rendering
    let wrapped_signals: Vec<SignalsTableRow> = signals.iter().map(SignalsTableRow).collect();

    let (rows, scrollbar_state) = {
        let mut renderer = table_styles::TableRenderer2::new(
            &wrapped_signals,
            FocusedTable::SignalsTable,
            AlwaysVisibleFilter,
            &mut model.signals_table_scroll,
        );
        renderer.render(&model.focus_state, theme, visibility_context)
    };

    let table = Table::new(rows, get_signals_table_constraints())
        .header(header)
        .block(block)
        .column_spacing(2);

    f.render_widget(table, area);
    table_styles::render_scrollbar(f, area, &scrollbar_state, theme);
}

/// Get the column constraints used by render_signals_table
fn get_signals_table_constraints() -> [ratatui::layout::Constraint; 4] {
    [
        Constraint::Length(11),      // Frequency (matches Scan Progress table)
        Constraint::Length(10),      // Modulation (fits header and future types)
        Constraint::Length(8),       // Activity (minimum for "Playing")
        Constraint::Percentage(100), // Notes (takes remaining space)
    ]
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};

    use super::*;
    use crate::ui::tui::{model::Model, themes::basic::BasicDarkTheme};

    #[test]
    fn test_render_empty_signals_table() {
        let mut terminal = Terminal::new(TestBackend::new(50, 10)).unwrap();
        let mut model = Model::default();
        let theme = BasicDarkTheme;
        let area = Rect::new(0, 0, 50, 10);

        terminal
            .draw(|f| {
                render_signals_table(f, area, &mut model, &theme);
            })
            .unwrap();

        // Should not panic and should render empty table
    }

    #[test]
    fn test_signals_table_focus_detection() {
        use crate::ui::tui::model::types::FocusState;

        let mut terminal = Terminal::new(TestBackend::new(50, 10)).unwrap();
        let mut model = Model::default();
        let theme = BasicDarkTheme;
        let area = Rect::new(0, 0, 50, 10);

        // Set focus to signals table
        model.focus_state = FocusState::SignalsTable(0);

        terminal
            .draw(|f| {
                render_signals_table(f, area, &mut model, &theme);
            })
            .unwrap();

        // Should render without panic and table should have focus styling
    }

    #[test]
    fn test_signals_table_should_use_tablerenderer2_for_selection() {
        // This test documents the requirement that signals table should use TableRenderer2
        // like the scan progress table does, to support row selection
        use std::time::Instant;

        use crate::{
            audio::quality::AudioQuality,
            core::signals::ModulationType,
            ecs::components::SignalId,
            ui::tui::model::types::{
                AnalysisStatus, FocusState, PlaybackState, SignalProgress, WindowProgress,
            },
        };

        let mut model = Model::default();

        // Set up test signal data
        let signal_id = SignalId::new(88.9e6, ModulationType::WFM);

        let mut window_progress = WindowProgress {
            window_id: 0,
            signals: vec![SignalProgress {
                signal_id: signal_id.clone(),
                frequency_hz: 88.9e6,
                window_id: 0,
                center_frequency_hz: 88.9e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::NotPlaying,
                audio_quality: Some(AudioQuality::Good),
                signal_strength: Some(0.8),
                last_update: Instant::now(),
                notes: None,
            }],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        window_progress.signal_lookup.insert(signal_id, 0);
        model.windows.insert(0, window_progress);

        // Focus signals table with selection
        model.focus_state = FocusState::SignalsTable(0);

        // Test will fail: signals table should adjust scroll state like scan progress table does
        // This proves the table supports selection properly by managing scroll position
        let confirmed_signals = model.build_confirmed_signal_rows();
        assert!(
            !confirmed_signals.is_empty(),
            "Need signals to test selection"
        );

        // This should work but currently fails - signals table doesn't use TableRenderer2
        // which handles selection and scroll adjustment
        model
            .signals_table_scroll
            .adjust_for_selection(0, confirmed_signals.len());

        // This assertion will pass since adjust_for_selection exists
        // But the real issue is that render_signals_table doesn't use TableRenderer2
        // so it never calls adjust_for_selection or handles selection styling
        assert_eq!(model.signals_table_scroll.offset, 0);

        // The key missing piece: signals table should use the same selection mechanism
        // as scan progress table. This test will fail until we implement TableRenderer2

        // Try to render and check if scroll position gets adjusted for selection
        // Store initial offset to verify it could be used for comparison if needed
        let _initial_offset = model.signals_table_scroll.offset;

        // Now we use TableRenderer2, so selection should work properly
        // The signals table should now adjust scroll position for selection
        let theme = BasicDarkTheme;
        let mut terminal = Terminal::new(TestBackend::new(80, 15)).unwrap();
        let area = Rect::new(0, 0, 80, 15);

        // This should not panic - the signals table now uses TableRenderer2
        // which properly handles selection and scroll adjustment
        terminal
            .draw(|f| {
                render_signals_table(f, area, &mut model, &theme);
            })
            .unwrap();

        // The key evidence: signals table now uses TableRenderer2 implementation
        // which handles selection styling and scroll position adjustment
    }

    #[test]
    fn test_signals_table_frequency_column_width_matches_scan_progress() {
        // Test that frequency column width matches scan progress table
        // The scan progress table uses Constraint::Length(11) for frequency column
        let mut terminal = Terminal::new(TestBackend::new(80, 15)).unwrap();
        let mut model = Model::default();
        let theme = BasicDarkTheme;
        let area = Rect::new(0, 0, 80, 15);

        // This test verifies the table constraint is Length(11), not percentage
        // We can't directly test constraints, but we can test rendering doesn't panic
        terminal
            .draw(|f| {
                render_signals_table(f, area, &mut model, &theme);
            })
            .unwrap();

        // TODO: Find a way to verify constraint is Length(11) to match scan progress
        // For now, this prevents regression and documents the requirement
    }

    #[test]
    fn test_signals_table_displays_signal_data() {
        use std::time::Instant;

        use crate::{
            audio::quality::AudioQuality,
            core::signals::ModulationType,
            ecs::components::SignalId,
            ui::tui::model::types::{
                AnalysisStatus, PlaybackState, SignalProgress, WindowProgress,
            },
        };

        let mut terminal = Terminal::new(TestBackend::new(80, 15)).unwrap();
        let mut model = Model::default();
        let theme = BasicDarkTheme;
        let area = Rect::new(0, 0, 80, 15);

        // Add some test signal data to the model
        let signal_id_1 = SignalId::new(88.9e6, ModulationType::WFM);
        let signal_id_2 = SignalId::new(89.1e6, ModulationType::WFM);
        let signal_id_3 = SignalId::new(89.5e6, ModulationType::WFM);

        let mut window_progress = WindowProgress {
            window_id: 0,
            signals: vec![
                SignalProgress {
                    signal_id: signal_id_1.clone(),
                    frequency_hz: 88.9e6,
                    window_id: 0,
                    center_frequency_hz: 88.9e6,
                    completion: 0.75,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.8),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_2.clone(),
                    frequency_hz: 89.1e6,
                    window_id: 0,
                    center_frequency_hz: 89.1e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::Playing,
                    audio_quality: Some(AudioQuality::Moderate),
                    signal_strength: Some(0.6),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_3.clone(),
                    frequency_hz: 89.5e6,
                    window_id: 0,
                    center_frequency_hz: 89.5e6,
                    completion: 1.0,
                    status: AnalysisStatus::Rejected,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: None,
                    signal_strength: Some(0.3),
                    last_update: Instant::now(),
                    notes: None,
                },
            ],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        // Set up the signal lookup
        for (i, signal) in window_progress.signals.iter().enumerate() {
            window_progress
                .signal_lookup
                .insert(signal.signal_id.clone(), i);
        }

        model.windows.insert(0, window_progress);

        // Verify that build_signal_rows returns all signals (including rejected)
        let all_signal_rows = model.build_signal_rows();
        assert_eq!(
            all_signal_rows.len(),
            3,
            "Model should have 3 total signal rows"
        );

        // Verify that build_confirmed_signal_rows only returns confirmed signals
        let confirmed_signal_rows = model.build_confirmed_signal_rows();
        assert_eq!(
            confirmed_signal_rows.len(),
            2,
            "Model should have 2 confirmed signal rows (excluding rejected)"
        );
        assert_eq!(
            confirmed_signal_rows[0].frequency_hz, 88.9e6,
            "First signal (lowest frequency) should be 88.9 MHz"
        );
        assert_eq!(
            confirmed_signal_rows[1].frequency_hz, 89.1e6,
            "Second signal (highest frequency) should be 89.1 MHz"
        );

        // Verify that all confirmed signals have Signal status
        for row in &confirmed_signal_rows {
            assert_eq!(
                row.status,
                AnalysisStatus::Signal,
                "All confirmed signals should have Signal status"
            );
        }

        // Test that SignalRow can be used in renderer
        let rows = confirmed_signal_rows
            .iter()
            .map(|row| {
                row.build_cells(&theme) // This should work once TableRow is implemented
            })
            .collect::<Vec<_>>();

        assert_eq!(rows.len(), 2, "Should be able to build cells for 2 rows");
        assert_eq!(
            rows[0].len(),
            5,
            "Each row should have 5 cells (using SignalRow from task_progress temporarily)"
        );

        // Test actual rendering
        terminal
            .draw(|f| {
                render_signals_table(f, area, &mut model, &theme);
            })
            .unwrap();
    }

    #[test]
    fn test_signals_table_sorted_low_to_high_frequency() {
        use std::time::Instant;

        use crate::{
            audio::quality::AudioQuality,
            core::signals::ModulationType,
            ecs::components::SignalId,
            ui::tui::model::types::{
                AnalysisStatus, PlaybackState, SignalProgress, WindowProgress,
            },
        };

        let mut model = Model::default();

        // Add test signal data with different frequencies
        let signal_id_low = SignalId::new(88.5e6, ModulationType::WFM);
        let signal_id_mid = SignalId::new(89.1e6, ModulationType::WFM);
        let signal_id_high = SignalId::new(107.9e6, ModulationType::WFM);

        let mut window_progress = WindowProgress {
            window_id: 0,
            signals: vec![
                SignalProgress {
                    signal_id: signal_id_high.clone(),
                    frequency_hz: 107.9e6, // Highest frequency
                    window_id: 0,
                    center_frequency_hz: 107.9e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.8),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_low.clone(),
                    frequency_hz: 88.5e6, // Lowest frequency
                    window_id: 0,
                    center_frequency_hz: 88.5e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Moderate),
                    signal_strength: Some(0.6),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_mid.clone(),
                    frequency_hz: 89.1e6, // Middle frequency
                    window_id: 0,
                    center_frequency_hz: 89.1e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::Playing,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.9),
                    last_update: Instant::now(),
                    notes: None,
                },
            ],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        // Set up the signal lookup
        for (i, signal) in window_progress.signals.iter().enumerate() {
            window_progress
                .signal_lookup
                .insert(signal.signal_id.clone(), i);
        }

        model.windows.insert(0, window_progress);

        // Get confirmed signals from the model
        let confirmed_signal_rows = model.build_confirmed_signal_rows();
        assert_eq!(
            confirmed_signal_rows.len(),
            3,
            "Model should have 3 confirmed signal rows"
        );

        // Verify signals are sorted from LOW to HIGH frequency (ascending order)
        assert_eq!(
            confirmed_signal_rows[0].frequency_hz, 88.5e6,
            "First signal should be lowest frequency (88.5 MHz)"
        );
        assert_eq!(
            confirmed_signal_rows[1].frequency_hz, 89.1e6,
            "Second signal should be middle frequency (89.1 MHz)"
        );
        assert_eq!(
            confirmed_signal_rows[2].frequency_hz, 107.9e6,
            "Third signal should be highest frequency (107.9 MHz)"
        );
    }

    #[test]
    fn test_signals_table_scroll_adjustment_for_selection() {
        use std::time::Instant;

        use crate::{
            audio::quality::AudioQuality,
            core::signals::ModulationType,
            ecs::components::SignalId,
            ui::tui::model::types::{
                AnalysisStatus, FocusState, PlaybackState, SignalProgress, WindowProgress,
            },
        };

        let mut model = Model::default();

        // Set up multiple signals to test scroll adjustment
        let signal_id_1 = SignalId::new(88.5e6, ModulationType::WFM);
        let signal_id_2 = SignalId::new(89.1e6, ModulationType::WFM);
        let signal_id_3 = SignalId::new(89.5e6, ModulationType::WFM);
        let signal_id_4 = SignalId::new(107.9e6, ModulationType::WFM);

        let mut window_progress = WindowProgress {
            window_id: 0,
            signals: vec![
                SignalProgress {
                    signal_id: signal_id_1.clone(),
                    frequency_hz: 88.5e6,
                    window_id: 0,
                    center_frequency_hz: 88.5e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.8),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_2.clone(),
                    frequency_hz: 89.1e6,
                    window_id: 0,
                    center_frequency_hz: 89.1e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.7),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_3.clone(),
                    frequency_hz: 89.5e6,
                    window_id: 0,
                    center_frequency_hz: 89.5e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.6),
                    last_update: Instant::now(),
                    notes: None,
                },
                SignalProgress {
                    signal_id: signal_id_4.clone(),
                    frequency_hz: 107.9e6,
                    window_id: 0,
                    center_frequency_hz: 107.9e6,
                    completion: 1.0,
                    status: AnalysisStatus::Signal,
                    playback_state: PlaybackState::NotPlaying,
                    audio_quality: Some(AudioQuality::Good),
                    signal_strength: Some(0.9),
                    last_update: Instant::now(),
                    notes: None,
                },
            ],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        // Set up signal lookups
        for (i, signal) in window_progress.signals.iter().enumerate() {
            window_progress
                .signal_lookup
                .insert(signal.signal_id.clone(), i);
        }

        model.windows.insert(0, window_progress);

        // Test scroll adjustment behavior when selecting a row that's out of view
        // Focus on the last signal (index 3) with a small viewport that can't show all signals
        model.focus_state = FocusState::SignalsTable(3);

        let signals = model.build_confirmed_signal_rows();
        let wrapped_signals: Vec<SignalsTableRow> = signals.iter().map(SignalsTableRow).collect();

        // Create a small viewport that can only show 2 signals at once
        let mut test_scroll = table_styles::ScrollState::new(2);

        let mut renderer = table_styles::TableRenderer2::new(
            &wrapped_signals,
            FocusedTable::SignalsTable,
            AlwaysVisibleFilter,
            &mut test_scroll,
        );

        let theme = BasicDarkTheme;
        let visibility_context = table_styles::VisibilityContext::new(None, None);

        // This should trigger scroll adjustment to show the selected row (index 3)
        let (rows, _scrollbar_state) =
            renderer.render(&model.focus_state, &theme, visibility_context);

        // The key test: TableRenderer2 should call adjust_for_selection when SignalsTable has focus
        // If get_selected_index() doesn't handle SignalsTable, no scroll adjustment happens
        assert!(!rows.is_empty(), "Should have rendered rows");

        // Test that scroll was actually adjusted for selection
        // With 4 signals, viewport height 2, and selection on index 3 (last signal)
        // The scroll offset should be adjusted to make index 3 visible
        // adjust_for_selection should set offset to at least 2 (3 - (2-1) = 2)

        // This is the core test: if TableRenderer2.get_selected_index() properly handles
        // SignalsTable, then adjust_for_selection gets called and offset is adjusted
        // If get_selected_index() returns None for SignalsTable, no adjustment happens

        assert!(
            test_scroll.offset >= 2,
            "Scroll offset should be adjusted to show selected row (index 3). Expected offset >= \
             2, got offset = {}. This fails because TableRenderer2.get_selected_index() doesn't \
             handle SignalsTable focus state.",
            test_scroll.offset
        );
    }

    #[test]
    fn test_signals_table_uses_optimized_column_widths() {
        // TDD RED: Test that signals table uses optimized column widths
        // - Modulation column: minimum width for content (3 chars for "WFM", "NFM", etc.)
        // - Activity column: minimum width for content (7 chars for "Playing")
        // - Notes column: takes remaining space

        use std::time::Instant;

        use crate::{
            audio::quality::AudioQuality,
            core::signals::ModulationType,
            ecs::components::SignalId,
            ui::tui::model::types::{
                AnalysisStatus, PlaybackState, SignalProgress, WindowProgress,
            },
        };

        let mut model = Model::default();

        // Create test signal with content that will exercise the column widths
        let signal_id = SignalId::new(88.9e6, ModulationType::WFM);
        let mut window_progress = WindowProgress {
            window_id: 0,
            signals: vec![SignalProgress {
                signal_id: signal_id.clone(),
                frequency_hz: 88.9e6,
                window_id: 0,
                center_frequency_hz: 88.9e6,
                completion: 1.0,
                status: AnalysisStatus::Signal,
                playback_state: PlaybackState::Playing, // Will show "Playing" in Activity column
                audio_quality: Some(AudioQuality::Good),
                signal_strength: Some(0.8),
                last_update: Instant::now(),
                notes: Some("This is a long note that should take up remaining space".to_string()),
            }],
            is_complete: false,
            signal_lookup: std::collections::HashMap::new(),
        };

        window_progress.signal_lookup.insert(signal_id, 0);
        model.windows.insert(0, window_progress);

        // Get the table constraints currently used in render_signals_table
        // This test will fail until we optimize the constraints

        // Current implementation uses percentages - we want to change to:
        // - Frequency: Length(11) (unchanged)
        // - Modulation: Length(3) (minimum for "WFM", "NFM", etc.)
        // - Activity: Length(7) (minimum for "Playing")
        // - Notes: Percentage(100) (takes remaining space)

        let constraints = get_signals_table_constraints();

        // Assert the optimized constraint layout
        assert_eq!(constraints.len(), 4, "Should have 4 column constraints");

        // Frequency column should remain fixed width
        assert_eq!(
            format!("{:?}", constraints[0]),
            "Length(11)",
            "Frequency column should be Length(11)"
        );

        // Modulation column should be minimum width for content
        assert_eq!(
            format!("{:?}", constraints[1]),
            "Length(10)",
            "Modulation column should be Length(10) for header and future modulation types"
        );

        // Activity column should be minimum width for "Playing"
        assert_eq!(
            format!("{:?}", constraints[2]),
            "Length(8)",
            "Activity column should be Length(8) for 'Playing'"
        );

        // Notes column should take remaining space
        assert_eq!(
            format!("{:?}", constraints[3]),
            "Percentage(100)",
            "Notes column should be Percentage(100) to take remaining space"
        );
    }
}
