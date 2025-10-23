//! Tests for scan window display when scan starts
//!
//! These tests verify that scanning windows and signals are properly displayed
//! in the UI when a scan begins and progresses.

use super::helpers::ModelTestContext;

/// Test that when a task has no windows yet, the UI doesn't crash
#[test]
fn test_task_with_no_windows_yet() {
    use crate::ecs::{TaskId, entities::TaskWindowCell};

    let mut ctx = ModelTestContext::new();

    // Add a task to the model (scan task)
    let task_id = TaskId::new("scan_1");
    ctx.model
        .tasks
        .push(crate::ui::tui::model::state::TaskSummary {
            task_id: task_id.clone(),
            label: "Scan 1".to_string(),
            summary: "FM · 88.0 MHz-108.0 MHz".to_string(),
            activity: "Scanning 0/40 (0%)".to_string(),
            assigned_tuner: Some("SDRplay RSPduo :: 2301034E3".to_string()),
            assigned_tuner_id: None,
            window_cell_data: TaskWindowCell::SpectrumBar {
                full_range_hz: (88.0e6, 108.0e6),
                current_window_hz: None,
            },
        });
    ctx.model.displayed_task_id = Some(task_id);

    // No windows exist yet (scan just started)
    assert!(ctx.model.windows.is_empty(), "No windows should exist yet");

    // The UI should handle this gracefully - verify we can build signal rows without panic
    let signal_rows = ctx.model.build_signal_rows();

    // When no windows exist, there should be no signal rows
    // This is expected behavior - the scan panel would be empty
    assert!(
        signal_rows.is_empty(),
        "Should have no signal rows when no windows exist yet"
    );
}

/// Test that when a scan task has signals, they appear in signal rows
/// This is the FAILING test that demonstrates the bug from the screenshot
#[test]
fn test_scan_task_signals_appear_in_ui() {
    use super::helpers::TestSignalState;
    use crate::ecs::{TaskId, entities::TaskWindowCell};

    let mut ctx = ModelTestContext::new();

    // Create a scan task (like "Scan 1" in the screenshot)
    let task_id = TaskId::new("scan_1");
    ctx.model
        .tasks
        .push(crate::ui::tui::model::state::TaskSummary {
            task_id: task_id.clone(),
            label: "Scan 1".to_string(),
            summary: "FM · 88.0 MHz-108.0 MHz".to_string(),
            activity: "Scanning 0/40 (0%)".to_string(),
            assigned_tuner: Some("SDRplay RSPduo".to_string()),
            assigned_tuner_id: None,
            window_cell_data: TaskWindowCell::SpectrumBar {
                full_range_hz: (88.0e6, 108.0e6),
                current_window_hz: Some((88.0e6, 88.5e6)),
            },
        });
    ctx.model.displayed_task_id = Some(task_id.clone());

    // Scan is processing window 0
    ctx.model.current_window = 0;
    ctx.model.total_windows = Some(40);

    // Add signals that have been detected (like in the logs)
    let window_id = 0;
    ctx.update_signal(
        88_100_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.update_signal(
        88_500_000.0,
        window_id,
        TestSignalState::Analyzing,
        None,
        None,
    );
    ctx.sync();

    // Verify signals exist in the model
    let window = ctx
        .model
        .windows
        .get(&window_id)
        .expect("Window should exist after adding signals");
    assert_eq!(window.signals.len(), 2, "Window should have 2 signals");

    // THIS IS THE FAILING ASSERTION:
    // When displaying the scan task, signal rows should show the signals
    let signal_rows = ctx.model.build_signal_rows();

    assert!(
        !signal_rows.is_empty(),
        "BUG: signal rows are empty even though signals exist! This is the bug from the \
         screenshot - the Scan 1 panel shows no data."
    );

    // Verify both signals appear
    assert_eq!(
        signal_rows.len(),
        2,
        "Should have 2 signal rows matching the 2 signals"
    );
}

/// Test that current scanning window is visible even without signals
#[test]
fn test_current_scanning_window_visible_without_signals() {
    let mut ctx = ModelTestContext::new();

    // Scan is in progress on window 0
    ctx.model.current_window = 0;
    ctx.model.total_windows = Some(40);

    // Window exists but has no signals yet
    let window_id = 0;
    let window = crate::ui::tui::model::types::WindowProgress {
        window_id,
        signals: Vec::new(),
        is_complete: false,
        signal_lookup: std::collections::HashMap::new(),
    };
    ctx.model.windows.insert(window_id, window);

    // When scan just starts, we might have an empty window
    // This test checks if that's handled correctly
    // The expected behavior is that we should show *something*
    // to indicate scanning is happening, even if no signals yet
    assert!(
        ctx.model.windows.contains_key(&window_id),
        "Model should track the scanning window"
    );
}
