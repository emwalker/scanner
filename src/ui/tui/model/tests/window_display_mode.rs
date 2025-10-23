//! Regression tests for window display mode logic
//!
//! These tests verify the behavior of when rejected candidates should be shown/hidden
//! in different scenarios. They are designed to be resilient to refactoring by testing
//! behavior rather than implementation details.

use super::helpers::ModelTestContext;
use crate::audio::quality::AudioQuality;
use crate::ecs::CandidateState;
use crate::ui::tui::model::types::{CandidateStatus, WindowDisplayMode};

/// Test that selected windows always show all candidates including rejected
#[test]
fn test_selected_window_shows_rejected_candidates() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with mix of signal and rejected candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Completed,
        None,
        None,
    );

    ctx.sync();

    // Mark window complete
    if let Some(window) = ctx.model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    let window = ctx.model.windows.get(&window_id).unwrap();

    // When window has selected candidate, should show all candidates
    let mode = window.display_mode(false, true);
    assert_eq!(mode, WindowDisplayMode::ShowAll);

    let displayable = window.displayable_candidates(false, true);
    assert_eq!(displayable.len(), 2, "Should show both signal and rejected");

    let rejected_count = displayable
        .iter()
        .filter(|c| c.status == CandidateStatus::Rejected)
        .count();
    assert_eq!(rejected_count, 1, "Should include the rejected candidate");
}

/// Test that complete non-selected windows hide rejected candidates
#[test]
fn test_complete_window_hides_rejected_candidates() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with mix of signal and rejected candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Completed,
        None,
        None,
    );

    ctx.sync();

    // Mark window complete
    if let Some(window) = ctx.model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    let window = ctx.model.windows.get(&window_id).unwrap();

    // When window is complete and not selected, should hide rejected
    let mode = window.display_mode(false, false);
    assert_eq!(mode, WindowDisplayMode::HideRejected);

    let displayable = window.displayable_candidates(false, false);
    assert_eq!(
        displayable.len(),
        1,
        "Should show only the signal candidate"
    );

    for candidate in displayable {
        assert_ne!(
            candidate.status,
            CandidateStatus::Rejected,
            "Should not show rejected candidates"
        );
    }
}

/// Test that actively scanning window shows all candidates including rejected
#[test]
fn test_current_scanning_window_shows_rejected_candidates() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with incomplete candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Analyzing,
        None,
        None,
    );

    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();

    // Verify window has incomplete candidates
    let has_incomplete = window.candidates.iter().any(|c| c.completion < 1.0);
    assert!(
        has_incomplete,
        "Test setup: window should have incomplete candidates"
    );

    // When window is current and actively scanning, should show all
    let mode = window.display_mode(true, false);
    assert_eq!(mode, WindowDisplayMode::ShowAll);

    // Add a rejected candidate
    ctx.update_candidate(
        88_500_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_500_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    let displayable = window.displayable_candidates(true, false);

    let rejected_count = displayable
        .iter()
        .filter(|c| c.status == CandidateStatus::Rejected)
        .count();
    assert!(
        rejected_count > 0,
        "Should show rejected candidates while scanning"
    );
}

/// Test that old window with all candidates complete hides rejected
#[test]
fn test_old_complete_window_hides_rejected_candidates() {
    let mut ctx = ModelTestContext::new();
    let old_window_id = 1;
    let current_window_id = 2;

    // Create old window with all candidates complete
    ctx.update_candidate(
        88_100_000.0,
        old_window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        old_window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        old_window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        old_window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(
        88_300_000.0,
        old_window_id,
        CandidateState::Completed,
        None,
        None,
    );

    // Create new current window
    ctx.update_candidate(
        89_100_000.0,
        current_window_id,
        CandidateState::Detected,
        None,
        None,
    );

    ctx.sync();

    ctx.model.current_window = current_window_id;

    // Mark old window as complete
    if let Some(window) = ctx.model.windows.get_mut(&old_window_id) {
        window.is_complete = true;
    }

    let old_window = ctx.model.windows.get(&old_window_id).unwrap();

    // Old window is not current and not selected
    let mode = old_window.display_mode(false, false);
    assert_eq!(mode, WindowDisplayMode::HideRejected);

    let displayable = old_window.displayable_candidates(false, false);
    assert_eq!(
        displayable.len(),
        1,
        "Old complete window should hide rejected candidates"
    );

    for candidate in displayable {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}

/// Test that window with only rejected candidates is hidden
#[test]
fn test_window_with_only_rejected_is_hidden() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with only rejected candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.sync();

    // Mark window as complete
    if let Some(window) = ctx.model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    let window = ctx.model.windows.get(&window_id).unwrap();

    // Window with only rejected candidates should not be displayed
    assert!(
        !window.should_display(false, false),
        "Window with only rejected candidates should be hidden"
    );

    let displayable = window.displayable_candidates(false, false);
    assert_eq!(
        displayable.len(),
        0,
        "Should have no displayable candidates"
    );
}

/// Test that window with only rejected candidates is shown if selected
#[test]
fn test_window_with_only_rejected_shown_if_selected() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with only rejected candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.sync();

    // Mark window as complete
    if let Some(window) = ctx.model.windows.get_mut(&window_id) {
        window.is_complete = true;
    }

    let window = ctx.model.windows.get(&window_id).unwrap();

    // Window with only rejected candidates should be shown if it has the selection
    assert!(
        window.should_display(false, true),
        "Window with only rejected candidates should be shown if selected"
    );

    let displayable = window.displayable_candidates(false, true);
    assert_eq!(
        displayable.len(),
        2,
        "Should show all rejected candidates when selected"
    );
}

/// Test transition: window stops showing rejected when all candidates complete
#[test]
fn test_rejected_hidden_when_scanning_completes() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    // Create window with incomplete candidates
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Analyzing,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    ctx.sync();

    ctx.model.current_window = window_id;

    let window = ctx.model.windows.get(&window_id).unwrap();

    // While scanning (incomplete), should show all
    let mode_scanning = window.display_mode(true, false);
    assert_eq!(mode_scanning, WindowDisplayMode::ShowAll);

    let displayable_scanning = window.displayable_candidates(true, false);
    assert_eq!(
        displayable_scanning.len(),
        2,
        "Should show both while scanning"
    );

    // Complete the analyzing candidate
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Signal,
        Some(AudioQuality::Good),
        Some(50.0),
    );
    ctx.update_candidate(
        88_100_000.0,
        window_id,
        CandidateState::Completed,
        None,
        None,
    );
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();

    // Verify all candidates are now complete
    let all_complete = window.candidates.iter().all(|c| c.completion >= 1.0);
    assert!(all_complete, "All candidates should be complete");

    // After completion, should hide rejected even if still current window
    let mode_complete = window.display_mode(true, false);
    assert_eq!(mode_complete, WindowDisplayMode::HideRejected);

    let displayable_complete = window.displayable_candidates(true, false);
    assert_eq!(
        displayable_complete.len(),
        1,
        "Should hide rejected after completion"
    );

    for candidate in displayable_complete {
        assert_ne!(candidate.status, CandidateStatus::Rejected);
    }
}

/// Test that non-current window with incomplete candidates hides rejected
#[test]
fn test_old_incomplete_window_hides_rejected() {
    let mut ctx = ModelTestContext::new();
    let old_window_id = 1;
    let current_window_id = 2;

    // Create old window with incomplete candidates (edge case: shouldn't normally happen)
    ctx.update_candidate(
        88_100_000.0,
        old_window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_100_000.0,
        old_window_id,
        CandidateState::Analyzing,
        None,
        None,
    );

    ctx.update_candidate(
        88_300_000.0,
        old_window_id,
        CandidateState::Detected,
        None,
        None,
    );
    ctx.update_candidate(
        88_300_000.0,
        old_window_id,
        CandidateState::Rejected,
        None,
        None,
    );

    // Create current window
    ctx.update_candidate(
        89_100_000.0,
        current_window_id,
        CandidateState::Detected,
        None,
        None,
    );

    ctx.sync();

    ctx.model.current_window = current_window_id;

    let old_window = ctx.model.windows.get(&old_window_id).unwrap();

    // Old window with incomplete candidates should still hide rejected if not current
    let mode = old_window.display_mode(false, false);
    assert_eq!(
        mode,
        WindowDisplayMode::HideRejected,
        "Non-current window should hide rejected even if incomplete"
    );

    let displayable = old_window.displayable_candidates(false, false);

    for candidate in displayable {
        assert_ne!(
            candidate.status,
            CandidateStatus::Rejected,
            "Old window should not show rejected candidates"
        );
    }
}
