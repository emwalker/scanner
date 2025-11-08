//! Isolated tests for signals table filtering logic
//! These tests verify the fixes for the two issues we resolved

use std::time::Instant;

use scanner::{
    audio::quality::AudioQuality,
    core::signals::ModulationType,
    ecs::components::signal::SignalId,
    ui::tui::model::{
        state::Model,
        types::{AnalysisStatus, PlaybackState, SignalProgress, WindowProgress},
    },
};

/// Test: build_signal_rows() includes all signals (for scan progress)
#[test]
fn test_scan_progress_shows_all_signals() {
    let mut model = Model::new();

    // Add mixed signal types like what would appear in scan progress
    let signals = vec![
        create_test_signal(88.1e6, AnalysisStatus::Signal), // Confirmed
        create_test_signal(88.3e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.5e6, AnalysisStatus::Analyzing), // In progress
        create_test_signal(88.7e6, AnalysisStatus::Detected), // Detected
        create_test_signal(88.9e6, AnalysisStatus::Error),  // Error
    ];

    add_signals_to_model(&mut model, signals);

    // Verify scan progress sees all signals
    let all_signals = model.build_signal_rows();
    assert_eq!(
        all_signals.len(),
        5,
        "Scan progress should show ALL signals including skipped"
    );

    // Verify we have the expected mix of statuses
    let statuses: Vec<_> = all_signals.iter().map(|s| &s.status).collect();
    assert!(
        statuses.contains(&&AnalysisStatus::Signal),
        "Should include confirmed signals"
    );
    assert!(
        statuses.contains(&&AnalysisStatus::Rejected),
        "Should include rejected/skipped signals"
    );
    assert!(
        statuses.contains(&&AnalysisStatus::Analyzing),
        "Should include analyzing signals"
    );
}

/// Test: build_confirmed_signal_rows() only includes confirmed signals (for signals table)
#[test]
fn test_signals_table_shows_only_confirmed() {
    let mut model = Model::new();

    // Same mixed signals as above
    let signals = vec![
        create_test_signal(88.1e6, AnalysisStatus::Signal), // Should appear in signals table
        create_test_signal(88.3e6, AnalysisStatus::Rejected), // Should NOT appear
        create_test_signal(88.5e6, AnalysisStatus::Analyzing), // Should NOT appear
        create_test_signal(88.7e6, AnalysisStatus::Detected), // Should NOT appear
        create_test_signal(88.9e6, AnalysisStatus::Signal), // Should appear in signals table
    ];

    add_signals_to_model(&mut model, signals);

    // Verify signals table only sees confirmed signals
    let confirmed_signals = model.build_confirmed_signal_rows();
    assert_eq!(
        confirmed_signals.len(),
        2,
        "Signals table should only show confirmed signals"
    );

    // Verify all returned signals are confirmed
    for signal in &confirmed_signals {
        assert_eq!(
            signal.status,
            AnalysisStatus::Signal,
            "All signals in table should be confirmed"
        );
    }

    // Verify specific frequencies are correct
    let frequencies: Vec<_> = confirmed_signals.iter().map(|s| s.frequency_hz).collect();
    assert!(
        frequencies.contains(&88.1e6),
        "Should include first confirmed signal"
    );
    assert!(
        frequencies.contains(&88.9e6),
        "Should include second confirmed signal"
    );
}

/// Test the core regression: signals table and scan progress show different data
#[test]
fn test_signals_table_scan_progress_separation() {
    let mut model = Model::new();

    // Create scenario like in the screenshot:
    // - Multiple skipped signals (should appear in scan progress but not signals table)
    // - One confirmed signal (should appear in both)
    let signals = vec![
        create_test_signal(87.1e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.5e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.7e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.9e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.1e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.3e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.9e6, AnalysisStatus::Signal),   // Confirmed (like in screenshot)
    ];

    add_signals_to_model(&mut model, signals);

    let all_signals = model.build_signal_rows(); // Used by scan progress
    let confirmed_signals = model.build_confirmed_signal_rows(); // Used by signals table

    // Core assertion: different counts
    assert_eq!(
        all_signals.len(),
        7,
        "Scan progress should show all 7 signals"
    );
    assert_eq!(
        confirmed_signals.len(),
        1,
        "Signals table should show only 1 confirmed signal"
    );

    // Verify the confirmed signal details
    assert_eq!(
        confirmed_signals[0].frequency_hz, 88.9e6,
        "Confirmed signal should be 88.9 MHz"
    );
    assert_eq!(confirmed_signals[0].status, AnalysisStatus::Signal);
}

/// Property test: confirmed signals are always a subset of all signals
#[cfg(test)]
#[test]
fn test_confirmed_signals_subset_property() {
    let mut model = Model::new();

    // Test with various combinations
    let test_cases = vec![
        // Case 1: All confirmed
        vec![
            create_test_signal(88.1e6, AnalysisStatus::Signal),
            create_test_signal(88.5e6, AnalysisStatus::Signal),
        ],
        // Case 2: None confirmed
        vec![
            create_test_signal(88.1e6, AnalysisStatus::Rejected),
            create_test_signal(88.5e6, AnalysisStatus::Analyzing),
        ],
        // Case 3: Mixed (realistic scenario)
        vec![
            create_test_signal(88.1e6, AnalysisStatus::Signal),
            create_test_signal(88.3e6, AnalysisStatus::Rejected),
            create_test_signal(88.5e6, AnalysisStatus::Rejected),
            create_test_signal(88.7e6, AnalysisStatus::Analyzing),
            create_test_signal(88.9e6, AnalysisStatus::Signal),
        ],
    ];

    for (i, signals) in test_cases.into_iter().enumerate() {
        model.windows.clear(); // Reset for each test case
        add_signals_to_model(&mut model, signals);

        let all_signals = model.build_signal_rows();
        let confirmed_signals = model.build_confirmed_signal_rows();

        // Property 1: confirmed count <= all count
        assert!(
            confirmed_signals.len() <= all_signals.len(),
            "Test case {}: Confirmed signals must be subset of all signals",
            i
        );

        // Property 2: all confirmed signal frequencies exist in all signals
        let all_frequencies: Vec<_> = all_signals.iter().map(|s| s.frequency_hz).collect();
        for confirmed in &confirmed_signals {
            assert!(
                all_frequencies.contains(&confirmed.frequency_hz),
                "Test case {}: Confirmed frequency {} must exist in all signals",
                i,
                confirmed.frequency_hz
            );
        }

        // Property 3: all confirmed signals have Signal status
        for signal in &confirmed_signals {
            assert_eq!(
                signal.status,
                AnalysisStatus::Signal,
                "Test case {}: All confirmed signals must have Signal status",
                i
            );
        }
    }
}

/// Regression test: Ensure signals table and scan progress remain separate after fix
/// This test documents the exact bug that was reported and fixed
#[test]
fn test_regression_signals_table_not_affecting_scan_progress() {
    let mut model = Model::new();

    // Recreate scenario from the original bug report:
    // User saw signals disappear from scan progress when we filtered signals table
    let signals = vec![
        // Multiple "Skipped" signals (as seen in screenshot)
        create_test_signal(87.100e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.500e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.700e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(87.900e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.100e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.300e6, AnalysisStatus::Rejected), // Skipped
        create_test_signal(88.500e6, AnalysisStatus::Rejected), // Skipped
        // One confirmed signal (as seen in screenshot)
        create_test_signal(88.900e6, AnalysisStatus::Signal), // Signal (Moderate)
    ];

    add_signals_to_model(&mut model, signals);

    // BEFORE the bug fix:
    // - build_signal_rows() was modified to filter, breaking scan progress
    // - Both scan progress and signals table would show only 1 signal

    // AFTER the bug fix:
    // - build_signal_rows() shows all signals (scan progress works correctly)
    // - build_confirmed_signal_rows() shows only confirmed signals (signals table works correctly)

    let scan_progress_signals = model.build_signal_rows(); // Used by scan progress
    let signals_table_signals = model.build_confirmed_signal_rows(); // Used by signals table

    // Critical regression test: these must be different!
    assert_eq!(
        scan_progress_signals.len(),
        8,
        "Regression check: Scan progress must show ALL signals including skipped"
    );

    assert_eq!(
        signals_table_signals.len(),
        1,
        "Regression check: Signals table must show ONLY confirmed signals"
    );

    // Verify scan progress includes skipped signals
    let rejected_count = scan_progress_signals
        .iter()
        .filter(|s| s.status == AnalysisStatus::Rejected)
        .count();
    assert_eq!(
        rejected_count, 7,
        "Scan progress should include all 7 skipped signals"
    );

    // Verify signals table includes only confirmed signals
    let confirmed_count = signals_table_signals
        .iter()
        .filter(|s| s.status == AnalysisStatus::Signal)
        .count();
    assert_eq!(
        confirmed_count, 1,
        "Signals table should include only 1 confirmed signal"
    );
    assert_eq!(
        signals_table_signals[0].frequency_hz, 88.900e6,
        "Confirmed signal should be 88.9 MHz"
    );

    // This test ensures we fixed both issues:
    // 1. Signals table now shows confirmed signals only
    // 2. Scan progress still shows all signals (no regression)
}

// Helper functions

fn create_test_signal(frequency: f64, status: AnalysisStatus) -> SignalProgress {
    let signal_id = SignalId::new(frequency, ModulationType::WFM);

    SignalProgress {
        signal_id,
        frequency_hz: frequency,
        window_id: 0,
        center_frequency_hz: frequency,
        completion: 1.0,
        status,
        playback_state: PlaybackState::NotPlaying,
        audio_quality: Some(AudioQuality::Good),
        signal_strength: Some(0.8),
        last_update: Instant::now(),
        notes: None,
    }
}

fn add_signals_to_model(model: &mut Model, signals: Vec<SignalProgress>) {
    let mut window_progress = WindowProgress {
        window_id: 0,
        signals,
        is_complete: false,
        signal_lookup: std::collections::HashMap::new(),
    };

    // Setup signal lookup
    for (i, signal) in window_progress.signals.iter().enumerate() {
        window_progress
            .signal_lookup
            .insert(signal.signal_id.clone(), i);
    }

    model.windows.insert(0, window_progress);
}
