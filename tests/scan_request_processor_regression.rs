//! Test for scan request processor tune transition regression
//!
//! This test verifies that SignalEntity::request_tune_transition is called correctly
//! to fix the regression where tune state transitions were not happening.

use scanner::{
    core::types::Result,
    ecs::{
        SignalEntity, TaskId,
        components::{station::TuneState, window::WindowId},
    },
};

/// Test SignalEntity tune transition functionality directly
/// This verifies the core functionality that was missing in ScanRequestProcessor
#[test]
fn test_signal_entity_tune_transition_works() -> Result<()> {
    let task_id = TaskId::new("test-scan");
    let window_id = WindowId::new(task_id.clone(), 1);
    let mut signal = SignalEntity::new(88.9e6, window_id.clone());

    // Confirm the signal so it can be tuned
    signal
        .analysis
        .confirm_analysis(scanner::audio::quality::AudioQuality::Good, 0.8);

    // Initial state: signal should be Idle
    assert!(matches!(signal.tune_state, TuneState::Idle));

    // Call request_tune_transition - this is what ScanRequestProcessor should do
    let result = signal.request_tune_transition(window_id.clone(), 88.9e6);

    // Should succeed
    assert!(
        result.is_ok(),
        "Tune transition should succeed for confirmed signal"
    );

    // Verify: Signal should now be in Transitioning state
    match &signal.tune_state {
        TuneState::Transitioning(transition) => {
            assert_eq!(transition.window_id, window_id);
            assert_eq!(transition.center_frequency, 88.9e6);
        }
        _ => panic!(
            "Expected signal to be in Transitioning state, got: {:?}",
            signal.tune_state
        ),
    }

    Ok(())
}

#[test]
fn test_signal_entity_rejects_unconfirmed_tune_requests() -> Result<()> {
    let task_id = TaskId::new("test-scan");
    let window_id = WindowId::new(task_id.clone(), 1);
    let mut signal = SignalEntity::new(88.9e6, window_id.clone());

    // Signal is NOT confirmed - analysis is still not started
    assert!(!signal.analysis.is_confirmed());
    assert!(matches!(signal.tune_state, TuneState::Idle));

    // Try to tune the unconfirmed signal
    let result = signal.request_tune_transition(window_id.clone(), 88.9e6);

    // Should fail with appropriate error
    assert!(
        result.is_err(),
        "Tune transition should fail for unconfirmed signal"
    );
    assert_eq!(result.unwrap_err(), "Cannot tune unconfirmed signal");

    // Verify: Signal should remain Idle
    assert!(
        matches!(signal.tune_state, TuneState::Idle),
        "Unconfirmed signal should remain Idle, got: {:?}",
        signal.tune_state
    );

    Ok(())
}

#[test]
fn test_signal_entity_rejects_double_tune_requests() -> Result<()> {
    let task_id = TaskId::new("test-scan");
    let window_id = WindowId::new(task_id.clone(), 1);
    let mut signal = SignalEntity::new(88.9e6, window_id.clone());

    // Confirm the signal
    signal
        .analysis
        .confirm_analysis(scanner::audio::quality::AudioQuality::Good, 0.8);

    // First tune request should succeed
    let result1 = signal.request_tune_transition(window_id.clone(), 88.9e6);
    assert!(result1.is_ok(), "First tune transition should succeed");
    assert!(matches!(signal.tune_state, TuneState::Transitioning(_)));

    // Second tune request should fail
    let result2 = signal.request_tune_transition(window_id.clone(), 89.3e6);
    assert!(result2.is_err(), "Second tune transition should fail");
    assert_eq!(result2.unwrap_err(), "Signal already tuning");

    // State should remain as first transition
    match &signal.tune_state {
        TuneState::Transitioning(transition) => {
            assert_eq!(transition.center_frequency, 88.9e6); // First request frequency
        }
        _ => panic!("Signal should still be in first Transitioning state"),
    }

    Ok(())
}
