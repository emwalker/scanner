use super::super::helpers::{ModelTestContext, TestSignalState};

#[test]
fn test_model_utility_functions() {
    let mut ctx = ModelTestContext::new();

    assert!(ctx.model.is_empty());
    assert!(!ctx.model.all_complete());
    assert_eq!(ctx.model.signal_count(), 0);

    let window_id = 1;
    let signals = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

    for (_, freq) in &signals {
        ctx.update_signal(*freq, window_id, TestSignalState::Detected, None, None);
    }
    ctx.sync();

    assert!(!ctx.model.is_empty());
    assert!(!ctx.model.all_complete());
    assert_eq!(ctx.model.signal_count(), 2);

    for (_, freq) in &signals {
        ctx.update_signal(*freq, window_id, TestSignalState::Rejected, None, None);
    }
    ctx.sync();

    assert!(!ctx.model.is_empty());
    ctx.model.total_windows = Some(1);
    assert!(ctx.model.all_complete());
    assert_eq!(ctx.model.signal_count(), 2);
}
