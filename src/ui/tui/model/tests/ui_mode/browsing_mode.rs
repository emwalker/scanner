use super::super::helpers::{ModelTestContext, TestSignalState};
use crate::{ecs::components::signal::SignalId, ui::tui::model::UiMode};

#[test]
fn test_browsing_mode_only_true_when_scan_paused() {
    let mut ctx = ModelTestContext::new();
    let window_id = 0;

    ctx.update_signal(
        88_900_000.0,
        window_id,
        TestSignalState::Detected,
        None,
        None,
    );
    ctx.sync();

    assert!(ctx.model.is_idle());
    assert!(!ctx.model.browsing_mode());

    ctx.model.enter_selection_mode();
    assert!(matches!(
        ctx.model.ui_mode,
        UiMode::NavigatingScanner { .. }
    ));
    assert!(ctx.model.selection_mode());
    assert!(!ctx.model.browsing_mode());

    if let Some(signal_index) = ctx.model.selected_signal_index() {
        let signal_id = ctx.model.windows.get(&window_id).unwrap().signals[signal_index]
            .signal_id
            .clone();
        ctx.model.ui_mode = UiMode::AwaitingTune {
            signal_index,
            window_id,
            tuning_signal_id: signal_id,
        };
    }
    assert!(matches!(ctx.model.ui_mode, UiMode::AwaitingTune { .. }));
    assert!(ctx.model.browsing_mode());

    if let Some(signal_index) = ctx.model.selected_signal_index() {
        ctx.model.ui_mode = UiMode::Listening {
            signal_index,
            window_id,
            playing_signal_id: SignalId::from_string("test-signal".to_string()),
        };
    }
    assert!(matches!(ctx.model.ui_mode, UiMode::Listening { .. }));
    assert!(ctx.model.browsing_mode());
}
