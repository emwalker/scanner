use crate::ui::tui::model::Model;
use std::sync::{Arc, Mutex};

#[test]
fn test_is_globally_paused_without_resource() {
    let model = Model::new();
    assert!(!model.is_globally_paused());
}

#[test]
fn test_is_globally_paused_active() {
    let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
    let mut model = Model::new();
    model.set_global_pause_resource(resource);
    assert!(!model.is_globally_paused());
}

#[test]
fn test_is_globally_paused_paused() {
    let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Paused {
        had_active_scans: true,
        had_active_audio: false,
    }));
    let mut model = Model::new();
    model.set_global_pause_resource(resource);
    assert!(model.is_globally_paused());
}
