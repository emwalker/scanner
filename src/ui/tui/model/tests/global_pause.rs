use std::sync::{Arc, Mutex};

use crate::ui::tui::model::Model;

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
        playing_stations: vec![],
    }));
    let mut model = Model::new();
    model.set_global_pause_resource(resource);
    assert!(model.is_globally_paused());
}
