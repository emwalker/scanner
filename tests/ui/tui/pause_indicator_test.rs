use std::sync::{Arc, Mutex};

use scanner::{ecs::GlobalPauseState, ui::tui::model::Model};

#[test]
fn test_pause_indicator_toggling() {
    let resource = Arc::new(Mutex::new(GlobalPauseState::Active));
    let mut model = Model::new();
    model.set_global_pause_resource(Arc::clone(&resource));

    assert!(!model.is_globally_paused());

    {
        let mut state = resource.lock().unwrap();
        *state = GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        };
    }

    assert!(model.is_globally_paused());

    {
        let mut state = resource.lock().unwrap();
        *state = GlobalPauseState::Active;
    }

    assert!(!model.is_globally_paused());
}

#[test]
fn test_pause_state_persistence_across_queries() {
    let resource = Arc::new(Mutex::new(GlobalPauseState::Paused {
        had_active_scans: true,
        playing_stations: vec![],
    }));
    let mut model = Model::new();
    model.set_global_pause_resource(Arc::clone(&resource));

    for _ in 0..10 {
        assert!(model.is_globally_paused());
    }
}
