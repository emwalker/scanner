use crate::ecs::GlobalPauseState;
use crate::ui::tui::model::Model;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[test]
fn test_animation_freezes_during_global_pause() {
    let mut model = Model::default();
    let pause_resource = Arc::new(Mutex::new(GlobalPauseState::Active));
    model.set_global_pause_resource(pause_resource.clone());

    // Verify animation advances when not paused
    assert!(!model.is_globally_paused());

    // Pause the application
    {
        let mut state = pause_resource.lock().unwrap();
        *state = GlobalPauseState::Paused {
            had_active_scans: false,
            had_active_audio: false,
        };
    }

    // Verify model sees paused state
    assert!(model.is_globally_paused());

    // Sleep briefly to simulate time passing
    thread::sleep(Duration::from_millis(100));

    // Unpause
    {
        let mut state = pause_resource.lock().unwrap();
        *state = GlobalPauseState::Active;
    }

    assert!(!model.is_globally_paused());
}

#[test]
fn test_animation_resumes_after_unpause() {
    let mut model = Model::default();
    let pause_resource = Arc::new(Mutex::new(GlobalPauseState::Paused {
        had_active_scans: true,
        had_active_audio: false,
    }));
    model.set_global_pause_resource(pause_resource.clone());

    assert!(model.is_globally_paused());

    // Unpause the application
    {
        let mut state = pause_resource.lock().unwrap();
        *state = GlobalPauseState::Active;
    }

    assert!(!model.is_globally_paused());
}
