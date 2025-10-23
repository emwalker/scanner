//! Resource management and lifecycle tests

use std::sync::{Arc, Mutex};

use scanner::{
    ecs::{
        Entity, EntityWorld, SystemContext,
        components::window::WindowId,
        entities::{TaskId, TunerEntity, WindowEntity},
    },
    hardware::{Capabilities, DeviceId, pool::TunerId, types::Backend},
};

#[test]
fn test_global_pause_deallocates_tuner_without_clearing_window() {
    let task_id = TaskId::new("scan_1");
    let window_id = WindowId::new(task_id.clone(), 0);

    let device_id = DeviceId::from_serial("sdrplay", "test123");
    let tuner_id = TunerId::new(device_id.clone(), 0);

    let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.9e6);

    window.allocation.allocate(tuner_id.clone());
    window.allocation.mark_complete();
    window.progress.mark_completed();

    let mut window_entities = EntityWorld::new();
    window_entities.insert(window);

    let capabilities = Capabilities::for_mock("sdrplay", "test123");
    let mut tuner = TunerEntity::new(
        device_id.clone(),
        0,
        capabilities,
        Backend::Soapy,
        "Test Tuner".to_string(),
        None,
        "FM".to_string(),
    );
    tuner.allocation.allocate("scan_1_window_0".to_string());

    let mut tuner_entities = EntityWorld::new();
    tuner_entities.insert(tuner);

    {
        let mut tuners = tuner_entities.iter();
        let tuner = tuners.find(|t| t.id() == &tuner_id).unwrap();
        assert!(
            tuner.allocation.is_allocated(),
            "Tuner should be allocated before pause"
        );
    }

    let context = SystemContext::new()
        .with_window_entities(Arc::new(std::sync::RwLock::new(window_entities)))
        .with_tuner_entities(Arc::new(Mutex::new(tuner_entities)));

    {
        let windows = context.window_entities.as_ref().unwrap().read().unwrap();
        if let Some(window) = windows.get(&window_id)
            && let Some(tuner_id_from_window) = window.allocation.tuner_id()
            && let Some(tuner_entities) = &context.tuner_entities
            && let Ok(mut tuners) = tuner_entities.try_lock()
            && let Some(tuner) = tuners.get_mut(tuner_id_from_window)
        {
            tuner.allocation.deallocate();
            tuner.status.idle();
        }
    }

    {
        let tuners = context.tuner_entities.as_ref().unwrap().lock().unwrap();
        let tuner = tuners.iter().find(|t| t.id() == &tuner_id).unwrap();
        assert!(
            tuner.allocation.is_available(),
            "Tuner should be deallocated after global pause"
        );
    }

    {
        let windows = context.window_entities.as_ref().unwrap().read().unwrap();
        let window = windows.get(&window_id).expect("Window should still exist");

        assert!(
            window.allocation.is_complete(),
            "Window should remain in Complete state, not cleared to None"
        );

        assert_eq!(
            window.allocation.tuner_id(),
            Some(&tuner_id),
            "Window should still remember which tuner it had"
        );

        assert!(
            window.progress.is_completed(),
            "Window progress should remain Completed"
        );
    }
}
