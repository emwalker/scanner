use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::{
    ecs::test_helpers::create_test_pool_with_entities,
    hardware,
    hardware::pool::{
        Pool, PoolFilter, TaskPriority, TaskRequirements, TunerActivity, TunerId, TunerState,
        TuningMode, test_utils, test_utils::add_test_device_to_pool,
    },
};

fn create_mock_device(device_id: &hardware::DeviceId) -> Box<dyn hardware::DeviceTrait> {
    let (driver, serial) = match device_id {
        hardware::DeviceId::Driver { driver, serial, .. } => (driver.as_str(), serial.as_str()),
        _ => ("mock", "unknown"),
    };
    Box::new(hardware::mock::MockDevice::new(driver, serial, false))
}

/// Helper to add test device to pool using ECS test utilities
fn add_mock_device_to_pool(pool: &Pool, device_id: hardware::DeviceId) {
    let device = create_mock_device(&device_id);
    let capabilities = device.capabilities().clone();
    let result = add_test_device_to_pool(
        pool,
        device_id,
        capabilities,
        crate::hardware::types::Backend::Mock,
        None,
    );
    match result {
        test_utils::AddTestDeviceResult::Added { .. } => {}
        test_utils::AddTestDeviceResult::FilteredOut { .. } => {
            panic!("Device was filtered out unexpectedly")
        }
        test_utils::AddTestDeviceResult::PoolBusy => panic!("Pool was busy"),
    }
}

#[test]
fn test_pooled_tuner_drop_doesnt_block_when_pool_locked() {
    let pool = Pool::new_unfiltered();
    let pool_arc = pool.pool_ref.clone();

    let device_id = hardware::DeviceId::from_serial("mock", "test001");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    let _pool_lock = pool_arc.lock().unwrap();

    let handle = thread::spawn(move || {
        drop(tuner);
    });

    let result = handle.join();
    assert!(result.is_ok());

    drop(_pool_lock);
}

#[test]
fn test_pooled_tuner_drop_during_shutdown() {
    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = hardware::DeviceId::from_serial("mock", "test002");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let pool_clone = Arc::clone(&pool);
    let handle = thread::spawn(move || {
        let tuner = pool_clone
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();
        thread::sleep(Duration::from_millis(50));
        drop(tuner);
    });

    thread::sleep(Duration::from_millis(10));

    let _status = pool.status();

    handle.join().unwrap();

    let final_status = pool.status();
    assert_eq!(final_status.allocated_tuner_count, 0);
}

#[test]
fn test_shutdown_mode() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test003");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    let status_before = pool.status();
    assert_eq!(status_before.allocated_tuner_count, 1);

    pool.shutdown();

    drop(tuner);

    let status_after = pool.status();
    assert_eq!(status_after.allocated_tuner_count, 0);
}

#[test]
fn test_shutdown_never_blocks() {
    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = hardware::DeviceId::from_serial("mock", "test004");
    add_mock_device_to_pool(&pool, device_id.clone());

    let pool_arc = pool.pool_ref.clone();
    let _lock = pool_arc.lock().unwrap();

    let pool_clone = Arc::clone(&pool);
    let handle = thread::spawn(move || {
        pool_clone.shutdown();
    });

    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_status_during_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test005");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let _tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    let status_before = pool.status();
    assert_eq!(status_before.allocated_tuner_count, 1);

    pool.shutdown();

    let status_after = pool.status();
    assert_eq!(status_after.allocated_tuner_count, 0);
    assert_eq!(status_after.available_tuner_count, 0);
    assert_eq!(status_after.device_count, 0);
}

#[test]
fn test_status_never_blocks_when_pool_locked() {
    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = hardware::DeviceId::from_serial("mock", "test006");
    add_mock_device_to_pool(&pool, device_id.clone());

    let pool_arc = pool.pool_ref.clone();
    let _lock = pool_arc.lock().unwrap();

    let pool_clone = Arc::clone(&pool);
    let handle = thread::spawn(move || {
        let status = pool_clone.status();
        // After migrating to device_entities, status() can still get device count
        // even when pool_ref is locked, since device_entities is a separate lock
        assert_eq!(status.device_count, 1);
    });

    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_acquire_rejected_during_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test007");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner_before = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner_before.is_some());
    drop(tuner_before);

    pool.shutdown();

    let tuner_after = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(
        tuner_after.is_none(),
        "Should not acquire tuner during shutdown"
    );
}

#[test]
fn test_acquire_never_blocks_when_pool_locked() {
    use std::time::Duration;

    let pool = Arc::new(Pool::new_unfiltered());

    let device_id = hardware::DeviceId::from_serial("mock", "test008");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let pool_arc = pool.pool_ref.clone();
    let _lock = pool_arc.lock().unwrap();

    let pool_clone = Arc::clone(&pool);
    let handle = thread::spawn(move || {
        let start = Instant::now();
        let result = pool_clone.try_acquire(&requirements, TunerActivity::Scanning);
        let elapsed = start.elapsed();

        assert!(result.is_none(), "Should return None when pool is locked");
        assert!(
            elapsed < Duration::from_millis(100),
            "Should return immediately, took {:?}",
            elapsed
        );
    });

    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_remove_device_rejected_during_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test010");
    add_mock_device_to_pool(&pool, device_id.clone());

    pool.shutdown();

    let result = pool.remove_device(&device_id);
    assert!(
        result.is_ok(),
        "Should succeed but skip removing device during shutdown"
    );
}

#[test]
fn test_remove_device_never_blocks_when_entities_locked() {
    use std::time::Duration;

    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test012");
    add_mock_device_to_pool(&pool, device_id.clone());

    let entities_arc = pool.device_entities.clone();
    let _lock = entities_arc.lock().unwrap();

    let handle = thread::spawn(move || {
        let pool_in_thread = pool;
        let start = Instant::now();
        let result = pool_in_thread.remove_device(&device_id);
        let elapsed = start.elapsed();

        assert!(
            result.is_err(),
            "Should return error when hardware entities are locked"
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "Should return immediately, took {:?}",
            elapsed
        );
    });

    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_tuner_operations_rejected_during_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test013");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    pool.shutdown();

    let mut graph = rustradio::graph::Graph::new();
    let add_source_result = tuner.add_source_to_graph(&mut graph, 100.0e6, 2.4e6, 30.0);
    assert!(
        add_source_result.is_err(),
        "add_source_to_graph should fail during shutdown: {:?}",
        add_source_result
    );
}

#[test]
fn test_is_shutdown() {
    let pool = Pool::new_unfiltered();

    assert!(
        !pool.is_shutdown(),
        "Pool should not be in shutdown mode initially"
    );

    pool.shutdown();

    assert!(
        pool.is_shutdown(),
        "Pool should be in shutdown mode after shutdown()"
    );
}

#[test]
fn test_is_shutdown_thread_safe() {
    let pool = Arc::new(Pool::new_unfiltered());

    let pool_clone = Arc::clone(&pool);
    let handle = thread::spawn(move || {
        // Check from another thread
        assert!(!pool_clone.is_shutdown());

        // Wait for main thread to trigger shutdown
        thread::sleep(Duration::from_millis(50));

        assert!(pool_clone.is_shutdown());
    });

    thread::sleep(Duration::from_millis(10));
    pool.shutdown();

    handle.join().unwrap();
}

#[test]
fn test_activity_tracking() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test014");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let _scanning_tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();

    let status = pool.status();
    assert_eq!(status.allocated_tuner_count, 1);

    let allocated_tuner = status
        .tuners
        .iter()
        .find(|t| t.state == TunerState::Allocated)
        .expect("Should have one allocated tuner");

    assert_eq!(allocated_tuner.activity, Some(TunerActivity::Scanning));

    drop(_scanning_tuner);

    let _listening_tuner = pool
        .try_acquire(&requirements, TunerActivity::Listening)
        .unwrap();

    let status = pool.status();
    let allocated_tuner = status
        .tuners
        .iter()
        .find(|t| t.state == TunerState::Allocated)
        .expect("Should have one allocated tuner");

    assert_eq!(allocated_tuner.activity, Some(TunerActivity::Listening));
}

#[test]
fn test_filter_by_backend() {
    let (pool, _tuner_entities, _device_entities) = create_test_pool_with_entities(
        PoolFilter::new().with_backend(hardware::types::Backend::Mock),
        None,
    );

    let mock_id = hardware::DeviceId::from_serial("mock", "test015");
    add_mock_device_to_pool(&pool, mock_id.clone());

    let soapy_id = hardware::DeviceId::from_serial("sdrplay", "test016");
    let soapy_device = create_mock_device(&soapy_id);
    let soapy_caps = soapy_device.capabilities().clone();
    // Different backend should be filtered out
    let result = add_test_device_to_pool(
        &pool,
        soapy_id,
        soapy_caps,
        hardware::types::Backend::Soapy,
        None,
    );
    assert!(matches!(
        result,
        test_utils::AddTestDeviceResult::FilteredOut { .. }
    ));

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner.is_some(), "Should acquire tuner from mock backend");

    let status = pool.status();
    assert_eq!(
        status.allocated_tuner_count, 1,
        "Should have one allocated tuner"
    );
    assert_eq!(
        status.available_tuner_count, 0,
        "Only soapy tuner was added, it's now allocated"
    );
}

#[test]
fn test_filter_by_driver() {
    let (pool, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(PoolFilter::new().with_driver("mock"), None);

    let mock_id1 = hardware::DeviceId::from_serial("mock", "test017");
    add_mock_device_to_pool(&pool, mock_id1.clone());

    let mock_id2 = hardware::DeviceId::from_serial("mock", "test018");
    // Same driver should NOT be filtered out
    let device_for_caps = create_mock_device(&mock_id2);
    let caps = device_for_caps.capabilities().clone();
    let result = add_test_device_to_pool(
        &pool,
        mock_id2.clone(),
        caps,
        hardware::types::Backend::Mock,
        None,
    );
    assert!(matches!(
        result,
        test_utils::AddTestDeviceResult::Added { .. }
    ));

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner.is_some(), "Should acquire mock tuner");

    let status = pool.status();
    let allocated = status
        .tuners
        .iter()
        .find(|t| t.state == TunerState::Allocated)
        .unwrap();
    assert!(
        format!("{:?}", allocated.id.device_id).contains("mock"),
        "Allocated tuner should be from mock driver"
    );
}

#[test]
fn test_filter_allow_tuners() {
    let pool = Pool::new_unfiltered();

    let device1_id = hardware::DeviceId::from_serial("mock", "test019");
    add_mock_device_to_pool(&pool, device1_id.clone());

    let device2_id = hardware::DeviceId::from_serial("mock", "test020");
    add_mock_device_to_pool(&pool, device2_id.clone());

    drop(pool);

    let tuner1 = TunerId::new(device1_id.clone(), 0);
    let (pool_filtered, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(PoolFilter::new().with_tuners(vec![tuner1.clone()]), None);

    add_mock_device_to_pool(&pool_filtered, device1_id.clone());

    // Device2 should be filtered out (only tuner1 is allowed)
    let device_for_caps = create_mock_device(&device2_id);
    let caps = device_for_caps.capabilities().clone();
    let result = add_test_device_to_pool(
        &pool_filtered,
        device2_id.clone(),
        caps,
        hardware::types::Backend::Mock,
        None,
    );
    assert!(matches!(
        result,
        test_utils::AddTestDeviceResult::FilteredOut { .. }
    ));

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool_filtered
        .try_acquire(&requirements, TunerActivity::Scanning)
        .unwrap();
    assert_eq!(tuner.id(), &tuner1, "Should only acquire allowed tuner");
}

#[test]
fn test_filter_single_tuner_mode() {
    let (pool, _tuner_entities, _device_entities) =
        create_test_pool_with_entities(PoolFilter::new().with_mode(TuningMode::SingleTuner), None);

    let device_id = hardware::DeviceId::from_serial("mock", "test021");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner1.is_some(), "Should acquire first tuner");

    let tuner2 = pool.try_acquire(&requirements, TunerActivity::Listening);
    assert!(
        tuner2.is_none(),
        "Should not acquire second tuner in SingleTuner mode"
    );

    drop(tuner1);

    let tuner3 = pool.try_acquire(&requirements, TunerActivity::Listening);
    assert!(
        tuner3.is_some(),
        "Should acquire tuner after first is released"
    );
}

#[test]
fn test_filter_combined_driver_and_mode() {
    let (pool, _tuner_entities, _device_entities) = create_test_pool_with_entities(
        PoolFilter::new()
            .with_driver("mock")
            .with_mode(TuningMode::SingleTuner),
        None,
    );

    let mock_id = hardware::DeviceId::from_serial("mock", "test022");
    add_mock_device_to_pool(&pool, mock_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner1.is_some(), "Should acquire mock tuner");

    let status = pool.status();
    let allocated = status
        .tuners
        .iter()
        .find(|t| t.state == TunerState::Allocated)
        .unwrap();
    assert!(
        format!("{:?}", allocated.id.device_id).contains("mock"),
        "Should allocate from mock driver"
    );

    let tuner2 = pool.try_acquire(&requirements, TunerActivity::Listening);
    assert!(
        tuner2.is_none(),
        "Should not allocate second tuner in SingleTuner mode"
    );
}

#[test]
fn test_initial_state_is_active() {
    let pool = Pool::new_unfiltered();
    assert!(pool.is_active());
    assert!(!pool.is_shutting_down());
}

#[test]
fn test_state_transitions() {
    let pool = Pool::new_unfiltered();

    assert!(pool.is_active());
    assert!(!pool.is_shutting_down());

    pool.shutdown();

    assert!(!pool.is_active());
    assert!(pool.is_shutting_down());
}

#[test]
fn test_acquire_only_allowed_in_active_state() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test026");
    add_mock_device_to_pool(&pool, device_id.clone());

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner1.is_some(), "Should acquire in Active state");
    drop(tuner1);

    pool.shutdown();

    let tuner2 = pool.try_acquire(&requirements, TunerActivity::Scanning);
    assert!(tuner2.is_none(), "Should not acquire in ShuttingDown state");
}

#[test]
fn test_shutdown_idempotent() {
    let pool = Pool::new_unfiltered();

    pool.shutdown();
    assert!(pool.is_shutting_down());

    pool.shutdown();
    assert!(pool.is_shutting_down());
}

#[test]
fn test_status_works_in_any_state() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test027");
    add_mock_device_to_pool(&pool, device_id.clone());

    let status1 = pool.status();
    assert_eq!(
        status1.device_count, 1,
        "Status should work in Active state"
    );

    pool.shutdown();

    let status2 = pool.status();
    assert_eq!(
        status2.device_count, 0,
        "Status returns empty during shutdown"
    );
}

#[test]
fn regression_test_callbacks_fire_on_tuner_acquire_and_release() {
    use std::sync::Mutex;

    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test028");
    add_mock_device_to_pool(&pool, device_id.clone());

    let callback_count = Arc::new(Mutex::new(0));
    let callback_count_clone = Arc::clone(&callback_count);

    pool.add_state_change_callback(Box::new(move |_status| {
        let mut count = callback_count_clone.lock().unwrap();
        *count += 1;
    }));

    assert_eq!(
        *callback_count.lock().unwrap(),
        0,
        "No callbacks should fire before acquiring tuner"
    );

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    {
        let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner.is_some(), "Should successfully acquire tuner");

        assert_eq!(
            *callback_count.lock().unwrap(),
            1,
            "Callback should fire once when tuner is acquired"
        );
    }

    assert_eq!(
        *callback_count.lock().unwrap(),
        2,
        "Callback should fire again when tuner is returned (dropped)"
    );

    let status = pool.status();
    assert_eq!(
        status.allocated_tuner_count, 0,
        "Tuner should be returned to pool"
    );
    assert_eq!(
        status.available_tuner_count, 1,
        "Tuner should be available again"
    );
}

#[test]
fn regression_test_multiple_callbacks_all_fire() {
    use std::sync::Mutex;

    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "test029");
    add_mock_device_to_pool(&pool, device_id.clone());

    let callback1_count = Arc::new(Mutex::new(0));
    let callback2_count = Arc::new(Mutex::new(0));
    let callback3_count = Arc::new(Mutex::new(0));

    let callback1_clone = Arc::clone(&callback1_count);
    let callback2_clone = Arc::clone(&callback2_count);
    let callback3_clone = Arc::clone(&callback3_count);

    pool.add_state_change_callback(Box::new(move |_status| {
        *callback1_clone.lock().unwrap() += 1;
    }));

    pool.add_state_change_callback(Box::new(move |_status| {
        *callback2_clone.lock().unwrap() += 1;
    }));

    pool.add_state_change_callback(Box::new(move |_status| {
        *callback3_clone.lock().unwrap() += 1;
    }));

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    {
        let _tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        assert_eq!(*callback1_count.lock().unwrap(), 1);
        assert_eq!(*callback2_count.lock().unwrap(), 1);
        assert_eq!(*callback3_count.lock().unwrap(), 1);
    }

    assert_eq!(
        *callback1_count.lock().unwrap(),
        2,
        "First callback should fire on acquire and release"
    );
    assert_eq!(
        *callback2_count.lock().unwrap(),
        2,
        "Second callback should fire on acquire and release"
    );
    assert_eq!(
        *callback3_count.lock().unwrap(),
        2,
        "Third callback should fire on acquire and release"
    );
}

#[test]
fn test_tuner_status_contains_id_and_state() {
    let pool = Pool::new_unfiltered();

    let device_id = hardware::DeviceId::from_serial("mock", "2301034E34:ST");
    add_mock_device_to_pool(&pool, device_id.clone());

    let status = pool.status();
    assert_eq!(status.tuners.len(), 1, "Should have one tuner");

    let tuner = &status.tuners[0];
    assert_eq!(tuner.state, TunerState::Available);
    assert_eq!(tuner.activity, None);

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let _tuner = pool
        .try_acquire(&requirements, TunerActivity::Listening)
        .unwrap();

    let status_allocated = pool.status();
    let allocated_tuner = &status_allocated.tuners[0];
    assert_eq!(allocated_tuner.state, TunerState::Allocated);
    assert_eq!(allocated_tuner.activity, Some(TunerActivity::Listening));
}

#[cfg(test)]
mod ecs_sync_tests {
    use super::*;

    #[test]
    fn test_device_addition_creates_tuner_entity() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test030");
        add_mock_device_to_pool(&pool, device_id.clone());

        let entities = pool.tuner_entities.lock().unwrap();
        assert_eq!(entities.len(), 1, "Should have one TunerEntity");

        let tuner_id = TunerId::new(device_id, 0);
        let entity = entities.get(&tuner_id).expect("TunerEntity should exist");

        assert!(entity.is_available(), "Entity should be available");
        assert!(entity.is_connected(), "Entity should be connected");
        assert!(
            entity.allocation.is_available(),
            "Allocation should be available"
        );
    }

    #[test]
    fn test_device_removal_deletes_tuner_entity() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test031");
        add_mock_device_to_pool(&pool, device_id.clone());

        {
            let entities = pool.tuner_entities.lock().unwrap();
            assert_eq!(entities.len(), 1, "Should have one entity after add");
        }

        pool.remove_device(&device_id).unwrap();

        {
            let entities = pool.tuner_entities.lock().unwrap();
            assert_eq!(entities.len(), 0, "Should have zero entities after remove");
        }
    }

    #[test]
    fn test_allocation_updates_entity_state() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test032");
        add_mock_device_to_pool(&pool, device_id.clone());

        let tuner_id = TunerId::new(device_id, 0);

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let entity = entities.get(&tuner_id).unwrap();
            assert!(
                entity.allocation.is_available(),
                "Should be available before allocation"
            );
        }

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let _tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let entity = entities.get(&tuner_id).unwrap();
            assert!(
                entity.allocation.is_allocated(),
                "Should be allocated after acquire"
            );
        }
    }

    #[test]
    fn test_deallocation_updates_entity_state() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test033");
        add_mock_device_to_pool(&pool, device_id.clone());

        let tuner_id = TunerId::new(device_id, 0);

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        {
            let tuner = pool
                .try_acquire(&requirements, TunerActivity::Listening)
                .unwrap();

            {
                let entities = pool.tuner_entities.lock().unwrap();
                let entity = entities.get(&tuner_id).unwrap();
                assert!(
                    entity.allocation.is_allocated(),
                    "Should be allocated while tuner is held"
                );
            }

            drop(tuner);
        }

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let entity = entities.get(&tuner_id).unwrap();
            assert!(
                entity.allocation.is_available(),
                "Should be available after tuner is dropped"
            );
        }
    }

    #[test]
    fn test_multiple_devices_entity_sync() {
        let pool = Pool::new_unfiltered();

        let device1_id = hardware::DeviceId::from_serial("mock", "test034");
        add_mock_device_to_pool(&pool, device1_id.clone());

        let device2_id = hardware::DeviceId::from_serial("mock", "test035");
        add_mock_device_to_pool(&pool, device2_id.clone());

        {
            let entities = pool.tuner_entities.lock().unwrap();
            assert_eq!(entities.len(), 2, "Should have two entities");
        }

        pool.remove_device(&device1_id).unwrap();

        {
            let entities = pool.tuner_entities.lock().unwrap();
            assert_eq!(
                entities.len(),
                1,
                "Should have one entity after removing one device"
            );
        }
    }

    #[test]
    fn test_allocation_deallocation_cycle_maintains_sync() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test036");
        add_mock_device_to_pool(&pool, device_id.clone());

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        for _i in 0..5 {
            let tuner = pool
                .try_acquire(&requirements, TunerActivity::Scanning)
                .unwrap();

            drop(tuner);
        }
    }

    #[test]
    fn test_shutdown_does_not_break_entity_sync() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test037");
        add_mock_device_to_pool(&pool, device_id.clone());

        pool.shutdown();

        let status = pool.status();
        assert_eq!(status.device_count, 0, "Devices cleared during shutdown");

        let entities = pool.tuner_entities.lock().unwrap();
        assert_eq!(
            entities.len(),
            1,
            "Entities remain during shutdown (not cleared)"
        );
    }

    #[test]
    fn test_concurrent_allocation_maintains_entity_sync() {
        let pool = Arc::new(Pool::new_unfiltered());

        let device1_id = hardware::DeviceId::from_serial("mock", "test038");
        add_mock_device_to_pool(&pool, device1_id.clone());

        let device2_id = hardware::DeviceId::from_serial("mock", "test039");
        add_mock_device_to_pool(&pool, device2_id.clone());

        let pool_clone1 = Arc::clone(&pool);
        let pool_clone2 = Arc::clone(&pool);

        let requirements1 = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let requirements2 = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let handle1 =
            thread::spawn(move || pool_clone1.try_acquire(&requirements1, TunerActivity::Scanning));

        let handle2 = thread::spawn(move || {
            pool_clone2.try_acquire(&requirements2, TunerActivity::Listening)
        });

        let tuner1 = handle1.join().unwrap();
        let tuner2 = handle2.join().unwrap();

        let acquired_count = [tuner1.is_some(), tuner2.is_some()]
            .iter()
            .filter(|&&x| x)
            .count();

        assert!(
            acquired_count >= 1,
            "At least one thread should acquire a tuner"
        );
        assert!(
            acquired_count <= 2,
            "At most two threads can acquire tuners (2 devices)"
        );
    }

    #[test]
    fn test_entity_query_drives_allocation() {
        let pool = Pool::new_unfiltered();

        let device1_id = hardware::DeviceId::from_serial("mock", "test040");
        add_mock_device_to_pool(&pool, device1_id.clone());

        let device2_id = hardware::DeviceId::from_serial("mock", "test041");
        add_mock_device_to_pool(&pool, device2_id.clone());

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner1 = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .expect("Should acquire first tuner via entity query");

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let allocated_entities = entities
                .iter()
                .filter(|e| e.allocation.is_allocated())
                .count();
            assert_eq!(
                allocated_entities, 1,
                "Exactly one entity should be allocated after first acquire"
            );
        }

        let tuner2 = pool
            .try_acquire(&requirements, TunerActivity::Listening)
            .expect("Should acquire second tuner via entity query");

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let allocated_entities = entities
                .iter()
                .filter(|e| e.allocation.is_allocated())
                .count();
            assert_eq!(
                allocated_entities, 2,
                "Both entities should be allocated after second acquire"
            );
        }

        drop(tuner1);

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let allocated_entities = entities
                .iter()
                .filter(|e| e.allocation.is_allocated())
                .count();
            assert_eq!(
                allocated_entities, 1,
                "One entity should remain allocated after first tuner dropped"
            );
        }

        drop(tuner2);

        {
            let entities = pool.tuner_entities.lock().unwrap();
            let available_entities = entities
                .iter()
                .filter(|e| e.allocation.is_available())
                .count();
            assert_eq!(
                available_entities, 2,
                "Both entities should be available after all tuners dropped"
            );
        }
    }

    #[test]
    fn test_pool_status_applies_filter() {
        let device1_id = hardware::DeviceId::from_serial("sdrplay", "2301034E34");
        let device2_id = hardware::DeviceId::from_serial("rtlsdr", "00000001");

        let (pool, _tuner_entities, _device_entities) =
            create_test_pool_with_entities(PoolFilter::new().with_driver("sdrplay"), None);

        add_mock_device_to_pool(&pool, device1_id.clone());

        let device_for_caps = create_mock_device(&device2_id);
        let caps = device_for_caps.capabilities().clone();
        let result = add_test_device_to_pool(
            &pool,
            device2_id.clone(),
            caps,
            hardware::types::Backend::Soapy,
            None,
        );
        assert!(matches!(
            result,
            test_utils::AddTestDeviceResult::FilteredOut { .. }
        ));

        let status = pool.status();

        assert_eq!(
            status.tuners.len(),
            1,
            "Status should only show filtered tuners"
        );
        assert!(
            format!("{:?}", status.tuners[0].id.device_id).contains("sdrplay"),
            "Status should only contain sdrplay tuner"
        );
    }

    #[test]
    fn test_pool_status_filters_by_channel() {
        let device1_id = hardware::DeviceId::from_serial("mock", "dev001");
        let device2_id = hardware::DeviceId::from_serial("mock", "dev002");

        let (pool, _tuner_entities, _device_entities) = create_test_pool_with_entities(
            PoolFilter::new().with_tuners(vec![TunerId::new(device2_id.clone(), 0)]),
            None,
        );

        let device_for_caps = create_mock_device(&device1_id);
        let caps = device_for_caps.capabilities().clone();
        let result = add_test_device_to_pool(
            &pool,
            device1_id.clone(),
            caps,
            hardware::types::Backend::Mock,
            None,
        );
        assert!(matches!(
            result,
            test_utils::AddTestDeviceResult::FilteredOut { .. }
        ));

        add_mock_device_to_pool(&pool, device2_id.clone());

        let status = pool.status();

        assert_eq!(
            status.tuners.len(),
            1,
            "Status should only show filtered tuner"
        );
        assert_eq!(
            status.tuners[0].id.device_id, device2_id,
            "Filtered tuner should be from device2"
        );
    }

    #[test]
    fn test_allocation_uses_filtered_status() {
        let device1_id = hardware::DeviceId::from_serial("sdrplay", "2301034E34");

        let (pool, _tuner_entities, _device_entities) =
            create_test_pool_with_entities(PoolFilter::new().with_driver("sdrplay"), None);

        add_mock_device_to_pool(&pool, device1_id.clone());

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .expect("Should allocate sdrplay tuner");

        assert!(
            format!("{:?}", tuner.id().device_id).contains("sdrplay"),
            "Allocated tuner should be from allowed driver only"
        );
    }

    #[test]
    fn test_tuner_id_preserves_channel_index() {
        let device1_id = hardware::DeviceId::from_serial("mock", "device1");
        let device2_id = hardware::DeviceId::from_serial("mock", "device2");
        let device3_id = hardware::DeviceId::from_serial("mock", "device3");

        let pool = Pool::new_unfiltered();

        add_mock_device_to_pool(&pool, device1_id.clone());
        add_mock_device_to_pool(&pool, device2_id.clone());
        add_mock_device_to_pool(&pool, device3_id.clone());

        let status = pool.status();
        assert_eq!(status.tuners.len(), 3, "Should have 3 tuners");

        assert_eq!(status.tuners[0].id.channel_index, 0);
        assert_eq!(status.tuners[1].id.channel_index, 0);
        assert_eq!(status.tuners[2].id.channel_index, 0);

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner0 = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .expect("Should acquire first tuner");
        assert_eq!(
            tuner0.id().channel_index,
            0,
            "Each device has channel_index 0"
        );

        let tuner1 = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .expect("Should acquire second tuner");
        assert_eq!(
            tuner1.id().channel_index,
            0,
            "Each device has channel_index 0"
        );

        let tuner2 = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .expect("Should acquire third tuner");
        assert_eq!(
            tuner2.id().channel_index,
            0,
            "Each device has channel_index 0"
        );

        assert_ne!(
            tuner0.id().device_id,
            tuner1.id().device_id,
            "Different tuners should have different device_ids"
        );
        assert_ne!(
            tuner1.id().device_id,
            tuner2.id().device_id,
            "Different tuners should have different device_ids"
        );
    }

    #[test]
    fn regression_test_pool_status_does_not_cause_lock_contention() {
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };

        let pool = Arc::new(Pool::new_unfiltered());

        let device_id = hardware::DeviceId::from_serial("mock", "test_contention");
        add_mock_device_to_pool(&pool, device_id.clone());

        let barrier = Arc::new(Barrier::new(2));
        let contention_count = Arc::new(AtomicUsize::new(0));

        let pool_clone = Arc::clone(&pool);
        let barrier_clone = Arc::clone(&barrier);

        let status_thread = thread::spawn(move || {
            barrier_clone.wait();

            for _ in 0..1000 {
                let _status = pool_clone.status();
            }
        });

        let contention_clone = Arc::clone(&contention_count);
        let entities_thread = thread::spawn(move || {
            barrier.wait();

            for _ in 0..1000 {
                if pool.device_entities.try_lock().is_err() {
                    contention_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });

        status_thread.join().unwrap();
        entities_thread.join().unwrap();

        let final_contention = contention_count.load(Ordering::SeqCst);
        assert!(
            final_contention < 150,
            "Lock contention should be low (< 150 contentions out of 1000 attempts), got {}. \
             pool.status() should only briefly lock device_entities to get count, not hold it \
             during status build.",
            final_contention
        );
    }

    #[test]
    fn regression_test_concurrent_status_and_device_entity_access_never_deadlocks() {
        use std::sync::Barrier;

        let pool = Arc::new(Pool::new_unfiltered());

        let device_id = hardware::DeviceId::from_serial("mock", "test_no_deadlock");
        add_mock_device_to_pool(&pool, device_id.clone());

        let iterations = 1000;
        let barrier = Arc::new(Barrier::new(3));

        let pool1 = Arc::clone(&pool);
        let barrier1 = Arc::clone(&barrier);
        let thread1 = thread::spawn(move || {
            barrier1.wait();
            for _ in 0..iterations {
                let _status = pool1.status();
            }
        });

        let pool2 = Arc::clone(&pool);
        let barrier2 = Arc::clone(&barrier);
        let thread2 = thread::spawn(move || {
            barrier2.wait();
            for _ in 0..iterations {
                if let Ok(entities) = pool2.device_entities.try_lock() {
                    let _count = entities.len();
                }
            }
        });

        let pool3 = Arc::clone(&pool);
        let barrier3 = Arc::clone(&barrier);
        let thread3 = thread::spawn(move || {
            barrier3.wait();
            for _ in 0..iterations {
                if let Ok(entities) = pool3.tuner_entities.try_lock() {
                    let _count = entities.len();
                }
            }
        });

        let start = Instant::now();

        thread1.join().unwrap();
        thread2.join().unwrap();
        thread3.join().unwrap();

        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "All threads should complete without deadlock, took {:?}",
            elapsed
        );
    }
}
