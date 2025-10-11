#[cfg(test)]
mod tests {
    use crate::hardware;
    use crate::hardware::pool::{
        AddDeviceResult, Pool, PoolFilter, TaskPriority, TaskRequirements, TunerActivity, TunerId,
        TunerState, TuningMode,
    };
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    fn create_mock_device(device_id: &hardware::DeviceId) -> Box<dyn hardware::DeviceTrait> {
        let (driver, serial) = match device_id {
            hardware::DeviceId::Backend { backend, serial } => (backend.as_str(), serial.as_str()),
            _ => ("mock", "unknown"),
        };
        Box::new(hardware::mock::MockDevice::new(driver, serial, false))
    }

    #[test]
    fn test_pooled_tuner_drop_doesnt_block_when_pool_locked() {
        let pool = Pool::new_unfiltered();
        let pool_arc = pool.pool_ref.clone();

        let device_id = hardware::DeviceId::from_serial("mock", "test001");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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

        thread::sleep(Duration::from_millis(10));

        let status_after = pool.status();
        assert_eq!(status_after.allocated_tuner_count, 0);
    }

    #[test]
    fn test_shutdown_never_blocks() {
        let pool = Arc::new(Pool::new_unfiltered());

        let device_id = hardware::DeviceId::from_serial("mock", "test004");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let status = pool_clone.status();
            assert_eq!(status.device_count, 0);
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_acquire_rejected_during_shutdown() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test007");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
    fn test_add_device_rejected_during_shutdown() {
        let pool = Pool::new_unfiltered();

        pool.shutdown();

        let device_id = hardware::DeviceId::from_serial("mock", "test009");
        let device = create_mock_device(&device_id);
        let result = pool.add_device(device, "mock".to_string());

        assert!(
            matches!(result, AddDeviceResult::ShutdownMode),
            "Should return ShutdownMode"
        );

        let status = pool.status();
        assert_eq!(
            status.device_count, 0,
            "Device should not be added during shutdown"
        );
    }

    #[test]
    fn test_remove_device_rejected_during_shutdown() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test010");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

        pool.shutdown();

        let result = pool.remove_device(&device_id);
        assert!(
            result.is_ok(),
            "Should succeed but skip removing device during shutdown"
        );
    }

    #[test]
    fn test_add_device_never_blocks_when_pool_locked() {
        use std::time::Duration;

        let pool = Pool::new_unfiltered();
        let pool_arc = pool.pool_ref.clone();

        // Hold the lock in this thread
        let _lock = pool_arc.lock().unwrap();

        // Try to add device from another thread
        let handle = thread::spawn(move || {
            let pool_in_thread = pool;
            let start = Instant::now();
            let device_id = hardware::DeviceId::from_serial("mock", "test011");
            let device = create_mock_device(&device_id);
            let result = pool_in_thread.add_device(device, "mock".to_string());
            let elapsed = start.elapsed();

            assert!(result.is_err(), "Should return error when pool is locked");
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
    fn test_remove_device_never_blocks_when_pool_locked() {
        use std::time::Duration;

        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test012");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let handle = thread::spawn(move || {
            let pool_in_thread = pool;
            let start = Instant::now();
            let result = pool_in_thread.remove_device(&device_id);
            let elapsed = start.elapsed();

            assert!(result.is_err(), "Should return error when pool is locked");
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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let mut tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        pool.shutdown();

        let tune_result = tuner.tune(100.0e6);
        assert!(
            tune_result.is_err(),
            "Tune should fail during shutdown: {:?}",
            tune_result
        );

        let gain_result = tuner.set_gain(30.0);
        assert!(
            gain_result.is_err(),
            "Set gain should fail during shutdown: {:?}",
            gain_result
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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let pool = Pool::new(PoolFilter::new().with_backend("soapy"));

        let soapy_id = hardware::DeviceId::from_serial("sdrplay", "test015");
        let soapy_device = create_mock_device(&soapy_id);
        pool.add_device(soapy_device, "soapy".to_string()).unwrap();

        let rtlsdr_id = hardware::DeviceId::from_serial("rtlsdr", "test016");
        let rtlsdr_device = create_mock_device(&rtlsdr_id);
        // Different backend should be filtered out
        let result = pool.add_device(rtlsdr_device, "rtlsdr".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner.is_some(), "Should acquire tuner from soapy backend");

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
        let pool = Pool::new(PoolFilter::new().with_driver("sdrplay"));

        let sdrplay_id = hardware::DeviceId::from_serial("sdrplay", "test017");
        let sdrplay_device = create_mock_device(&sdrplay_id);
        pool.add_device(sdrplay_device, "soapy".to_string())
            .unwrap();

        let rtlsdr_id = hardware::DeviceId::from_serial("rtlsdr", "test018");
        let rtlsdr_device = create_mock_device(&rtlsdr_id);
        // RTL-SDR should be filtered out by driver check
        let result = pool.add_device(rtlsdr_device, "soapy".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner.is_some(), "Should acquire sdrplay tuner");

        let status = pool.status();
        let allocated = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .unwrap();
        assert!(
            format!("{:?}", allocated.id.device_id).contains("sdrplay"),
            "Allocated tuner should be from sdrplay driver"
        );
    }

    #[test]
    fn test_filter_allow_tuners() {
        let pool = Pool::new_unfiltered();

        let device1_id = hardware::DeviceId::from_serial("mock", "test019");
        let device1 = create_mock_device(&device1_id);
        pool.add_device(device1, "mock".to_string()).unwrap();

        let device2_id = hardware::DeviceId::from_serial("mock", "test020");
        let device2 = create_mock_device(&device2_id);
        pool.add_device(device2, "mock".to_string()).unwrap();

        drop(pool);

        let tuner1 = TunerId::new(device1_id.clone(), 0);
        let pool_filtered = Pool::new(PoolFilter::new().with_tuners(vec![tuner1.clone()]));

        let device1_again = create_mock_device(&device1_id);
        pool_filtered
            .add_device(device1_again, "mock".to_string())
            .unwrap();

        let device2_again = create_mock_device(&device2_id);
        // Device2 should be filtered out (only tuner1 is allowed)
        let result = pool_filtered.add_device(device2_again, "mock".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

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
        let pool = Pool::new(PoolFilter::new().with_mode(TuningMode::SingleTuner));

        let device_id = hardware::DeviceId::from_serial("mock", "test021");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let pool = Pool::new(
            PoolFilter::new()
                .with_driver("sdrplay")
                .with_mode(TuningMode::SingleTuner),
        );

        let sdrplay_id = hardware::DeviceId::from_serial("sdrplay", "test022");
        let sdrplay_device = create_mock_device(&sdrplay_id);
        pool.add_device(sdrplay_device, "soapy".to_string())
            .unwrap();

        let rtlsdr_id = hardware::DeviceId::from_serial("rtlsdr", "test023");
        let rtlsdr_device = create_mock_device(&rtlsdr_id);
        // RTL-SDR should be filtered out by driver check
        let result = pool.add_device(rtlsdr_device, "rtlsdr".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner1.is_some(), "Should acquire sdrplay tuner");

        let status = pool.status();
        let allocated = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .unwrap();
        assert!(
            format!("{:?}", allocated.id.device_id).contains("sdrplay"),
            "Should allocate from sdrplay driver"
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
    fn test_add_device_only_allowed_in_active_state() {
        let pool = Pool::new_unfiltered();

        let device1_id = hardware::DeviceId::from_serial("mock", "test024");
        let device1 = create_mock_device(&device1_id);
        let result1 = pool.add_device(device1, "mock".to_string());
        assert!(
            matches!(result1, AddDeviceResult::Added { .. }),
            "Should add device in Active state"
        );

        pool.shutdown();

        let device2_id = hardware::DeviceId::from_serial("mock", "test025");
        let device2 = create_mock_device(&device2_id);
        let result2 = pool.add_device(device2, "mock".to_string());
        assert!(
            matches!(result2, AddDeviceResult::ShutdownMode),
            "Should not add device in ShuttingDown state"
        );
    }

    #[test]
    fn test_acquire_only_allowed_in_active_state() {
        let pool = Pool::new_unfiltered();

        let device_id = hardware::DeviceId::from_serial("mock", "test026");
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
        let device = create_mock_device(&device_id);
        pool.add_device(device, "mock".to_string()).unwrap();

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
}
