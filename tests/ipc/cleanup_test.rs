use std::{thread, time::Duration};

use scanner::hardware::pool::{
    Pool, TaskPriority, TaskRequirements, TunerActivity, test_utils::add_test_device_to_pool,
};

#[test]
fn test_no_zombie_processes_after_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "zombie001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "zombie001");
    add_test_device_to_pool(
        &pool,
        device_id,
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
    );

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner);
    pool.shutdown();

    thread::sleep(Duration::from_millis(500));

    super::common::assert_no_zombies();
}

#[test]
fn test_socket_cleanup_after_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "socket001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "socket001");
    add_test_device_to_pool(
        &pool,
        device_id,
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
    );

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner);
    pool.shutdown();

    thread::sleep(Duration::from_millis(500));

    super::common::assert_sockets_cleaned("/tmp/scanner-mock-socket001-*.sock");
}

#[test]
fn test_tuner_drop_non_blocking_during_shutdown() {
    let pool = std::sync::Arc::new(Pool::new_unfiltered());

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "nonblock001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "nonblock001");
    add_test_device_to_pool(
        &pool,
        device_id,
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
    );

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    let pool_clone = std::sync::Arc::clone(&pool);
    let handle = thread::spawn(move || {
        pool_clone.shutdown();
    });

    thread::sleep(Duration::from_millis(10));
    drop(tuner);

    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_shutdown_terminates_all_subprocesses() {
    let pool = Pool::new_unfiltered();

    for i in 0..3 {
        let serial = format!("multi{:03}", i);
        let device_id = scanner::hardware::DeviceId::from_serial("mock", &serial);
        let caps = scanner::hardware::Capabilities::for_mock("mock", &serial);
        add_test_device_to_pool(
            &pool,
            device_id,
            caps,
            scanner::hardware::types::Backend::Mock,
            None,
        );
    }

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let mut tuners = Vec::new();
    for _ in 0..3 {
        if let Some(tuner) = pool.try_acquire(&requirements, TunerActivity::Scanning) {
            tuners.push(tuner);
        }
    }

    drop(tuners);
    pool.shutdown();

    thread::sleep(Duration::from_millis(500));

    super::common::assert_no_zombies();
}
