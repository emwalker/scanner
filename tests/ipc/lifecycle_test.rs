use std::{thread, time::Duration};

use scanner::hardware::pool::{
    Pool, TaskPriority, TaskRequirements, TunerActivity, test_utils::add_test_device_to_pool,
};

#[test]
fn test_subprocess_spawns_lazily() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "lazy001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "lazy001");
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
}

#[test]
fn test_subprocess_reuse_for_second_allocation() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "reuse001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "reuse001");
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

    let tuner1 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner1);

    let tuner2 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner2);
    pool.shutdown();
}

#[test]
fn test_subprocess_graceful_shutdown() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "shutdown001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "shutdown001");
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
fn test_subprocess_persists_across_allocations() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "persist001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "persist001");
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

    let tuner1 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner1);

    let tuner2 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire tuner");

    drop(tuner2);
    pool.shutdown();
}
