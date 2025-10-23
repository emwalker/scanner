use std::{thread, time::Duration};

use scanner::hardware::pool::{
    Pool, TaskPriority, TaskRequirements, TunerActivity, test_utils::add_test_device_to_pool,
};

#[test]
fn test_control_message_round_trip() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "comm001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "comm001");
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

    thread::sleep(Duration::from_millis(100));

    drop(tuner);
    pool.shutdown();
}

#[test]
fn test_iq_data_packets_with_channel_tags() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "iq001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "iq001");
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

    thread::sleep(Duration::from_millis(200));

    drop(tuner);
    pool.shutdown();
}

#[test]
fn test_terminal_isolation() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "iso001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "iso001");
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
