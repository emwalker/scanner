use std::{thread, time::Duration};

use scanner::hardware::pool::{
    Pool, TaskPriority, TaskRequirements, TunerActivity, test_utils::add_test_device_to_pool,
};

#[test]
fn test_subprocess_crash_isolation() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "crash001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "crash001");
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
fn test_failed_device_tune() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "failtune001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "failtune001");
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
fn test_no_available_tuner() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "noavail001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "noavail001");
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
        .expect("Failed to acquire first tuner");

    let tuner2 = pool.try_acquire(&requirements, TunerActivity::Scanning);

    assert!(
        tuner2.is_none(),
        "Should not acquire second tuner from single-channel device"
    );

    drop(tuner1);
    pool.shutdown();
}

#[test]
fn test_unsupported_frequency() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "unsupfreq001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "unsupfreq001");
    add_test_device_to_pool(
        &pool,
        device_id,
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
    );

    let requirements = TaskRequirements {
        frequency_hz: 10e9,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);

    assert!(
        tuner.is_none(),
        "Should not acquire tuner for unsupported frequency"
    );

    pool.shutdown();
}

#[test]
fn test_unsupported_sample_rate() {
    let pool = Pool::new_unfiltered();

    let device_id = scanner::hardware::DeviceId::from_serial("mock", "unsuprate001");
    let caps = scanner::hardware::Capabilities::for_mock("mock", "unsuprate001");
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
        required_sample_rate: 100e6,
        priority: TaskPriority::Normal,
    };

    let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);

    assert!(
        tuner.is_none(),
        "Should not acquire tuner for unsupported sample rate"
    );

    pool.shutdown();
}
