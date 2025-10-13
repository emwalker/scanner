use scanner::hardware::mock::MockDevice;
use scanner::hardware::pool::{Pool, TaskPriority, TaskRequirements, TunerActivity};
use std::thread;
use std::time::Duration;

#[test]
fn test_control_message_round_trip() {
    let pool = Pool::new_unfiltered();

    let device = Box::new(MockDevice::new("mock", "comm001", false));
    pool.add_device(device, scanner::hardware::types::Backend::Mock);

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

    let device = Box::new(MockDevice::new("mock", "iq001", false));
    pool.add_device(device, scanner::hardware::types::Backend::Mock);

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

    let device = Box::new(MockDevice::new("mock", "iso001", false));
    pool.add_device(device, scanner::hardware::types::Backend::Mock);

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
