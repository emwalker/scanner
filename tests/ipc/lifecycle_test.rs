use scanner::hardware::mock::MockDevice;
use scanner::hardware::pool::{Pool, TaskPriority, TaskRequirements, TunerActivity};
use std::thread;
use std::time::Duration;

#[test]
fn test_subprocess_spawns_lazily() {
    let pool = Pool::new_with_subprocesses();

    let device = Box::new(MockDevice::new("mock", "lazy001", false));
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

#[test]
fn test_subprocess_reuse_for_second_allocation() {
    let pool = Pool::new_with_subprocesses();

    let device = Box::new(MockDevice::new("mock", "reuse001", false));
    pool.add_device(device, scanner::hardware::types::Backend::Mock);

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
    let pool = Pool::new_with_subprocesses();

    let device = Box::new(MockDevice::new("mock", "shutdown001", false));
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

    thread::sleep(Duration::from_millis(500));

    super::common::assert_no_zombies();
}

#[test]
fn test_subprocess_persists_across_allocations() {
    let pool = Pool::new_with_subprocesses();

    let device = Box::new(MockDevice::new("mock", "persist001", false));
    pool.add_device(device, scanner::hardware::types::Backend::Mock);

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
