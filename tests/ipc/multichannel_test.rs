use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use scanner::{
    ecs::{EntityWorld, test_helpers::create_test_device_and_tuners},
    hardware::{
        Capabilities,
        pool::{Pool, PoolFilter, TaskPriority, TaskRequirements, TunerActivity},
    },
};

#[test]
fn test_two_tuners_share_one_subprocess() {
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let device_entities = Arc::new(Mutex::new(EntityWorld::new()));

    let pool = Pool::with_entity_worlds(
        PoolFilter::allow_all(),
        None,
        tuner_entities.clone(),
        device_entities.clone(),
    );

    let mut caps = Capabilities::for_mock("mock", "dual001");
    caps.channels = 2;

    let (device, tuners) = create_test_device_and_tuners(
        caps.device_id.clone(),
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
        vec![],
    );

    {
        let mut devices = device_entities.lock().unwrap();
        devices.insert(device);
    }

    {
        let mut tuner_world = tuner_entities.lock().unwrap();
        for tuner in tuners {
            tuner_world.insert(tuner);
        }
    }

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire first tuner");

    let tuner2 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire second tuner");

    thread::sleep(Duration::from_millis(100));

    drop(tuner1);
    drop(tuner2);
    pool.shutdown();
}

#[test]
fn test_stop_one_channel_doesnt_affect_other() {
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let device_entities = Arc::new(Mutex::new(EntityWorld::new()));

    let pool = Pool::with_entity_worlds(
        PoolFilter::allow_all(),
        None,
        tuner_entities.clone(),
        device_entities.clone(),
    );

    let mut caps = Capabilities::for_mock("mock", "dual002");
    caps.channels = 2;

    let (device, tuners) = create_test_device_and_tuners(
        caps.device_id.clone(),
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
        vec![],
    );

    {
        let mut devices = device_entities.lock().unwrap();
        devices.insert(device);
    }

    {
        let mut tuner_world = tuner_entities.lock().unwrap();
        for tuner in tuners {
            tuner_world.insert(tuner);
        }
    }

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire first tuner");

    let tuner2 = pool
        .try_acquire(&requirements, TunerActivity::Scanning)
        .expect("Failed to acquire second tuner");

    drop(tuner1);

    thread::sleep(Duration::from_millis(100));

    drop(tuner2);
    pool.shutdown();
}

#[test]
fn test_concurrent_streams_independent() {
    let tuner_entities = Arc::new(Mutex::new(EntityWorld::new()));
    let device_entities = Arc::new(Mutex::new(EntityWorld::new()));

    let pool = Pool::with_entity_worlds(
        PoolFilter::allow_all(),
        None,
        tuner_entities.clone(),
        device_entities.clone(),
    );

    let mut caps = Capabilities::for_mock("mock", "dual003");
    caps.channels = 2;

    let (device, tuners) = create_test_device_and_tuners(
        caps.device_id.clone(),
        caps,
        scanner::hardware::types::Backend::Mock,
        None,
        vec![],
    );

    {
        let mut devices = device_entities.lock().unwrap();
        devices.insert(device);
    }

    {
        let mut tuner_world = tuner_entities.lock().unwrap();
        for tuner in tuners {
            tuner_world.insert(tuner);
        }
    }

    let requirements1 = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let requirements2 = TaskRequirements {
        frequency_hz: 101.5e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2.4e6,
        priority: TaskPriority::Normal,
    };

    let tuner1 = pool
        .try_acquire(&requirements1, TunerActivity::Scanning)
        .expect("Failed to acquire first tuner");

    let tuner2 = pool
        .try_acquire(&requirements2, TunerActivity::Listening)
        .expect("Failed to acquire second tuner");

    thread::sleep(Duration::from_millis(200));

    drop(tuner1);
    drop(tuner2);
    pool.shutdown();
}
