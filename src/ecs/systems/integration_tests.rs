//! Integration tests for ECS systems with Pool

use crate::ecs::Scheduler;
use crate::ecs::components::Priority;
use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
use crate::ecs::system::{System, SystemContext};
use crate::ecs::systems::{
    AllocationRequest, AllocationSystem, AudioCoordinationSystem, DiscoverySystem,
    ScanRequestProcessorSystem,
};
use crate::ecs::{AudioEntity, EntityWorld, ScanEntity, StationEntity};
use crate::hardware::pool::{Pool, PoolFilter};
use std::sync::{Arc, RwLock};

#[test]
fn test_discovery_system_with_pool() {
    let pool = Pool::new(PoolFilter::allow_all(), None);

    let mut context = SystemContext::new().with_tuner_entities(Arc::clone(&pool.tuner_entities));

    let mut system = DiscoverySystem::new();
    let result = system.run(&mut context);

    assert!(result.is_ok());
}

#[test]
fn test_scheduler_with_multiple_systems() {
    let mut scheduler = Scheduler::new();
    scheduler.add_system(Box::new(DiscoverySystem::new()));
    scheduler.add_system(Box::new(AllocationSystem::new()));

    let pool = Pool::new(PoolFilter::allow_all(), None);
    let mut context = SystemContext::new().with_tuner_entities(Arc::clone(&pool.tuner_entities));

    let result = scheduler.run(&mut context);
    assert!(result.is_ok());
    assert_eq!(scheduler.system_count(), 2);
}

#[test]
fn test_allocation_system_integration() {
    let pool = Pool::new(PoolFilter::allow_all(), None);

    let mut context = SystemContext::new().with_tuner_entities(Arc::clone(&pool.tuner_entities));

    let mut system = AllocationSystem::new();
    system.request_allocation(AllocationRequest {
        requester_id: "test_scan".to_string(),
        frequency_hz: 88_900_000.0,
        sample_rate_hz: 2_000_000.0,
        priority: Priority::Medium,
        for_audio: false,
        filter: None,
        allocated_count: 0,
    });

    let result = system.run(&mut context);
    assert!(result.is_ok());
}

// Phase 4: Integration tests for request component flows

fn create_test_scan(freq_min: f64, freq_max: f64) -> ScanEntity {
    let config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        1.0e6,
        2.0e6,
        40.0,
        0.5,
        10,
    );
    ScanEntity::new(config)
}

#[test]
fn test_scan_request_processor_pause_flow() {
    let mut world = EntityWorld::new();
    let mut scan = create_test_scan(88.0e6, 108.0e6);

    scan.progress.start_window(0);

    // TUI sets pause request
    scan.request_pause(5);
    assert!(scan.pause_request.is_some());
    assert!(scan.is_scanning());

    world.insert(scan);
    let scan_entities = Arc::new(RwLock::new(world));

    // System processes request
    let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = scan_entities.read().unwrap();
    for scan in entities.iter() {
        assert!(scan.pause_request.is_none(), "Request should be cleared");
        assert!(scan.is_paused(), "Scan should be paused");
    }
}

#[test]
fn test_scan_request_processor_resume_flow() {
    let mut world = EntityWorld::new();
    let mut scan = create_test_scan(88.0e6, 108.0e6);

    // Start paused
    scan.progress.pause(5);
    assert!(scan.is_paused());

    // TUI sets resume request
    scan.request_resume(5);
    assert!(scan.resume_request.is_some());

    world.insert(scan);
    let scan_entities = Arc::new(RwLock::new(world));

    // System processes request
    let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = scan_entities.read().unwrap();
    for scan in entities.iter() {
        assert!(scan.resume_request.is_none(), "Request should be cleared");
        assert!(scan.is_scanning(), "Scan should be scanning");
    }
}

#[test]
fn test_audio_coordination_stop_listening_flow() {
    use crate::audio::quality::AudioQuality;
    use crate::core::types::{ModulationType, Signal};
    use std::time::SystemTime;

    let signal = Signal {
        frequency_hz: 88.9e6,
        signal_strength: 0.8,
        bandwidth_hz: 200_000.0,
        modulation: ModulationType::WFM,
        audio_sample_rate: 48000,
        detected_at: SystemTime::now(),
        analysis_duration_ms: 100,
        detection_center_freq: 88.9e6,
        audio_quality: AudioQuality::Good,
    };

    let mut audio_world = EntityWorld::new();
    let mut audio = AudioEntity::new(signal, 88.9e6, None);

    // TUI sets stop_listening request
    audio.request_stop_listening();
    assert!(audio.stop_listening_request.is_some());
    assert!(audio.is_playing());

    audio_world.insert(audio);

    let station_world = EntityWorld::new();
    let audio_entities = Arc::new(RwLock::new(audio_world));
    let station_entities = Arc::new(RwLock::new(station_world));

    // System processes request
    let mut context = SystemContext::new()
        .with_audio_entities(audio_entities.clone())
        .with_station_entities(station_entities);

    let mut system = AudioCoordinationSystem::new();
    let result = system.run(&mut context);
    assert!(result.is_ok());

    // Verify state changed
    let entities = audio_entities.read().unwrap();
    for audio in entities.iter() {
        assert!(
            audio.stop_listening_request.is_none(),
            "Request should be cleared"
        );
        assert!(!audio.is_playing(), "Audio should be stopped");
    }
}

#[test]
fn test_dual_read_both_paths_work() {
    let mut world = EntityWorld::new();
    let mut scan = create_test_scan(88.0e6, 108.0e6);

    // TUI sets request component (ECS path)
    scan.request_pause(5);
    assert!(scan.pause_request.is_some());

    // Old command path would also be sent (but we're just testing ECS here)

    world.insert(scan);
    let scan_entities = Arc::new(RwLock::new(world));

    // System processes ECS request
    let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());
    let mut system = ScanRequestProcessorSystem::new();

    system.run(&mut context).unwrap();

    // Verify ECS path worked
    let entities = scan_entities.read().unwrap();
    for scan in entities.iter() {
        assert!(scan.is_paused(), "ECS path should pause scan");
    }
}

#[test]
fn test_scheduler_with_request_processor() {
    let mut scheduler = Scheduler::new();
    scheduler.add_system(Box::new(ScanRequestProcessorSystem::new()));
    scheduler.add_system(Box::new(AudioCoordinationSystem::new()));

    let mut scan_world = EntityWorld::new();
    let mut scan = create_test_scan(88.0e6, 108.0e6);
    scan.request_pause(5);
    scan_world.insert(scan);

    let scan_entities = Arc::new(RwLock::new(scan_world));
    let station_entities = Arc::new(RwLock::new(EntityWorld::<StationEntity>::new()));
    let audio_entities = Arc::new(RwLock::new(EntityWorld::<AudioEntity>::new()));

    let mut context = SystemContext::new()
        .with_scan_entities(scan_entities.clone())
        .with_station_entities(station_entities)
        .with_audio_entities(audio_entities);

    let result = scheduler.run(&mut context);
    assert!(result.is_ok());

    let entities = scan_entities.read().unwrap();
    for scan in entities.iter() {
        assert!(scan.is_paused(), "Request should be processed by scheduler");
    }
}
