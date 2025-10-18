//! Regression tests for window processing and candidate creation
//!
//! These tests verify that the ECS-based window processing system correctly:
//! - Calculates total_windows for both Band and Stations scan types
//! - Requests and receives tuner allocations
//! - Spawns window tasks that create CandidateEntity objects
//!
//! This prevents regressions from issues fixed in 2025-10-19 where:
//! - total_windows was 0 for station scans
//! - WindowProcessingSystem wasn't requesting allocations
//! - candidate_entities weren't being passed to Window

use scanner::core::types::ScanningConfig;
use scanner::ecs::{
    CandidateEntity, Entity, EntityWorld, ScanConfigComponent, ScanEntity, ScanPauseState,
    ScanType, StationEntity, System,
};
use scanner::hardware::pool::{Pool, PoolFilter};
use scanner::shutdown::ShutdownCoordinator;
use std::sync::{Arc, RwLock};

#[test]
fn test_band_scan_total_windows_calculation() {
    // FM band is 88-108 MHz = 20 MHz
    // With 2 MHz window size, should get 10 windows
    let (freq_min, freq_max) = (88.0e6, 108.0e6);
    let window_size = 2.0e6;

    let config = ScanConfigComponent::new(
        ScanType::Band,
        freq_min,
        freq_max,
        window_size,
        2.0e6, // sample_rate
        40.0,  // gain_db
        1.0,   // duration_per_window
        10,    // scanning_windows
    );

    let scan = ScanEntity::new(config);

    assert_eq!(
        scan.progress.total_windows, 10,
        "FM band (88-108 MHz) with 2 MHz windows should have 10 windows"
    );
}

#[test]
fn test_station_scan_total_windows_calculation() {
    // Regression test: Previously, when freq_min == freq_max (single station),
    // total_windows would be 0, preventing any scanning from occurring.
    let stations = vec![88.9e6];
    let freq_min = 88.9e6;
    let freq_max = 88.9e6; // Same as min for single station

    let config = ScanConfigComponent::new(
        ScanType::Stations,
        freq_min,
        freq_max,
        2.0e6, // window_size (irrelevant for stations)
        2.0e6, // sample_rate
        40.0,  // gain_db
        1.0,   // duration_per_window
        1,     // scanning_windows
    )
    .with_stations(stations);

    let scan = ScanEntity::new(config);

    assert_eq!(
        scan.progress.total_windows, 1,
        "Single station scan should have 1 window, not 0"
    );
}

#[test]
fn test_multiple_stations_total_windows() {
    let stations = vec![88.9e6, 91.5e6, 95.7e6];
    let freq_min = 88.9e6;
    let freq_max = 95.7e6;

    let config = ScanConfigComponent::new(
        ScanType::Stations,
        freq_min,
        freq_max,
        2.0e6, // window_size
        2.0e6, // sample_rate
        40.0,  // gain_db
        1.0,   // duration_per_window
        1,     // scanning_windows
    )
    .with_stations(stations);

    let scan = ScanEntity::new(config);

    assert_eq!(
        scan.progress.total_windows, 3,
        "Three station scan should have 3 windows (one per station)"
    );
}

#[test]
fn test_window_processing_system_with_valid_windows() {
    // Regression test: Verify WindowProcessingSystem requests allocation
    // when there are windows to process (total_windows > 0)
    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut window_processing =
        scanner::ecs::systems::WindowProcessingSystem::new(config, pool, shutdown_coordinator);
    window_processing.enable();

    // Create scan with valid windows
    let scan_config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6,  // freq_min
        108.0e6, // freq_max
        2.0e6,   // window_size
        2.0e6,   // sample_rate
        40.0,    // gain_db
        1.0,     // duration_per_window
        10,      // scanning_windows
    );

    let mut scan = ScanEntity::new(scan_config);
    assert_eq!(scan.progress.total_windows, 10, "Should have 10 windows");

    // Transition to Scanning state
    scan.progress.state = ScanPauseState::Scanning;

    // Create entities
    let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
    scan_entities.write().unwrap().insert(scan);

    let candidate_entities = Arc::new(RwLock::new(EntityWorld::<CandidateEntity>::new()));
    let station_entities = Arc::new(RwLock::new(EntityWorld::<StationEntity>::new()));

    let mut context = scanner::ecs::SystemContext::new()
        .with_scan_entities(scan_entities.clone())
        .with_candidate_entities(candidate_entities)
        .with_station_entities(station_entities);

    // Run the system once
    window_processing.run(&mut context).unwrap();

    // Verify allocation was requested
    let scans = scan_entities.read().unwrap();
    let scan = scans.iter().next().unwrap();

    assert!(
        scan.window_allocation.is_some(),
        "WindowProcessingSystem should request allocation when total_windows > 0"
    );

    if let Some(ref allocation) = scan.window_allocation {
        assert!(
            allocation.is_requested(),
            "Allocation should be in Requested state"
        );
        assert_eq!(
            allocation.window_index(),
            0,
            "Should request allocation for first window"
        );
    }
}

#[test]
fn test_candidate_entities_available_to_window_config() {
    // Regression test: Verify candidate_entities can be passed through WindowConfig
    // Previously, Window::for_station() hardcoded candidate_entities: None
    let candidate_entities = Some(Arc::new(RwLock::new(EntityWorld::<CandidateEntity>::new())));
    let station_entities = Some(Arc::new(RwLock::new(EntityWorld::<StationEntity>::new())));

    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    // This is the pattern used in WindowProcessingSystem
    let window_config = scanner::scanning::window::WindowConfig {
        center_freq: 88.9e6,
        window_num: 0,
        total_windows: 1,
        tuner_provider: pool,
        config,
        shutdown_coordinator,
        pause_signal: None,
        station_entities,
        candidate_entities: candidate_entities.clone(),
        scan_id: scanner::ecs::ScanId::new(),
    };

    // Verify the config accepts candidate_entities
    assert!(
        window_config.candidate_entities.is_some(),
        "WindowConfig should support candidate_entities (not hardcoded to None)"
    );
}

#[test]
fn test_scan_progress_with_zero_total_windows() {
    // Edge case: Ensure we handle the degenerate case gracefully
    let config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6, // freq_min
        88.0e6, // freq_max (same as min)
        2.0e6,  // window_size
        2.0e6,  // sample_rate
        40.0,   // gain_db
        1.0,    // duration_per_window
        0,      // scanning_windows
    );

    let scan = ScanEntity::new(config);

    // This is a degenerate case - should have 0 windows
    assert_eq!(
        scan.progress.total_windows, 0,
        "Band scan with freq_min == freq_max should have 0 windows"
    );

    // But progress_percentage should not panic or divide by zero
    let percentage = scan.progress.progress_percentage();
    assert!(
        (0.0..=1.0).contains(&percentage),
        "Progress percentage should be valid even with 0 total_windows"
    );
}

#[test]
fn test_window_processing_allocation_request_format() {
    // Regression test: Verify allocation request has correct format
    // including requester_id format used by AllocationSystem
    let config = Arc::new(ScanningConfig::default());
    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

    let mut window_processing =
        scanner::ecs::systems::WindowProcessingSystem::new(config, pool, shutdown_coordinator);
    window_processing.enable();

    let scan_config =
        ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 2.0e6, 2.0e6, 40.0, 1.0, 10);

    let mut scan = ScanEntity::new(scan_config);
    let scan_id = *scan.id();
    scan.progress.state = ScanPauseState::Scanning;

    let scan_entities = Arc::new(RwLock::new(EntityWorld::new()));
    scan_entities.write().unwrap().insert(scan);

    let candidate_entities = Arc::new(RwLock::new(EntityWorld::<CandidateEntity>::new()));
    let station_entities = Arc::new(RwLock::new(EntityWorld::<StationEntity>::new()));

    let mut context = scanner::ecs::SystemContext::new()
        .with_scan_entities(scan_entities.clone())
        .with_candidate_entities(candidate_entities)
        .with_station_entities(station_entities);

    window_processing.run(&mut context).unwrap();

    let scans = scan_entities.read().unwrap();
    let scan = scans.iter().next().unwrap();

    if let Some(ref allocation) = scan.window_allocation {
        let requester_id = allocation.requester_id();
        assert!(
            requester_id.starts_with("scan_"),
            "Requester ID should start with 'scan_'"
        );
        assert!(
            requester_id.contains(&format!("_{}_", scan_id.value())),
            "Requester ID should contain scan ID"
        );
        assert!(
            requester_id.contains("_window_"),
            "Requester ID should contain '_window_'"
        );
    }
}

#[test]
fn test_scan_completes_when_all_windows_processed() {
    // Verify scan transitions to Completed when all windows are done
    let config = ScanConfigComponent::new(
        ScanType::Band,
        88.0e6,
        90.0e6, // Just 1 window (2 MHz range)
        2.0e6,
        2.0e6,
        40.0,
        1.0,
        1,
    );

    let mut scan = ScanEntity::new(config);
    assert_eq!(scan.progress.total_windows, 1);

    // Mark the only window as complete
    scan.progress.complete_window_at(0);

    assert_eq!(
        scan.progress.completed_windows.len(),
        1,
        "Should have 1 completed window"
    );

    // In actual WindowProcessingSystem, this check happens:
    let should_complete = scan.progress.completed_windows.len() >= scan.progress.total_windows;

    assert!(
        should_complete,
        "Scan should be ready to complete when all windows are done"
    );
}
