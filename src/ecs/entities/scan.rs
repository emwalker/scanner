//! Scan entity - represents an active scan operation

use crate::ecs::Entity;
use crate::ecs::components::scan::{
    ScanConfigComponent, ScanId, ScanLifecycleComponent, ScanProgressComponent,
    ScanResultsComponent, ScanType,
};

/// Entity representing an active scan operation
///
/// A scan entity combines configuration, progress tracking, results, and
/// lifecycle information into a single cohesive unit that can be managed
/// by ECS systems.
#[derive(Debug, Clone)]
pub struct ScanEntity {
    /// Unique identifier for this scan
    id: ScanId,

    /// Scan configuration (frequencies, window size, etc.)
    pub config: ScanConfigComponent,

    /// Progress tracking (current window, pause state)
    pub progress: ScanProgressComponent,

    /// Results (candidates, stations discovered)
    pub results: ScanResultsComponent,

    /// Lifecycle timestamps
    pub lifecycle: ScanLifecycleComponent,

    /// Coordinator guidance: worker should pause (advisory)
    pub should_pause: bool,

    /// Coordinator guidance: worker should complete (advisory)
    pub should_complete: bool,
}

impl ScanEntity {
    /// Create a new scan entity from configuration
    pub fn new(config: ScanConfigComponent) -> Self {
        let total_windows = config.total_windows();

        Self {
            id: ScanId::new(),
            config,
            progress: ScanProgressComponent::new(total_windows),
            results: ScanResultsComponent::new(),
            lifecycle: ScanLifecycleComponent::new(),
            should_pause: false,
            should_complete: false,
        }
    }

    /// Check if scan is paused
    pub fn is_paused(&self) -> bool {
        self.progress.is_paused()
    }

    /// Check if scan is completed
    pub fn is_completed(&self) -> bool {
        self.progress.is_completed()
    }

    /// Check if scan is actively scanning
    pub fn is_scanning(&self) -> bool {
        self.progress.is_scanning()
    }

    /// Check if currently listening
    pub fn is_listening(&self) -> bool {
        self.progress.is_listening()
    }

    /// Get current window index
    pub fn current_window(&self) -> usize {
        self.progress.current_window
    }

    /// Get progress percentage (0.0 to 1.0)
    pub fn progress_percentage(&self) -> f64 {
        self.progress.progress_percentage()
    }

    /// Get scan type
    pub fn scan_type(&self) -> ScanType {
        self.config.scan_type
    }
}

impl Entity for ScanEntity {
    type Id = ScanId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::EntityWorld;
    use proptest::prelude::*;

    fn create_test_scan(freq_min: f64, freq_max: f64) -> ScanEntity {
        let config = ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            1.0e6, // 1 MHz window
            2.4e6, // 2.4 MHz sample rate
            40.0,  // 40 dB gain
            1.0,   // 1 second per window
            3,     // 3 scanning windows
        );
        ScanEntity::new(config)
    }

    fn arb_scan_type() -> impl Strategy<Value = ScanType> {
        prop_oneof![Just(ScanType::Band), Just(ScanType::Stations),]
    }

    fn arb_scan_config() -> impl Strategy<Value = ScanConfigComponent> {
        (
            arb_scan_type(),
            88.0e6..108.0e6f64,
            1.0e6..3.0e6f64,
            2.0e6..3.0e6f64,
            20.0..50.0f64,
            0.5..5.0f64,
            1..10usize,
        )
            .prop_map(
                |(
                    scan_type,
                    freq_min,
                    window_size,
                    sample_rate,
                    gain_db,
                    duration,
                    num_windows,
                )| {
                    let freq_max = freq_min + (window_size * num_windows as f64);
                    ScanConfigComponent::new(
                        scan_type,
                        freq_min,
                        freq_max,
                        window_size,
                        sample_rate,
                        gain_db,
                        duration,
                        num_windows,
                    )
                },
            )
    }

    fn arb_scan_entity() -> impl Strategy<Value = ScanEntity> {
        (arb_scan_config(), 0..100usize, any::<bool>(), any::<bool>()).prop_map(
            |(config, windows_completed, paused, completed)| {
                let mut entity = ScanEntity::new(config);
                let target_windows = if completed {
                    entity.progress.total_windows
                } else {
                    windows_completed.min(entity.progress.total_windows)
                };
                for _ in 0..target_windows {
                    entity
                        .progress
                        .start_window(entity.progress.windows_completed);
                    entity.progress.complete_window();
                }
                if paused && !completed {
                    entity.progress.pause(entity.progress.windows_completed);
                }
                if completed {
                    entity.progress.mark_complete();
                }
                entity
            },
        )
    }

    #[test]
    fn test_scan_entity_creation() {
        let scan = create_test_scan(88.0e6, 98.0e6);

        assert_eq!(scan.config.freq_min, 88.0e6);
        assert_eq!(scan.config.freq_max, 98.0e6);
        assert_eq!(scan.config.scan_type, ScanType::Band);
        assert_eq!(scan.progress.total_windows, 10);
        assert!(scan.is_scanning());
        assert!(!scan.is_paused());
        assert!(!scan.is_completed());
    }

    #[test]
    fn test_scan_entity_trait_implementation() {
        let scan = create_test_scan(88.0e6, 98.0e6);
        let id = scan.id();
        assert!(id.value() > 0);
    }

    #[test]
    fn test_scan_convenience_methods() {
        let mut scan = create_test_scan(88.0e6, 98.0e6);

        assert!(scan.is_scanning());
        assert_eq!(scan.current_window(), 0);
        assert_eq!(scan.progress_percentage(), 0.0);
        assert_eq!(scan.scan_type(), ScanType::Band);

        scan.progress.pause(5);
        assert!(scan.is_paused());
        assert!(!scan.is_scanning());

        scan.progress.start_listening(5);
        assert!(scan.is_listening());

        scan.progress.mark_complete();
        assert!(scan.is_completed());
    }

    #[test]
    fn test_scan_progress_tracking() {
        let mut scan = create_test_scan(88.0e6, 98.0e6);

        scan.progress.start_window(0);
        assert_eq!(scan.current_window(), 0);

        scan.progress.complete_window();
        assert_eq!(scan.progress.windows_completed, 1);
        assert_eq!(scan.progress_percentage(), 0.1);

        scan.progress.start_window(1);
        scan.progress.complete_window();
        assert_eq!(scan.progress.windows_completed, 2);
        assert_eq!(scan.progress_percentage(), 0.2);
    }

    #[test]
    fn test_scan_entity_in_world() {
        let mut world = EntityWorld::new();
        let scan = create_test_scan(88.0e6, 98.0e6);
        let id = *scan.id();

        world.insert(scan);
        assert_eq!(world.len(), 1);

        let retrieved = world.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().config.freq_min, 88.0e6);
    }

    #[test]
    fn test_multiple_scans_in_world() {
        let mut world = EntityWorld::new();

        let scan1 = create_test_scan(88.0e6, 98.0e6);
        let scan2 = create_test_scan(98.0e6, 108.0e6);

        let id1 = *scan1.id();
        let id2 = *scan2.id();

        world.insert(scan1);
        world.insert(scan2);

        assert_eq!(world.len(), 2);
        assert!(world.get(&id1).is_some());
        assert!(world.get(&id2).is_some());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_modify_scan_in_world() {
        let mut world = EntityWorld::new();
        let scan = create_test_scan(88.0e6, 98.0e6);
        let id = *scan.id();

        world.insert(scan);

        {
            let scan_mut = world.get_mut(&id).unwrap();
            scan_mut.progress.pause(5);
            scan_mut.results.add_station();
        }

        let scan = world.get(&id).unwrap();
        assert!(scan.is_paused());
        assert_eq!(scan.results.stations_discovered, 1);
    }

    #[test]
    fn test_iterate_active_scans() {
        let mut world = EntityWorld::new();

        let mut scan1 = create_test_scan(88.0e6, 98.0e6);
        let scan2 = create_test_scan(98.0e6, 108.0e6);
        let mut scan3 = create_test_scan(108.0e6, 118.0e6);

        scan1.progress.mark_complete();
        scan3.progress.pause(5);

        world.insert(scan1);
        world.insert(scan2);
        world.insert(scan3);

        let active_count = world.iter().filter(|s| s.is_scanning()).count();
        assert_eq!(active_count, 1);

        let paused_count = world.iter().filter(|s| s.is_paused()).count();
        assert_eq!(paused_count, 1);

        let completed_count = world.iter().filter(|s| s.is_completed()).count();
        assert_eq!(completed_count, 1);
    }

    #[test]
    fn test_scan_lifecycle_integration() {
        let mut scan = create_test_scan(88.0e6, 98.0e6);

        assert!(!scan.lifecycle.is_started());
        scan.lifecycle.start();
        assert!(scan.lifecycle.is_started());

        scan.lifecycle.pause();
        assert_eq!(scan.lifecycle.pause_count(), 1);

        scan.progress.mark_complete();
        scan.lifecycle.complete();
        assert!(scan.lifecycle.is_completed());
        assert!(scan.is_completed());
    }

    #[test]
    fn test_scan_results_tracking() {
        let mut scan = create_test_scan(88.0e6, 98.0e6);

        scan.results.add_candidate();
        scan.results.add_candidate();
        scan.results.reject_candidate();
        scan.results.add_station();

        assert_eq!(scan.results.total_candidates(), 3);
        assert_eq!(scan.results.stations_discovered, 1);
    }

    proptest! {
        #[test]
        fn prop_progress_bounds(scan in arb_scan_entity()) {
            prop_assert!(scan.progress.current_window <= scan.progress.total_windows);
            prop_assert!(scan.progress.windows_completed <= scan.progress.total_windows);
        }

        #[test]
        fn prop_percentage_bounds(scan in arb_scan_entity()) {
            let percentage = scan.progress_percentage();
            prop_assert!(percentage >= 0.0);
            prop_assert!(percentage <= 1.0);
        }

        #[test]
        fn prop_completion_consistency(scan in arb_scan_entity()) {
            if scan.is_completed() {
                prop_assert_eq!(scan.progress.windows_completed, scan.progress.total_windows);
            }
        }

        #[test]
        fn prop_percentage_calculation(scan in arb_scan_entity()) {
            let expected = if scan.progress.total_windows == 0 {
                1.0
            } else {
                scan.progress.windows_completed as f64 / scan.progress.total_windows as f64
            };
            prop_assert!((scan.progress_percentage() - expected).abs() < 1e-6);
        }

        #[test]
        fn prop_state_exclusivity(scan in arb_scan_entity()) {
            let states = [scan.is_scanning(),
                scan.is_paused(),
                scan.is_completed(),
                scan.is_listening()];
            let active_states = states.iter().filter(|&&s| s).count();
            prop_assert!(active_states == 1, "Expected exactly one state to be active, got {}", active_states);
        }
    }
}
