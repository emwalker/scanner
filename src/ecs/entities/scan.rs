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
}
