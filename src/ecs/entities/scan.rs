//! Scan entity - represents an active scan operation

use crate::ecs::{
    Entity,
    components::scan::{
        PauseRequestComponent, ResumeRequestComponent, ScanConfigComponent, ScanId,
        ScanLifecycleComponent, ScanProgressComponent, ScanResultsComponent, ScanTunerComponent,
        ScanType,
    },
};

/// Entity representing an active scan operation
///
/// A scan entity combines configuration, progress tracking, results, and
/// lifecycle information into a single cohesive unit that can be managed
/// by ECS systems.
#[derive(Debug)]
pub struct ScanEntity {
    /// Unique identifier for this scan
    id: ScanId,

    /// Human-readable scan number (e.g., 1, 2, 3)
    scan_number: u64,

    /// Scan configuration (frequencies, window size, etc.)
    pub config: ScanConfigComponent,

    /// Progress tracking (current window, pause state)
    pub progress: ScanProgressComponent,

    /// Results
    pub results: ScanResultsComponent,

    /// Lifecycle timestamps
    pub lifecycle: ScanLifecycleComponent,

    /// Tuner assignment
    pub tuner: ScanTunerComponent,

    /// Coordinator guidance: worker should pause (advisory)
    pub should_pause: bool,

    /// Coordinator guidance: worker should complete (advisory)
    pub should_complete: bool,

    /// Request to pause scanning (ECS Phase 1)
    pub pause_request: Option<PauseRequestComponent>,

    /// Request to resume scanning (ECS Phase 1)
    pub resume_request: Option<ResumeRequestComponent>,
}

impl ScanEntity {
    /// Create a new scan entity from configuration
    pub fn new(config: ScanConfigComponent, scan_number: u64) -> Self {
        let total_windows = config.total_windows();

        Self {
            id: ScanId::new(),
            scan_number,
            config,
            progress: ScanProgressComponent::new(total_windows),
            results: ScanResultsComponent::new(),
            lifecycle: ScanLifecycleComponent::new(),
            tuner: ScanTunerComponent::new(),
            should_pause: false,
            should_complete: false,
            pause_request: None,
            resume_request: None,
        }
    }

    /// Get scan number
    pub fn scan_number(&self) -> u64 {
        self.scan_number
    }

    /// Check if scan is pending
    pub fn is_pending(&self) -> bool {
        self.progress.is_pending()
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
    pub fn current_window_index(&self) -> Option<usize> {
        self.progress
            .current_window
            .as_ref()
            .map(|w| w.window_index)
    }

    /// Get progress percentage (0.0 to 1.0)
    pub fn progress_percentage(&self) -> f64 {
        self.progress.progress_percentage()
    }

    /// Get scan type
    pub fn scan_type(&self) -> ScanType {
        self.config.scan_type
    }

    /// Get the frequency currently being listened to, if any
    pub fn listening_frequency(&self) -> Option<f64> {
        // For now, return None - will be populated from results in future
        None
    }

    /// Request pause at current window
    pub fn request_pause(&mut self, window_num: usize) {
        self.pause_request = Some(PauseRequestComponent::new(window_num));
    }

    /// Request pause and tune to a specific station
    pub fn request_pause_with_station(
        &mut self,
        window_num: usize,
        station_frequency_hz: f64,
        window_center_frequency_hz: f64,
    ) {
        self.pause_request = Some(PauseRequestComponent::with_station(
            window_num,
            station_frequency_hz,
            window_center_frequency_hz,
        ));
    }

    /// Clear pause request
    pub fn clear_pause_request(&mut self) {
        self.pause_request = None;
    }

    /// Request resume from paused state
    pub fn request_resume(&mut self, window_num: usize) {
        self.resume_request = Some(ResumeRequestComponent::new(window_num));
    }

    /// Clear resume request
    pub fn clear_resume_request(&mut self) {
        self.resume_request = None;
    }
}

impl Clone for ScanEntity {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            scan_number: self.scan_number,
            config: self.config.clone(),
            progress: self.progress.clone(),
            results: self.results.clone(),
            lifecycle: self.lifecycle.clone(),
            tuner: self.tuner.clone(),
            should_pause: self.should_pause,
            should_complete: self.should_complete,
            pause_request: self.pause_request.clone(),
            resume_request: self.resume_request.clone(),
        }
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
    use proptest::prelude::*;

    use super::*;
    use crate::ecs::{EntityWorld, TaskId, WindowId};

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
        ScanEntity::new(config, 1)
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
                let mut entity = ScanEntity::new(config, 1);
                let task_id = TaskId::new("arb_test".to_string());
                let target_windows = if completed {
                    entity.progress.total_windows
                } else {
                    windows_completed.min(entity.progress.total_windows)
                };
                for _ in 0..target_windows {
                    let window_id =
                        WindowId::new(task_id.clone(), entity.progress.windows_completed);
                    entity.progress.start_window(window_id.clone());
                    entity.progress.complete_window();
                    if paused && !completed {
                        entity.progress.pause(window_id);
                    }
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
        // Range 88-98 MHz with 1 MHz steps: 11 windows (88, 89, ..., 98)
        assert_eq!(scan.progress.total_windows, 11);
        assert!(scan.is_pending());
        assert!(!scan.is_scanning());
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
        let task_id = TaskId::new("test_scan".to_string());

        assert!(scan.is_pending());
        let window_id = WindowId::new(task_id.clone(), 0);
        scan.progress.start_window(window_id);
        assert!(scan.is_scanning());
        assert_eq!(scan.current_window_index(), Some(0));
        assert_eq!(scan.progress_percentage(), 0.0);
        assert_eq!(scan.scan_type(), ScanType::Band);

        let window_id_5 = WindowId::new(task_id.clone(), 5);
        scan.progress.pause(window_id_5.clone());
        assert!(scan.is_paused());
        assert!(!scan.is_scanning());

        scan.progress.start_listening(window_id_5);
        assert!(scan.is_listening());

        scan.progress.mark_complete();
        assert!(scan.is_completed());
    }

    #[test]
    fn test_scan_progress_tracking() {
        let mut scan = create_test_scan(88.0e6, 98.0e6);
        let task_id = TaskId::new("test_scan".to_string());

        let window_id = WindowId::new(task_id.clone(), 0);
        scan.progress.start_window(window_id);
        assert_eq!(scan.current_window_index(), Some(0));

        scan.progress.complete_window();
        assert_eq!(scan.progress.windows_completed, 1);
        // 1 out of 11 windows = ~0.09
        assert!((scan.progress_percentage() - 0.0909).abs() < 0.01);

        let window_id_1 = WindowId::new(task_id.clone(), 1);
        scan.progress.start_window(window_id_1);
        scan.progress.complete_window();
        assert_eq!(scan.progress.windows_completed, 2);
        // 2 out of 11 windows = ~0.18
        assert!((scan.progress_percentage() - 0.1818).abs() < 0.01);
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
            let task_id = TaskId::new("test_scan".to_string());
            let window_id = WindowId::new(task_id, 5);
            scan_mut.progress.pause(window_id);
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
        let mut scan2 = create_test_scan(98.0e6, 108.0e6);
        let mut scan3 = create_test_scan(108.0e6, 118.0e6);

        scan1.progress.mark_complete();

        let task_id = TaskId::new("test_scan".to_string());
        let window_id2 = WindowId::new(task_id.clone(), 0);
        scan2.progress.start_window(window_id2);

        let window_id3 = WindowId::new(task_id, 5);
        scan3.progress.pause(window_id3);

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

        scan.results.add_signal();
        scan.results.add_signal();
        scan.results.reject_signal();
        scan.results.add_station();

        assert_eq!(scan.results.total_signals(), 3);
        assert_eq!(scan.results.stations_discovered, 1);
    }

    proptest! {
        #[test]
        fn prop_progress_bounds(scan in arb_scan_entity()) {
            if let Some(window_id) = &scan.progress.current_window {
                prop_assert!(window_id.window_index <= scan.progress.total_windows);
            }
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
            let states = [scan.is_pending(),
                scan.is_scanning(),
                scan.is_paused(),
                scan.is_completed(),
                scan.is_listening()];
            let active_states = states.iter().filter(|&&s| s).count();
            prop_assert!(active_states == 1, "Expected exactly one state to be active, got {}", active_states);
        }
    }
}
