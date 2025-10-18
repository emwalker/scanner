//! Scan request processor system - processes pause/resume request components

use crate::core::types::Result;
use crate::ecs::Entity;
use crate::ecs::system::{System, SystemContext};
use tracing::debug;

/// System that processes pause and resume request components on ScanEntity
///
/// This system:
/// - Queries for ScanEntity with pause_request.is_some()
/// - Processes pause requests by updating scan state
/// - Clears pause_request after processing
/// - Queries for ScanEntity with resume_request.is_some()
/// - Processes resume requests by updating scan state
/// - Clears resume_request after processing
pub struct RequestProcessorSystem;

impl Default for RequestProcessorSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestProcessorSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for RequestProcessorSystem {
    fn name(&self) -> &'static str {
        "ScanRequestProcessor"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let scan_entities = match &context.scan_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let mut scans = scan_entities.write().unwrap();

        for scan in scans.iter_mut() {
            // Process pause request component
            if let Some(ref pause_request) = scan.pause_request {
                debug!(
                    scan_id = ?scan.id(),
                    window_num = pause_request.window_num,
                    has_station = pause_request.station_frequency_hz.is_some(),
                    "ScanRequestProcessor: Processing pause request"
                );

                // If pause request includes station info, transition to Listening state
                if let Some(station_freq) = pause_request.station_frequency_hz
                    && let Some(window_center_freq) = pause_request.window_center_frequency_hz
                {
                    scan.progress.start_listening(pause_request.window_num);
                    scan.lifecycle.pause();

                    // Set tune_request on matching StationEntity
                    if let Some(ref station_entities) = context.station_entities {
                        let mut stations = station_entities.write().unwrap();
                        for station in stations.iter_mut() {
                            if (station.frequency() - station_freq).abs() < 1000.0 {
                                station.request_tune(pause_request.window_num, window_center_freq);
                                debug!(
                                    scan_id = ?scan.id(),
                                    station_id = ?station.id(),
                                    station_frequency_mhz = station_freq / 1e6,
                                    "ScanRequestProcessor: Set tune_request, transitioned to Listening"
                                );
                                break;
                            }
                        }
                    }
                } else {
                    // Regular pause without station
                    scan.progress.pause(pause_request.window_num);
                    scan.lifecycle.pause();
                }

                scan.clear_pause_request();
            }

            // Process resume request component
            if let Some(ref resume_request) = scan.resume_request {
                debug!(
                    scan_id = ?scan.id(),
                    window_num = resume_request.window_num,
                    is_listening = scan.progress.is_listening(),
                    "ScanRequestProcessor: Processing resume request"
                );

                // If we were listening, clear the tune request to stop audio
                if scan.progress.is_listening()
                    && let Some(ref station_entities) = context.station_entities
                {
                    let mut stations = station_entities.write().unwrap();
                    for station in stations.iter_mut() {
                        if station.has_tune_request() {
                            station.clear_tune_request();
                            debug!(
                                scan_id = ?scan.id(),
                                station_id = ?station.id(),
                                "ScanRequestProcessor: Cleared tune_request to stop audio"
                            );
                        }
                    }
                }

                scan.progress.resume();
                scan.clear_resume_request();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::{EntityWorld, ScanEntity};
    use std::sync::{Arc, RwLock};

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
    fn test_no_requests() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_scan(88.0e6, 108.0e6));

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(scan.pause_request.is_none());
            assert!(scan.resume_request.is_none());
        }
    }

    #[test]
    fn test_processes_pause_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.start_window(0);
        scan.request_pause(5);
        assert!(scan.pause_request.is_some());
        assert!(scan.is_scanning());

        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(
                scan.pause_request.is_none(),
                "Pause request should be cleared"
            );
            assert!(scan.is_paused(), "Scan should be paused");
        }
    }

    #[test]
    fn test_processes_resume_request() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);
        scan.progress.pause(5);
        assert!(scan.is_paused());

        scan.request_resume(5);
        assert!(scan.resume_request.is_some());

        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(
                scan.resume_request.is_none(),
                "Resume request should be cleared"
            );
            assert!(scan.is_scanning(), "Scan should be scanning");
        }
    }

    #[test]
    fn test_processes_both_requests_in_sequence() {
        let mut system = RequestProcessorSystem::new();

        let mut world = EntityWorld::new();
        let mut scan = create_test_scan(88.0e6, 108.0e6);

        scan.request_pause(3);
        world.insert(scan);

        let scan_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_scan_entities(scan_entities.clone());

        system.run(&mut context).unwrap();

        {
            let entities = scan_entities.read().unwrap();
            for scan in entities.iter() {
                assert!(scan.is_paused());
            }
        }

        {
            let mut entities = scan_entities.write().unwrap();
            for scan in entities.iter_mut() {
                scan.request_resume(3);
            }
        }

        system.run(&mut context).unwrap();

        let entities = scan_entities.read().unwrap();
        for scan in entities.iter() {
            assert!(scan.is_scanning());
        }
    }
}
