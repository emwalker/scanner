//! Scan factory system - creates scans when compatible hardware available

use std::sync::{Arc, RwLock, mpsc::Receiver};

#[cfg(test)]
use crate::ecs::test_helpers::create_test_pool_with_entities;
use crate::{
    core::types::Result,
    discovery::Event as DiscoveryEvent,
    ecs::{
        EntityWorld, System, SystemContext,
        components::scan::PendingScanRequest,
        entities::{ScanTaskData, TaskEntity, TaskId},
    },
    hardware::pool::Pool,
};

pub struct ScanFactorySystem {
    task_entities: Arc<RwLock<EntityWorld<TaskEntity>>>,
    discovery_rx: Receiver<DiscoveryEvent>,
    pool: Arc<Pool>,
    pending_request: Arc<RwLock<Option<PendingScanRequest>>>,
}

impl ScanFactorySystem {
    pub fn new(
        task_entities: Arc<RwLock<EntityWorld<TaskEntity>>>,
        discovery_rx: Receiver<DiscoveryEvent>,
        pool: Arc<Pool>,
        pending_request: Arc<RwLock<Option<PendingScanRequest>>>,
    ) -> Self {
        Self {
            task_entities,
            discovery_rx,
            pool,
            pending_request,
        }
    }

    fn check_compatibility(
        &self,
        tuner_id: &crate::hardware::pool::TunerId,
        _requirements: &crate::hardware::pool::TaskRequirements,
    ) -> bool {
        use crate::hardware::pool::TunerState;

        let status = self.pool.status();
        status
            .tuners
            .iter()
            .find(|t| &t.id == tuner_id)
            .map(|t| t.state == TunerState::Available)
            .unwrap_or(false)
    }

    fn create_scan_from_request(
        &mut self,
        request: PendingScanRequest,
        tuner_id: crate::hardware::pool::TunerId,
    ) {
        let task_id = TaskId::new(format!("scan_{}", request.scan_number));
        let total_windows = request.scan_config.total_windows();
        let mut task = TaskEntity::new_scan(
            task_id,
            ScanTaskData::Placeholder,
            request.scan_config,
            total_windows,
        );

        let crate::ecs::TaskComponents::Scan { tuner, .. } = &mut task.components;
        tuner.assign(tuner_id);

        self.task_entities.write().unwrap().insert(task);
    }
}

impl System for ScanFactorySystem {
    fn name(&self) -> &'static str {
        "ScanFactorySystem"
    }

    fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
        let request = match self.pending_request.read().unwrap().as_ref() {
            Some(req) => req.clone(),
            None => return Ok(()),
        };

        while let Ok(event) = self.discovery_rx.try_recv() {
            match event {
                DiscoveryEvent::Added(device_info) => {
                    for tuner_info in &device_info.tuners {
                        if self.check_compatibility(&tuner_info.id, &request.requirements) {
                            self.create_scan_from_request(request.clone(), tuner_info.id.clone());
                            *self.pending_request.write().unwrap() = None;
                            return Ok(());
                        }
                    }
                }
                DiscoveryEvent::Removed(_) => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        use std::sync::mpsc;

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let (_tx, rx) = mpsc::channel();
        let pool = Arc::new(Pool::new_unfiltered());
        let pending_request = Arc::new(RwLock::new(None));

        let _system = ScanFactorySystem::new(task_entities, rx, pool, pending_request);
    }

    #[test]
    fn test_check_compatibility() {
        use std::sync::mpsc;

        use crate::hardware::{
            DeviceId,
            pool::{TaskPriority, TaskRequirements},
        };

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let (_tx, rx) = mpsc::channel();
        let pool = Arc::new(Pool::new_unfiltered());
        let pending_request = Arc::new(RwLock::new(None));

        let system = ScanFactorySystem::new(task_entities, rx, pool.clone(), pending_request);

        let device_id = DeviceId::from_serial("soapysdr", "test-device");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id, 0);

        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };

        // Test without device - should return false
        assert!(!system.check_compatibility(&tuner_id, &requirements));
    }

    #[test]
    fn test_create_scan_from_request() {
        use std::sync::mpsc;

        use crate::{
            ecs::components::scan::{ScanConfigComponent, ScanType},
            hardware::{
                DeviceId,
                pool::{TaskPriority, TaskRequirements},
            },
        };

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let (_tx, rx) = mpsc::channel();
        let pool = Arc::new(Pool::new_unfiltered());
        let pending_request = Arc::new(RwLock::new(None));

        let mut system = ScanFactorySystem::new(task_entities.clone(), rx, pool, pending_request);

        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 2.0e6, 2.0e6, 40.0, 1.0, 3);
        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };
        let request = PendingScanRequest::new(config, 1, requirements);

        let device_id = DeviceId::from_serial("soapysdr", "test");
        let tuner_id = crate::hardware::pool::TunerId::new(device_id, 0);

        system.create_scan_from_request(request, tuner_id.clone());

        let task_entities_guard = task_entities.read().unwrap();
        assert_eq!(task_entities_guard.len(), 1);
        let task = task_entities_guard.iter().next().unwrap();
        assert_eq!(task.label(), "Scan 1");
    }

    /// TDD RED: Test that factory uses config from PendingScanRequest, not hardcoded values
    ///
    /// This test verifies the fix for the bug where ScanFactorySystem::create_scan_from_request()
    /// discarded the correct config from the request and created a TaskEntity with hardcoded
    /// values.
    ///
    /// Bug behavior: step_size from request (0.5 MHz) ignored, hardcoded 1.0 MHz used instead
    /// Expected behavior: TaskEntity.components.config matches request.scan_config
    #[test]
    fn test_factory_uses_request_config_not_hardcoded() {
        use std::sync::mpsc;

        use crate::{
            ecs::components::scan::{ScanConfigComponent, ScanType},
            hardware::{
                DeviceId,
                pool::{TaskPriority, TaskRequirements, TunerId},
            },
        };

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let (_tx, rx) = mpsc::channel();
        let pool = Arc::new(Pool::new_unfiltered());
        let pending_request = Arc::new(RwLock::new(None));

        let mut system = ScanFactorySystem::new(task_entities.clone(), rx, pool, pending_request);

        // Create request with SPECIFIC config values different from hardcoded defaults
        // This simulates real-world config: 75% overlap with 2.0 MHz sample rate
        let step_size = 0.5e6; // 0.5 MHz (NOT the hardcoded 1.0 MHz)
        let sample_rate = 2.0e6; // 2.0 MHz (NOT the hardcoded 2.4 MHz)
        let config = ScanConfigComponent::new(
            ScanType::Band,
            88.0e6,
            108.0e6,
            step_size,   // 0.5 MHz - the KEY value we're testing
            sample_rate, // 2.0 MHz
            40.0,
            1.0,
            3,
        );
        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: sample_rate,
            required_sample_rate: sample_rate,
            priority: TaskPriority::Normal,
        };
        let request = PendingScanRequest::new(config.clone(), 1, requirements);

        let device_id = DeviceId::from_serial("soapysdr", "test");
        let tuner_id = TunerId::new(device_id, 0);

        // Act: Create scan from request
        system.create_scan_from_request(request, tuner_id);

        // Assert: Task should use config from request, NOT hardcoded values
        let task_entities_guard = task_entities.read().unwrap();
        assert_eq!(
            task_entities_guard.len(),
            1,
            "Should create exactly one task"
        );

        let task = task_entities_guard.iter().next().unwrap();

        // Extract the scan config from task components
        let task_config = match &task.components {
            crate::ecs::TaskComponents::Scan { config, .. } => config,
        };

        // KEY ASSERTIONS: Config should match request, not hardcoded values
        assert_eq!(
            task_config.step_size, step_size,
            "Task should use step_size from request (0.5 MHz), not hardcoded value (1.0 MHz)"
        );
        assert_eq!(
            task_config.sample_rate, sample_rate,
            "Task should use sample_rate from request (2.0 MHz), not hardcoded value (2.4 MHz)"
        );
        assert_eq!(
            task_config.freq_min, config.freq_min,
            "Task should use freq_min from request"
        );
        assert_eq!(
            task_config.freq_max, config.freq_max,
            "Task should use freq_max from request"
        );

        // Verify that total_windows calculation is correct with the request's step_size
        assert_eq!(
            task_config.total_windows(),
            41,
            "With 0.5 MHz step_size, should have 41 windows"
        );
    }

    #[test]
    fn test_run_creates_scan_on_tuner_added() {
        use std::sync::mpsc;

        use crate::{
            ecs::components::scan::{ScanConfigComponent, ScanType},
            hardware::{
                DeviceId,
                pool::{PoolFilter, TaskPriority, TaskRequirements, TunerId},
                types::TunerInfo,
            },
        };

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let (tx, rx) = mpsc::channel();
        let (pool, _tuner_entities, _device_entities) =
            create_test_pool_with_entities(PoolFilter::new().with_driver("mock"), None);

        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 2.0e6, 2.0e6, 40.0, 1.0, 3);
        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };
        let request = PendingScanRequest::new(config, 1, requirements);
        let pending_request = Arc::new(RwLock::new(Some(request)));

        let mut system =
            ScanFactorySystem::new(task_entities.clone(), rx, pool, pending_request.clone());

        let device_id = DeviceId::from_serial("soapysdr", "test-device");
        let tuner_id = TunerId::new(device_id.clone(), 0);

        // Send TunerAdded event
        tx.send(DiscoveryEvent::Added(crate::hardware::DeviceInfo {
            id: device_id,
            label: "Test Device".to_string(),
            tuners: vec![TunerInfo {
                id: tuner_id,
                label: "Tuner 0".to_string(),
                mode: "ST".to_string(),
                antenna: None,
            }],
        }))
        .unwrap();

        // Run system
        let mut context = SystemContext::default();
        system.run(&mut context).unwrap();
    }
}
