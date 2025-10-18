//! Tuner allocation system

use crate::core::types::{Result, ScannerError};
use crate::ecs::Entity;
use crate::ecs::components::Priority;
use crate::ecs::system::{System, SystemContext};
use crate::hardware::pool::TunerId;
use tracing::debug;

/// System that handles tuner allocation based on priorities and constraints
///
/// This system matches allocation requests against available tuners,
/// considering:
/// - Tuner availability (connected, not already allocated)
/// - Priority settings (audio vs scanning)
/// - Constraint filters (frequency ranges, sample rates)
pub struct AllocationSystem {
    pending_requests: Vec<AllocationRequest>,
}

#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub requester_id: String,
    pub frequency_hz: f64,
    pub sample_rate_hz: f64,
    pub priority: Priority,
    pub for_audio: bool,
    pub filter: Option<std::sync::Arc<crate::hardware::pool::PoolFilter>>,
    pub allocated_count: usize,
}

impl Default for AllocationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationSystem {
    pub fn new() -> Self {
        Self {
            pending_requests: Vec::new(),
        }
    }

    pub fn request_allocation(&mut self, request: AllocationRequest) {
        debug!(
            requester_id = %request.requester_id,
            frequency_hz = request.frequency_hz,
            priority = ?request.priority,
            for_audio = request.for_audio,
            "Adding allocation request"
        );
        self.pending_requests.push(request);
    }
}

impl System for AllocationSystem {
    fn name(&self) -> &'static str {
        "TunerAllocation"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        if self.pending_requests.is_empty() {
            return Ok(());
        }

        let tuner_entities = match &context.tuner_entities {
            Some(entities) => entities.clone(),
            None => {
                return Err(ScannerError::Custom(
                    "No tuner entities in context".to_string(),
                ));
            }
        };

        let mut successfully_allocated = Vec::new();

        for request in &self.pending_requests {
            let tuner_id = {
                let entities = match tuner_entities.try_lock() {
                    Ok(entities) => entities,
                    Err(_) => return Ok(()),
                };
                let mut best_tuner: Option<TunerId> = None;

                for entity in entities.iter() {
                    if !entity.is_available() {
                        continue;
                    }

                    if let Some(ref filter) = request.filter
                        && !filter.is_allowed(
                            entity.id(),
                            &entity.device.backend,
                            request.allocated_count,
                        )
                    {
                        continue;
                    }

                    let allows_activity = if request.for_audio {
                        entity.priorities.allows_audio()
                    } else {
                        entity.priorities.allows_scanning()
                    };

                    if !allows_activity {
                        continue;
                    }

                    if !entity
                        .constraints
                        .allows_frequency_and_rate(request.frequency_hz, request.sample_rate_hz)
                    {
                        continue;
                    }

                    if !entity
                        .device
                        .capabilities
                        .supports_frequency(request.frequency_hz)
                    {
                        continue;
                    }

                    if !entity
                        .device
                        .capabilities
                        .supports_sample_rate(request.sample_rate_hz)
                    {
                        continue;
                    }

                    best_tuner = Some(entity.id().clone());
                    break;
                }

                best_tuner
            };

            if let Some(tuner_id) = tuner_id {
                let mut entities = match tuner_entities.try_lock() {
                    Ok(entities) => entities,
                    Err(_) => return Ok(()),
                };
                if let Some(entity) = entities.get_mut(&tuner_id) {
                    entity.allocation.allocate(request.requester_id.clone());
                    if request.for_audio {
                        entity.status.start_listening();
                    } else {
                        entity.status.start_scanning();
                    }
                    successfully_allocated.push(request.requester_id.clone());
                    debug!(
                        tuner_id = ?tuner_id,
                        requester_id = %request.requester_id,
                        "Allocated tuner"
                    );
                }
            }
        }

        self.pending_requests
            .retain(|req| !successfully_allocated.contains(&req.requester_id));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::EntityWorld;
    use crate::ecs::TunerEntity;
    use crate::hardware::{Capabilities, DeviceId};
    use std::sync::{Arc, Mutex};

    fn create_test_entity(device_serial: &str, channel: usize) -> TunerEntity {
        let device_id = DeviceId::from_serial("sdrplay", device_serial);
        let capabilities = Capabilities::for_device(&device_id);

        TunerEntity::new(
            device_id,
            channel,
            capabilities,
            crate::hardware::types::Backend::Soapy,
        )
    }

    #[test]
    fn test_allocation_system_with_no_requests() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_entity("12345", 0));

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allocation_system_finds_available_tuner() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_entity("12345", 0));
        world.insert(create_test_entity("12345", 1));

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities.clone());

        system.request_allocation(AllocationRequest {
            requester_id: "scan_1".to_string(),
            frequency_hz: 88_900_000.0,
            sample_rate_hz: 2_000_000.0,
            priority: Priority::Medium,
            for_audio: false,
            filter: None,
            allocated_count: 0,
        });

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.pending_requests.len(), 0);

        let entities = context_entities.lock().unwrap();
        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();
        assert_eq!(allocated_count, 1);
    }

    #[test]
    fn test_allocation_system_skips_allocated_tuners() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        let mut entity1 = create_test_entity("12345", 0);
        entity1.allocation.allocate("existing_scan".to_string());
        world.insert(entity1);
        world.insert(create_test_entity("12345", 1));

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities.clone());

        system.request_allocation(AllocationRequest {
            requester_id: "scan_1".to_string(),
            frequency_hz: 88_900_000.0,
            sample_rate_hz: 2_000_000.0,
            priority: Priority::Medium,
            for_audio: false,
            filter: None,
            allocated_count: 0,
        });

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.lock().unwrap();
        let allocated_to_existing = entities
            .iter()
            .any(|e| e.allocation.allocated_to == Some("existing_scan".to_string()));
        let allocated_to_scan1 = entities
            .iter()
            .any(|e| e.allocation.allocated_to == Some("scan_1".to_string()));

        assert!(
            allocated_to_existing,
            "First entity should still be allocated to existing_scan"
        );
        assert!(
            allocated_to_scan1,
            "Second entity should be allocated to scan_1"
        );
    }

    #[test]
    fn test_allocation_system_respects_priority_settings() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        let mut entity = create_test_entity("12345", 0);
        entity.priorities.set_scanning_priority(Priority::None);
        world.insert(entity);

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities.clone());

        system.request_allocation(AllocationRequest {
            requester_id: "scan_1".to_string(),
            frequency_hz: 88_900_000.0,
            sample_rate_hz: 2_000_000.0,
            priority: Priority::Medium,
            for_audio: false,
            filter: None,
            allocated_count: 0,
        });

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.pending_requests.len(), 1);

        let entities = context_entities.lock().unwrap();
        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();
        assert_eq!(allocated_count, 0);
    }

    #[test]
    fn test_allocation_system_respects_constraints() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        let mut entity = create_test_entity("12345", 0);
        entity
            .constraints
            .set_allowed_freq_range(90_000_000.0..100_000_000.0);
        world.insert(entity);

        let context_entities = Arc::new(Mutex::new(world));
        let mut context = SystemContext::new().with_tuner_entities(context_entities.clone());

        system.request_allocation(AllocationRequest {
            requester_id: "scan_1".to_string(),
            frequency_hz: 88_900_000.0,
            sample_rate_hz: 2_000_000.0,
            priority: Priority::Medium,
            for_audio: false,
            filter: None,
            allocated_count: 0,
        });

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.pending_requests.len(), 1);

        let entities = context_entities.lock().unwrap();
        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();
        assert_eq!(allocated_count, 0);
    }
}
