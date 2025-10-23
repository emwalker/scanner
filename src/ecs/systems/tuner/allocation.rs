//! Tuner allocation system

use tracing::debug;

use crate::{
    core::types::{Result, ScannerError},
    ecs::{
        Entity,
        components::Priority,
        system::{System, SystemContext},
    },
    hardware::pool::TunerId,
};

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

    #[allow(clippy::cognitive_complexity)]
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        debug!("AllocationSystem: Starting run");

        // First, collect allocation requests from WindowEntity allocation components
        if let Some(ref window_entities) = context.window_entities
            && let Ok(windows) = window_entities.read()
        {
            debug!(
                window_count = windows.len(),
                "AllocationSystem: Checking window entities"
            );
            for window in windows.iter() {
                debug!(
                    window_id = %window.id(),
                    allocation_state = ?window.allocation,
                    "AllocationSystem: Examining window"
                );
                if let crate::ecs::components::window::WindowAllocationComponent::Requested {
                    requirements,
                    activity,
                    requester_id,
                } = &window.allocation
                {
                    // Check if we already have this request
                    if !self
                        .pending_requests
                        .iter()
                        .any(|r| r.requester_id == *requester_id)
                    {
                        debug!(
                            requester_id = %requester_id,
                            frequency_hz = requirements.frequency_hz,
                            "AllocationSystem: Adding window allocation request from WindowEntity"
                        );
                        self.pending_requests.push(AllocationRequest {
                            requester_id: requester_id.clone(),
                            frequency_hz: requirements.frequency_hz,
                            sample_rate_hz: requirements.required_sample_rate,
                            priority: crate::ecs::Priority::Medium,
                            for_audio: matches!(
                                activity,
                                crate::hardware::pool::TunerActivity::Listening
                            ),
                            filter: None,
                            allocated_count: 0,
                        });
                    } else {
                        debug!(
                            requester_id = %requester_id,
                            "AllocationSystem: Request already pending"
                        );
                    }
                }
            }
        } else {
            debug!("AllocationSystem: No window entities available");
        }

        if self.pending_requests.is_empty() {
            debug!("AllocationSystem: No pending requests, exiting early");
            return Ok(());
        }

        debug!(
            pending_count = self.pending_requests.len(),
            "AllocationSystem: Processing pending requests"
        );

        let pool = match &context.pool {
            Some(pool) => pool,
            None => {
                return Err(ScannerError::Custom("No pool in context".to_string()));
            }
        };

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
            debug!(
                requester_id = %request.requester_id,
                frequency_hz = request.frequency_hz,
                for_audio = request.for_audio,
                "AllocationSystem: Processing allocation request"
            );
            let tuner_id = {
                // Get filtered available tuners from Pool
                let pool_status = pool.status();
                let available_tuner_ids: Vec<TunerId> = pool_status
                    .tuners
                    .iter()
                    .filter(|t| t.state == crate::hardware::pool::TunerState::Available)
                    .map(|t| t.id.clone())
                    .collect();

                debug!(
                    available_count = available_tuner_ids.len(),
                    "AllocationSystem: Found available tuners from pool"
                );

                let entities = match tuner_entities.try_lock() {
                    Ok(entities) => entities,
                    Err(_) => {
                        debug!("AllocationSystem: Failed to lock tuner entities");
                        return Ok(());
                    }
                };
                let mut best_tuner: Option<TunerId> = None;

                // Only consider tuners that passed the Pool filter
                for tuner_id in &available_tuner_ids {
                    let entity = match entities.get(tuner_id) {
                        Some(e) => e,
                        None => {
                            debug!(tuner_id = ?tuner_id, "AllocationSystem: Tuner not found in entities");
                            continue;
                        }
                    };

                    if !entity.is_available() {
                        debug!(tuner_id = ?tuner_id, "AllocationSystem: Tuner not available in entity");
                        continue;
                    }

                    let allows_activity = if request.for_audio {
                        entity.priorities.allows_audio()
                    } else {
                        entity.priorities.allows_scanning()
                    };

                    if !allows_activity {
                        debug!(tuner_id = ?tuner_id, for_audio = request.for_audio, "AllocationSystem: Activity not allowed by priorities");
                        continue;
                    }

                    if !entity
                        .constraints
                        .allows_frequency_and_rate(request.frequency_hz, request.sample_rate_hz)
                    {
                        debug!(tuner_id = ?tuner_id, "AllocationSystem: Frequency/rate not allowed by constraints");
                        continue;
                    }

                    if !entity
                        .device
                        .capabilities
                        .supports_frequency(request.frequency_hz)
                    {
                        debug!(tuner_id = ?tuner_id, "AllocationSystem: Frequency not supported by capabilities");
                        continue;
                    }

                    if !entity
                        .device
                        .capabilities
                        .supports_sample_rate(request.sample_rate_hz)
                    {
                        debug!(tuner_id = ?tuner_id, "AllocationSystem: Sample rate not supported by capabilities");
                        continue;
                    }

                    debug!(tuner_id = ?tuner_id, "AllocationSystem: Found suitable tuner");
                    best_tuner = Some(tuner_id.clone());
                    break;
                }

                if best_tuner.is_none() {
                    debug!("AllocationSystem: No suitable tuner found for request");
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

        // Update WindowEntity allocation from Requested to Allocated
        if let Some(ref window_entities) = context.window_entities
            && let Ok(mut windows) = window_entities.write()
        {
            for window in windows.iter_mut() {
                if let crate::ecs::components::window::WindowAllocationComponent::Requested {
                    requester_id,
                    ..
                } = &window.allocation
                {
                    // Check if this request was successfully allocated
                    if successfully_allocated.contains(requester_id) {
                        // Find the allocated tuner_id
                        if let Ok(tuners) = context.tuner_entities.as_ref().unwrap().lock()
                            && let Some(tuner_entity) = tuners
                                .iter()
                                .find(|e| e.allocation.allocated_to.as_ref() == Some(requester_id))
                        {
                            debug!(
                                requester_id = %requester_id,
                                tuner_id = ?tuner_entity.id(),
                                "AllocationSystem: Updating WindowEntity allocation to Allocated"
                            );
                            window.allocation.allocate(tuner_entity.id().clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        ecs::{DeviceEntity, EntityWorld, TunerEntity},
        hardware::{
            Capabilities, DeviceId,
            pool::{Pool, PoolFilter},
        },
    };

    fn create_test_entity(device_serial: &str, channel: usize) -> TunerEntity {
        let device_id = DeviceId::from_serial("sdrplay", device_serial);
        let capabilities = Capabilities::for_device(&device_id);

        TunerEntity::new(
            device_id,
            channel,
            capabilities,
            crate::hardware::types::Backend::Soapy,
            format!("Test Tuner {}", channel),
            None,
            "FM".to_string(),
        )
    }

    fn create_test_pool(tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>) -> Arc<Pool> {
        let device_entities = Arc::new(Mutex::new(EntityWorld::<DeviceEntity>::new()));
        Arc::new(Pool::with_entity_worlds(
            PoolFilter::allow_all(),
            None,
            tuner_entities,
            device_entities,
        ))
    }

    #[test]
    fn test_allocation_system_with_no_requests() {
        let mut system = AllocationSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_entity("12345", 0));

        let context_entities = Arc::new(Mutex::new(world));
        let pool = create_test_pool(context_entities.clone());
        let mut context = SystemContext::new()
            .with_tuner_entities(context_entities)
            .with_pool(pool);

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
        let pool = create_test_pool(context_entities.clone());
        let mut context = SystemContext::new()
            .with_tuner_entities(context_entities.clone())
            .with_pool(pool);

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
        let pool = create_test_pool(context_entities.clone());
        let mut context = SystemContext::new()
            .with_tuner_entities(context_entities.clone())
            .with_pool(pool);

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
        let pool = create_test_pool(context_entities.clone());
        let mut context = SystemContext::new()
            .with_tuner_entities(context_entities.clone())
            .with_pool(pool);

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
        let pool = create_test_pool(context_entities.clone());
        let mut context = SystemContext::new()
            .with_tuner_entities(context_entities.clone())
            .with_pool(pool);

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
