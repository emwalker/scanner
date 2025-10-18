//! Integration tests for ECS systems with Pool

use crate::ecs::Scheduler;
use crate::ecs::components::Priority;
use crate::ecs::system::{System, SystemContext};
use crate::ecs::systems::{AllocationRequest, AllocationSystem, DiscoverySystem};
use crate::hardware::pool::{Pool, PoolFilter};
use std::sync::Arc;

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
