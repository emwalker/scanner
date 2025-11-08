//! Unified tuner allocation system
//!
//! This system processes the unified TunerAllocationQueue, handling allocation
//! requests from both windows (for scanning) and stations (for playback).
//!
//! All tuner allocation goes through a single FIFO queue, ensuring fair
//! allocation across competing uses.

use tracing::debug;

#[cfg(test)]
use crate::hardware::pool::test_utils::add_test_device_to_pool;
use crate::{
    core::types::Result,
    ecs::{
        Entity, TunerRequester,
        system::{System, SystemContext},
    },
};

/// Unified tuner allocation system
///
/// Processes the TunerAllocationQueue in FIFO order, allocating tuners
/// to windows and stations as they become available.
pub struct TunerAllocationSystem;

impl TunerAllocationSystem {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TunerAllocationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for TunerAllocationSystem {
    fn name(&self) -> &'static str {
        "TunerAllocationSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let allocation_queue = match &context.tuner_allocation_queue {
            Some(q) => q.clone(),
            None => return Ok(()),
        };

        loop {
            // Peek at front of queue without removing
            let request = {
                let queue = allocation_queue.lock().unwrap();
                queue.front().cloned()
            };

            let request = match request {
                Some(r) => r,
                None => break, // Queue empty
            };

            // Process based on requester type
            let processed = match &request.requester {
                TunerRequester::Window(window_id) => {
                    self.process_window_request(window_id, &request, context)
                }
                TunerRequester::Station(station_id) => {
                    self.process_signal_request(station_id, &request, context)
                }
            };

            if processed {
                // Success - remove from queue
                allocation_queue.lock().unwrap().pop_front();
            } else {
                // Failure - leave in queue and stop processing
                // (no tuner available, retry next tick)
                break;
            }
        }

        Ok(())
    }
}

impl TunerAllocationSystem {
    fn process_window_request(
        &self,
        window_id: &crate::ecs::WindowId,
        request: &crate::ecs::TunerAllocationRequest,
        context: &mut SystemContext,
    ) -> bool {
        // Try to acquire tuner from pool
        let pool = match &context.pool {
            Some(p) => p.clone(),
            None => return true, // No pool, remove from queue
        };

        let config = match &context.config {
            Some(c) => c.clone(),
            None => return true, // No config, remove from queue
        };

        let shutdown_coordinator = match &context.shutdown_coordinator {
            Some(s) => s.clone(),
            None => return true, // No shutdown coordinator, remove from queue
        };

        let tuner = match pool.acquire(&request.requirements, request.activity.clone()) {
            Ok(t) => t,
            Err(_) => return false, // Tuner unavailable, retry next tick
        };

        let tuner_id = tuner.id().clone();

        // Create segment to hold tuner (RAII - keeps tuner allocated)
        let segment = match crate::hardware::pool::Segment::from_tuner(
            tuner,
            request.requirements.frequency_hz,
            &config,
            shutdown_coordinator.token(),
            context.global_pause_resource.clone(),
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!(
                    window_id = ?window_id,
                    error = %e,
                    "TunerAllocationSystem: Failed to create segment"
                );
                return false;
            }
        };

        // Update window allocation state: Requested → Allocated
        // Store segment to keep tuner allocated
        if let Some(window_entities) = &context.window_entities
            && let Ok(mut windows) = window_entities.try_write()
            && let Some(window) = windows.get_mut(window_id)
        {
            window.allocation.allocate(tuner_id.clone());
            window.segment = Some(crate::ecs::SegmentComponent::new(std::sync::Arc::new(
                segment,
            )));
            debug!(
                window_id = ?window_id,
                tuner_id = ?tuner_id,
                "TunerAllocationSystem: Allocated tuner and created segment for window"
            );
            return true;
        }

        false
    }

    fn process_signal_request(
        &self,
        station_id: &crate::ecs::StationId,
        request: &crate::ecs::TunerAllocationRequest,
        context: &mut SystemContext,
    ) -> bool {
        // Try to acquire tuner from pool
        let pool = match &context.pool {
            Some(p) => p.clone(),
            None => return true, // No pool, remove from queue
        };

        let config = match &context.config {
            Some(c) => c.clone(),
            None => return true, // No config, remove from queue
        };

        let shutdown_coordinator = match &context.shutdown_coordinator {
            Some(s) => s.clone(),
            None => return true, // No shutdown coordinator, remove from queue
        };

        let tuner = match pool.acquire(&request.requirements, request.activity.clone()) {
            Ok(t) => t,
            Err(_) => return false, // Tuner unavailable, retry next tick
        };

        let tuner_id = tuner.id().clone();

        // Create segment to hold tuner (RAII - keeps tuner allocated)
        let _segment = match crate::hardware::pool::Segment::from_tuner(
            tuner,
            request.requirements.frequency_hz,
            &config,
            shutdown_coordinator.token(),
            context.global_pause_resource.clone(),
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!(
                    signal_id = ?station_id,
                    error = %e,
                    "TunerAllocationSystem: Failed to create segment for signal"
                );
                return false;
            }
        };

        // Update signal TuneState: RequestQueued → Active
        if let Some(signal_entities) = &context.signal_entities
            && let Ok(mut signals) = signal_entities.try_write()
        {
            // Find signal with matching frequency
            let signal_opt = signals.iter_mut().find(|s| {
                const FREQ_TOLERANCE_HZ: f64 = 1000.0;
                (s.frequency() - request.requirements.frequency_hz).abs() < FREQ_TOLERANCE_HZ
            });

            if let Some(signal) = signal_opt {
                // Transition to Active state
                if let crate::ecs::components::station::TuneState::RequestQueued {
                    mut allocation,
                    ..
                } = signal.tune_state.clone()
                {
                    allocation
                        .transition(crate::ecs::components::station::TuneAllocationState::Active);
                    signal.tune_state =
                        crate::ecs::components::station::TuneState::Active { allocation };

                    debug!(
                        signal_id = ?signal.id(),
                        tuner_id = ?tuner_id,
                        "TunerAllocationSystem: Allocated tuner for signal"
                    );

                    // TODO: Store segment for signal playback
                    // For now, segment is dropped here which returns tuner
                    // This is a temporary limitation until audio playback integration

                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex, RwLock},
    };

    use super::*;
    use crate::{
        core::{config::ScanningConfig, signals::ModulationType},
        ecs::{
            Entity, EntityWorld, TaskId, TunerAllocationRequest, TunerRequester, WindowEntity,
            WindowId,
        },
        hardware::pool::{Pool, TaskPriority, TaskRequirements, TunerActivity},
        shutdown::ShutdownCoordinator,
    };

    fn create_test_window(task_id: TaskId, window_index: usize) -> WindowEntity {
        let window_id = WindowId::new(task_id.clone(), window_index);
        WindowEntity::new(window_id, task_id, 88.0e6 + (window_index as f64 * 2.0e6))
    }

    #[test]
    fn test_window_allocation_queued_and_processed_fifo() {
        // Setup: Create 3 windows, 1 tuner available
        let pool = Pool::new_unfiltered();

        // Add one mock tuner
        let device_id = crate::hardware::DeviceId::from_serial("mock", "test-001");
        let caps = crate::hardware::Capabilities::for_mock("mock", "test-001");
        add_test_device_to_pool(
            &pool,
            device_id,
            caps,
            crate::hardware::types::Backend::Mock,
            None,
        );

        let task_id = TaskId::new("test-scan");
        let window_0_id = WindowId::new(task_id.clone(), 0);
        let window_1_id = WindowId::new(task_id.clone(), 1);
        let window_2_id = WindowId::new(task_id.clone(), 2);

        // Create windows
        let mut window_0 = create_test_window(task_id.clone(), 0);
        let mut window_1 = create_test_window(task_id.clone(), 1);
        let mut window_2 = create_test_window(task_id.clone(), 2);

        // Set windows to Requested state
        let requirements_0 = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };
        let requirements_1 = TaskRequirements {
            frequency_hz: 90.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };
        let requirements_2 = TaskRequirements {
            frequency_hz: 92.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };

        window_0.allocation.request(
            requirements_0.clone(),
            TunerActivity::Scanning,
            "window_0".to_string(),
        );
        window_1.allocation.request(
            requirements_1.clone(),
            TunerActivity::Scanning,
            "window_1".to_string(),
        );
        window_2.allocation.request(
            requirements_2.clone(),
            TunerActivity::Scanning,
            "window_2".to_string(),
        );

        // Store windows
        let mut window_world = EntityWorld::new();
        window_world.insert(window_0);
        window_world.insert(window_1);
        window_world.insert(window_2);
        let window_entities = Arc::new(RwLock::new(window_world));

        // Queue requests for window 0, 1, 2 (FIFO order)
        let allocation_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = allocation_queue.lock().unwrap();
            queue.push_back(TunerAllocationRequest {
                requester: TunerRequester::Window(window_0_id.clone()),
                requirements: requirements_0,
                activity: TunerActivity::Scanning,
                requester_id: "window_0".to_string(),
            });
            queue.push_back(TunerAllocationRequest {
                requester: TunerRequester::Window(window_1_id.clone()),
                requirements: requirements_1,
                activity: TunerActivity::Scanning,
                requester_id: "window_1".to_string(),
            });
            queue.push_back(TunerAllocationRequest {
                requester: TunerRequester::Window(window_2_id.clone()),
                requirements: requirements_2,
                activity: TunerActivity::Scanning,
                requester_id: "window_2".to_string(),
            });
        }

        // Run TunerAllocationSystem once
        let config = Arc::new(ScanningConfig::default());
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let mut context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_pool(Arc::new(pool))
            .with_config(config)
            .with_shutdown_coordinator(shutdown_coordinator)
            .with_tuner_allocation_queue(allocation_queue.clone());

        let mut system = TunerAllocationSystem::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify: Window 0 gets tuner (Requested → Allocated)
        {
            let windows = window_entities.read().unwrap();
            let window_0 = windows.get(&window_0_id).unwrap();
            assert!(
                window_0.allocation.is_allocated(),
                "Window 0 should be allocated (first in queue)"
            );

            // Verify: Windows 1, 2 still in queue (still Requested)
            let window_1 = windows.get(&window_1_id).unwrap();
            assert!(
                window_1.allocation.is_requested(),
                "Window 1 should still be requested (no tuner available)"
            );

            let window_2 = windows.get(&window_2_id).unwrap();
            assert!(
                window_2.allocation.is_requested(),
                "Window 2 should still be requested (no tuner available)"
            );
        }

        // Verify queue state
        {
            let queue = allocation_queue.lock().unwrap();
            assert_eq!(
                queue.len(),
                2,
                "Queue should have 2 requests remaining (window 1, 2)"
            );
        }

        // TODO: Return tuner, run system again, verify Window 1 gets tuner (FIFO order)
        // This requires implementing tuner deallocation first
    }

    #[test]
    fn test_window_stuck_in_requested_on_no_tuner() {
        // Setup: Window requests tuner, no tuners available
        let pool = Pool::new_unfiltered();
        // NOTE: Not adding any devices - no tuners available

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = create_test_window(task_id.clone(), 0);

        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: TaskPriority::Normal,
        };

        window.allocation.request(
            requirements.clone(),
            TunerActivity::Scanning,
            "window_0".to_string(),
        );

        let mut window_world = EntityWorld::new();
        window_world.insert(window);
        let window_entities = Arc::new(RwLock::new(window_world));

        // Queue allocation request
        let allocation_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = allocation_queue.lock().unwrap();
            queue.push_back(TunerAllocationRequest {
                requester: TunerRequester::Window(window_id.clone()),
                requirements: requirements.clone(),
                activity: TunerActivity::Scanning,
                requester_id: "window_0".to_string(),
            });
        }

        // Run TunerAllocationSystem
        let config = Arc::new(ScanningConfig::default());
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let mut context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_pool(Arc::new(pool))
            .with_config(config)
            .with_shutdown_coordinator(shutdown_coordinator)
            .with_tuner_allocation_queue(allocation_queue.clone());

        let mut system = TunerAllocationSystem::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify: Window stays in Requested state
        {
            let windows = window_entities.read().unwrap();
            let window = windows.get(&window_id).unwrap();
            assert!(
                window.allocation.is_requested(),
                "Window should still be requested (no tuner available)"
            );
        }

        // Verify: Request stays in queue
        {
            let queue = allocation_queue.lock().unwrap();
            assert_eq!(
                queue.len(),
                1,
                "Request should stay in queue when no tuner available"
            );
        }

        // TODO: Add tuner, run system again, verify window transitions to Allocated
        // This requires implementing pool.add_device at runtime
    }

    #[test]
    fn test_system_name() {
        let system = TunerAllocationSystem::new();
        assert_eq!(system.name(), "TunerAllocationSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = TunerAllocationSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_runs_with_empty_queue() {
        let allocation_queue = Arc::new(Mutex::new(VecDeque::new()));
        let mut context = SystemContext::new().with_tuner_allocation_queue(allocation_queue);

        let mut system = TunerAllocationSystem::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_allocation() {
        use crate::ecs::components::station::{TuneAllocationComponent, TuneRequestComponent};

        // Setup: Create pool with 1 tuner
        let pool = Pool::new_unfiltered();
        let device_id = crate::hardware::DeviceId::from_serial("mock", "test-001");
        let caps = crate::hardware::Capabilities::for_mock("mock", "test-001");
        add_test_device_to_pool(
            &pool,
            device_id,
            caps,
            crate::hardware::types::Backend::Mock,
            None,
        );

        // Create SignalEntity with RequestQueued state
        let window_id = WindowId::new(TaskId::new("test"), 0);
        let mut signal_entity =
            crate::ecs::SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);

        // Set signal to RequestQueued state
        let request = TuneRequestComponent::new(window_id.clone());
        let allocation = TuneAllocationComponent::new();
        signal_entity.tune_state = crate::ecs::components::station::TuneState::RequestQueued {
            request,
            allocation,
        };

        // Get signal ID for queue request
        let signal_id = signal_entity.id().clone();

        // Store signal
        let mut signal_world = EntityWorld::new();
        signal_world.insert(signal_entity);
        let signal_entities = Arc::new(RwLock::new(signal_world));

        // Queue allocation request for signal
        let allocation_queue = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut queue = allocation_queue.lock().unwrap();
            queue.push_back(TunerAllocationRequest {
                requester: TunerRequester::Station(crate::ecs::StationId::new()), // Placeholder during transition
                requirements: TaskRequirements {
                    frequency_hz: 88.9e6,
                    bandwidth_hz: 200_000.0,
                    required_sample_rate: 2.0e6,
                    priority: TaskPriority::Normal,
                },
                activity: TunerActivity::Listening,
                requester_id: format!("signal_{:?}", signal_id),
            });
        }

        // Run TunerAllocationSystem
        let config = Arc::new(ScanningConfig::default());
        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities.clone())
            .with_pool(Arc::new(pool))
            .with_config(config)
            .with_shutdown_coordinator(shutdown_coordinator)
            .with_tuner_allocation_queue(allocation_queue.clone());

        let mut system = TunerAllocationSystem::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify: Signal should transition to Active state
        {
            let signals = signal_entities.read().unwrap();
            let signal = signals.get(&signal_id).unwrap();
            assert!(
                signal.tune_state.is_active(),
                "Signal should be in Active state after allocation"
            );
        }

        // Verify: Queue should be empty
        {
            let queue = allocation_queue.lock().unwrap();
            assert_eq!(queue.len(), 0, "Queue should be empty after processing");
        }
    }
}
