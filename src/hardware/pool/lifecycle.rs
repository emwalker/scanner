//! Tuner lifecycle operations (add/remove/acquire/release)

use crate::core::types::{Result, ScannerError};
use crate::ecs::Entity;
use crate::hardware;
use crate::hardware::pool::state::{Pool, PoolInner};
use crate::hardware::pool::tuner::Tuner;
use crate::hardware::pool::types::{
    AddDeviceResult, DeviceEntry, PoolStatus, TaskRequirements, TunerActivity, TunerAllocation,
    TunerId, TunerState, TunerStatus,
};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error};

impl Pool {
    fn create_and_insert_device_entry(
        &self,
        device: Option<Box<dyn hardware::DeviceTrait>>,
        device_id: hardware::DeviceId,
        backend: &hardware::types::Backend,
        capabilities: &hardware::Capabilities,
    ) -> Option<hardware::DeviceId> {
        let num_tuners = capabilities.channels;

        debug!(
            device_id = ?device_id,
            backend = ?backend,
            num_tuners = num_tuners,
            has_device = device.is_some(),
            "Adding device to pool"
        );

        let device_entry = DeviceEntry {
            device: device.map(|d| Arc::new(Mutex::new(d))),
            capabilities: capabilities.clone(),
            backend: backend.clone(),
            num_tuners,
            added_at: Instant::now(),
        };

        let mut inner = match self.pool_ref.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                debug!(device_id = ?device_id, "Add device failed - pool is locked");
                return None;
            }
        };

        inner.devices.insert(device_id.clone(), device_entry);
        Some(device_id)
    }

    fn expose_tuners(
        &self,
        device_id: &hardware::DeviceId,
        num_tuners: usize,
        capabilities: &hardware::Capabilities,
        backend: &hardware::types::Backend,
    ) -> usize {
        let mut exposed_count = 0;

        let mut entities = match self.tuner_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to expose tuners - entities locked");
                return 0;
            }
        };

        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();

        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);

            // Check if this tuner passes the filter
            if !self.filter.is_allowed(&tuner_id, backend, allocated_count) {
                debug!(
                    tuner_id = ?tuner_id,
                    "Tuner filtered out - not exposing"
                );
                continue;
            }

            debug!(
                tuner_id = ?tuner_id,
                "Exposing tuner {}/{}", channel_index + 1, num_tuners
            );

            let entity = crate::ecs::TunerEntity::new(
                device_id.clone(),
                channel_index,
                capabilities.clone(),
                backend.clone(),
            );
            entities.insert(entity);
            exposed_count += 1;
            debug!(tuner_id = ?tuner_id, "Created TunerEntity");
        }

        exposed_count
    }

    pub fn add_device(
        &self,
        device: Box<dyn hardware::DeviceTrait>,
        backend: hardware::types::Backend,
    ) -> AddDeviceResult {
        let device_id = device.id().clone();
        let capabilities = device.capabilities().clone();
        let num_tuners = capabilities.channels;

        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(
                    *state_guard,
                    crate::hardware::pool::state::PoolState::Active(_)
                ) {
                    debug!(device_id = ?device_id, "Add device rejected - pool not in Active state");
                    return AddDeviceResult::ShutdownMode;
                }
                drop(state_guard);
            }
            Err(_) => {
                debug!(device_id = ?device_id, "Add device skipped - state lock contention");
                return AddDeviceResult::PoolBusy;
            }
        }

        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!(device_id = ?device_id, "Add device skipped - pool in shutdown mode");
            return AddDeviceResult::ShutdownMode;
        }

        let device_id = match self.create_and_insert_device_entry(
            Some(device),
            device_id,
            &backend,
            &capabilities,
        ) {
            Some(id) => id,
            None => return AddDeviceResult::PoolBusy,
        };

        let exposed_count = self.expose_tuners(&device_id, num_tuners, &capabilities, &backend);

        if exposed_count == 0 {
            // No tuners passed the filter - remove the device
            debug!(
                device_id = ?device_id,
                "Removing device - no tuners passed filter"
            );
            if let Err(e) = self.remove_device(&device_id) {
                debug!(
                    device_id = ?device_id,
                    error = ?e,
                    "Failed to remove filtered device (ignoring)"
                );
            }
            return AddDeviceResult::FilteredOut {
                device_id,
                reason: "No tuners passed filter criteria".to_string(),
            };
        }

        AddDeviceResult::Added {
            device_id,
            tuner_count: exposed_count,
        }
    }

    pub fn add_device_metadata(
        &self,
        device_id: hardware::DeviceId,
        capabilities: hardware::Capabilities,
        backend: hardware::types::Backend,
    ) -> AddDeviceResult {
        let num_tuners = capabilities.channels;

        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(
                    *state_guard,
                    crate::hardware::pool::state::PoolState::Active(_)
                ) {
                    debug!(device_id = ?device_id, "Add device metadata rejected - pool not in Active state");
                    return AddDeviceResult::ShutdownMode;
                }
                drop(state_guard);
            }
            Err(_) => {
                debug!(device_id = ?device_id, "Add device metadata skipped - state lock contention");
                return AddDeviceResult::PoolBusy;
            }
        }

        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!(device_id = ?device_id, "Add device metadata skipped - pool in shutdown mode");
            return AddDeviceResult::ShutdownMode;
        }

        let device_id =
            match self.create_and_insert_device_entry(None, device_id, &backend, &capabilities) {
                Some(id) => id,
                None => return AddDeviceResult::PoolBusy,
            };

        let exposed_count = self.expose_tuners(&device_id, num_tuners, &capabilities, &backend);

        if exposed_count == 0 {
            // No tuners passed the filter - remove the device
            debug!(
                device_id = ?device_id,
                "Removing device - no tuners passed filter"
            );
            if let Err(e) = self.remove_device(&device_id) {
                debug!(
                    device_id = ?device_id,
                    error = ?e,
                    "Failed to remove filtered device (ignoring)"
                );
            }
            return AddDeviceResult::FilteredOut {
                device_id,
                reason: "No tuners passed filter criteria".to_string(),
            };
        }

        self.notify_state_change();

        AddDeviceResult::Added {
            device_id,
            tuner_count: exposed_count,
        }
    }

    /// Remove device (hot-unplug)
    ///
    /// Removes device and all its tuners. Returns error if any tuner is allocated.
    /// During shutdown, returns immediately without blocking if pool is locked.
    pub fn remove_device(&self, device_id: &hardware::DeviceId) -> Result<()> {
        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!(device_id = ?device_id, "Remove device skipped - pool in shutdown mode");
            return Ok(());
        }

        // Use try_lock to avoid blocking
        let mut inner = match self.pool_ref.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                debug!(device_id = ?device_id, "Remove device failed - pool is locked");
                return Err(ScannerError::PoolLockTimeout);
            }
        };

        let mut entities = match self.tuner_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to remove device - entities locked");
                return Err(ScannerError::Custom("Entities locked".to_string()));
            }
        };

        let has_allocated_tuners = entities
            .iter()
            .any(|e| &e.id().device_id == device_id && e.allocation.is_allocated());

        if has_allocated_tuners {
            debug!(device_id = ?device_id, "Cannot remove device - tuners in use");
            return Err(ScannerError::DeviceInUse(device_id.clone()));
        }

        let device_entry = inner
            .devices
            .get(device_id)
            .ok_or_else(|| ScannerError::DeviceNotFound(device_id.clone()))?;

        let num_tuners = device_entry.num_tuners;

        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);
            entities.remove(&tuner_id);
            debug!(tuner_id = ?tuner_id, "Removed TunerEntity");
        }

        inner.devices.remove(device_id);

        debug!(device_id = ?device_id, "Device and all tuners removed");
        drop(inner);

        self.notify_state_change();
        Ok(())
    }

    /// Acquire tuner matching requirements with activity tracking
    ///
    /// Tuners are filtered based on the pool's PoolFilter before capability matching.
    /// The `activity` parameter tracks what the tuner will be used for (Scanning, Listening, etc.)
    pub fn acquire(
        &self,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Result<Tuner> {
        self.try_acquire(requirements, activity)
            .ok_or_else(|| ScannerError::NoAvailableTuner(requirements.clone()))
    }

    /// Check if pool is in valid state for acquisition
    ///
    /// Returns None if pool state is not Active, shutdown mode is enabled,
    /// or state lock cannot be acquired.
    fn check_acquisition_preconditions(&self) -> Option<()> {
        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(
                    *state_guard,
                    crate::hardware::pool::state::PoolState::Active(_)
                ) {
                    debug!("Acquire rejected - pool not in Active state");
                    return None;
                }
            }
            Err(_) => {
                debug!("Acquire rejected - state lock contention");
                return None;
            }
        }

        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!("Acquire rejected - pool in shutdown mode");
            return None;
        }

        Some(())
    }

    /// Select best matching tuner and mark it as allocated
    ///
    /// Returns allocation data or None if no suitable tuner found.
    /// Updates entity state to mark tuner as allocated.
    fn allocate_matching_tuner(
        &self,
        inner: &PoolInner,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Option<TunerAllocation> {
        use crate::ecs::components::Priority;
        use crate::ecs::systems::AllocationRequest;
        use crate::ecs::{System, SystemContext};

        let entities = self.tuner_entities.try_lock().ok()?;
        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();
        let available_count = entities.iter().filter(|e| e.is_available()).count();

        debug!(
            available_tuners = available_count,
            allocated_count = allocated_count,
            requirements = ?requirements,
            "Pool acquire: checking available tuners"
        );
        drop(entities);

        let requester_id = format!(
            "pool_task_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        {
            let mut system = self.allocation_system.lock().unwrap();
            system.request_allocation(AllocationRequest {
                requester_id: requester_id.clone(),
                frequency_hz: requirements.frequency_hz,
                sample_rate_hz: requirements.bandwidth_hz,
                priority: Priority::Medium,
                for_audio: activity == TunerActivity::Listening,
                filter: Some(Arc::clone(&self.filter)),
                allocated_count,
            });

            let mut context =
                SystemContext::new().with_tuner_entities(Arc::clone(&self.tuner_entities));

            if let Err(e) = System::run(&mut *system, &mut context) {
                debug!(error = ?e, "AllocationSystem failed");
                return None;
            }
        }

        let entities = self.tuner_entities.try_lock().ok()?;
        let tuner_id = entities
            .iter()
            .find(|e| {
                e.allocation.is_allocated()
                    && e.allocation.allocated_to.as_ref() == Some(&requester_id)
            })?
            .id()
            .clone();

        let device_entry = match inner.devices.get(&tuner_id.device_id) {
            Some(d) => d,
            None => {
                error!(
                    device_id = ?tuner_id.device_id,
                    tuner_id = ?tuner_id,
                    "Device not found for tuner during acquisition"
                );
                return None;
            }
        };

        let backend = device_entry.backend.clone();
        let model = device_entry.capabilities.device_id.to_string();
        let capabilities = device_entry.capabilities.clone();

        debug!(tuner_id = ?tuner_id, "Tuner acquired via AllocationSystem");

        Some(TunerAllocation {
            tuner_id,
            backend,
            model,
            capabilities,
            activity,
        })
    }

    /// Spawn or reuse subprocess for the allocated tuner
    ///
    /// On failure, automatically rolls back the allocation by returning
    /// the tuner to the available pool.
    fn spawn_subprocess_with_rollback(
        &self,
        allocation: &TunerAllocation,
    ) -> Option<crate::hardware::pool::tuner::TunerBackend> {
        match self.get_or_spawn_subprocess(&allocation.tuner_id.device_id) {
            Ok(subprocess) => {
                Some(crate::hardware::pool::tuner::TunerBackend::Subprocess { subprocess })
            }
            Err(e) => {
                debug!(
                    tuner_id = ?allocation.tuner_id,
                    error = ?e,
                    "Failed to spawn subprocess for tuner - rolling back allocation"
                );

                if let Ok(mut entities) = self.tuner_entities.try_lock()
                    && let Some(entity) = entities.get_mut(&allocation.tuner_id)
                {
                    entity.allocation.deallocate();
                    entity.status.idle();
                    debug!(tuner_id = ?allocation.tuner_id, "Rolled back allocation");
                }

                None
            }
        }
    }

    /// Create the final Tuner object with callbacks
    fn create_tuner_with_callbacks(
        &self,
        allocation: TunerAllocation,
        backend: crate::hardware::pool::tuner::TunerBackend,
    ) -> crate::hardware::pool::tuner::Tuner {
        let on_state_change = Arc::clone(&self.on_state_change);
        let pool_ref = Arc::clone(&self.pool_ref);
        let tuner_entities = Arc::clone(&self.tuner_entities);
        let shutdown_mode_clone = Arc::clone(&self.shutdown_mode);

        self.notify_state_change();

        crate::hardware::pool::tuner::Tuner {
            tuner_id: allocation.tuner_id.clone(),
            backend,
            tuner_entities: Arc::clone(&self.tuner_entities),
            on_return: Box::new(move || {
                if shutdown_mode_clone.load(Ordering::SeqCst) {
                    return;
                }

                let entities = match tuner_entities.try_lock() {
                    Ok(entities) => entities,
                    Err(_) => return,
                };

                let device_count = match pool_ref.try_lock() {
                    Ok(inner) => inner.devices.len(),
                    Err(_) => entities.len(),
                };

                let status = crate::hardware::pool::Pool::build_status_from_entities(
                    &entities,
                    device_count,
                );

                if let Ok(callbacks) = on_state_change.lock() {
                    for callback in callbacks.iter() {
                        callback(status.clone());
                    }
                }
            }),
            shutdown_mode: Arc::clone(&self.shutdown_mode),
        }
    }

    /// Try to acquire tuner matching requirements (non-blocking)
    ///
    /// Similar to `acquire()` but returns `None` instead of an error if no tuner is available.
    /// Returns `None` if the pool is in shutdown mode or if the pool lock cannot be acquired.
    /// The `activity` parameter tracks what the tuner will be used for.
    pub fn try_acquire(
        &self,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Option<Tuner> {
        self.check_acquisition_preconditions()?;

        let inner = self.pool_ref.try_lock().ok()?;
        let allocation = self.allocate_matching_tuner(&inner, requirements, activity)?;
        drop(inner);

        let backend = self.spawn_subprocess_with_rollback(&allocation)?;

        Some(self.create_tuner_with_callbacks(allocation, backend))
    }

    /// Get or spawn subprocess for a device (lazy spawning)
    ///
    /// Returns Arc to subprocess handle. Multiple tuners on same device share one subprocess.
    /// Subprocess is spawned on first call for a given device_id, then reused on subsequent calls.
    fn get_or_spawn_subprocess(
        &self,
        device_id: &hardware::DeviceId,
    ) -> Result<Arc<crate::hardware::pool::SubprocessHandle>> {
        use crate::hardware::pool::SubprocessHandle;

        let mut subprocesses = self
            .subprocesses
            .lock()
            .map_err(|e| ScannerError::Custom(format!("Subprocess lock failed: {}", e)))?;

        if let Some(handle) = subprocesses.get(device_id) {
            debug!(device_id = ?device_id, "Reusing existing subprocess");
            return Ok(Arc::clone(handle));
        }

        let pool_inner = self
            .pool_ref
            .try_lock()
            .map_err(|_| ScannerError::PoolLockTimeout)?;

        let device_entry = pool_inner
            .devices
            .get(device_id)
            .ok_or_else(|| ScannerError::DeviceNotFound(device_id.clone()))?;

        debug!(
            device_id = ?device_id,
            num_tuners = device_entry.num_tuners,
            "Spawning new subprocess (first allocation)"
        );

        drop(pool_inner);

        let handle = Arc::new(SubprocessHandle::spawn(
            device_id.clone(),
            Arc::clone(&self.shutdown_mode),
            self.parent_log_file.as_deref(),
        )?);

        subprocesses.insert(device_id.clone(), Arc::clone(&handle));

        Ok(handle)
    }

    fn create_empty_pool_status() -> PoolStatus {
        PoolStatus {
            available_tuner_count: 0,
            allocated_tuner_count: 0,
            device_count: 0,
            tuners: vec![],
        }
    }

    fn collect_tuner_statuses_from_entities(
        entities: &crate::ecs::EntityWorld<crate::ecs::TunerEntity>,
    ) -> Vec<TunerStatus> {
        entities
            .iter()
            .map(|entity| {
                let (state, activity) = if entity.allocation.is_allocated() {
                    let activity = match entity.status.activity {
                        crate::ecs::components::TunerActivity::Idle => TunerActivity::Other,
                        crate::ecs::components::TunerActivity::Scanning => TunerActivity::Scanning,
                        crate::ecs::components::TunerActivity::Listening => {
                            TunerActivity::Listening
                        }
                        crate::ecs::components::TunerActivity::Other => TunerActivity::Other,
                    };
                    (TunerState::Allocated, Some(activity))
                } else {
                    (TunerState::Available, None)
                };

                TunerStatus {
                    id: entity.id().clone(),
                    state,
                    activity,
                }
            })
            .collect()
    }

    /// Build PoolStatus from EntityWorld (helper for status computation)
    pub(crate) fn build_status_from_entities(
        entities: &crate::ecs::EntityWorld<crate::ecs::TunerEntity>,
        device_count: usize,
    ) -> PoolStatus {
        let available_count = entities.iter().filter(|e| e.is_available()).count();
        let allocated_count = entities
            .iter()
            .filter(|e| e.allocation.is_allocated())
            .count();

        PoolStatus {
            available_tuner_count: available_count,
            allocated_tuner_count: allocated_count,
            device_count,
            tuners: Self::collect_tuner_statuses_from_entities(entities),
        }
    }

    /// Get pool status (for TUI display)
    ///
    /// Returns empty status if in shutdown mode or if pool is locked.
    /// This prevents blocking during shutdown when threads are being torn down.
    pub fn status(&self) -> PoolStatus {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!("Pool status requested during shutdown - returning empty");
            return Self::create_empty_pool_status();
        }

        let entities = match self.tuner_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!("Pool status requested but entities are locked - returning empty");
                return Self::create_empty_pool_status();
            }
        };

        let inner = match self.pool_ref.try_lock() {
            Ok(inner) => inner,
            Err(_) => {
                debug!("Pool status requested but pool is locked - returning empty");
                return Self::create_empty_pool_status();
            }
        };

        Self::build_status_from_entities(&entities, inner.devices.len())
    }
}
