//! Tuner lifecycle operations (add/remove/acquire/release)

use std::sync::{Arc, atomic::Ordering};

use tracing::{debug, info};

use crate::{
    core::types::{Result, ScannerError},
    ecs::Entity,
    hardware,
    hardware::pool::{
        PoolFilter,
        state::{Pool, PoolInner},
        tuner::Tuner,
        types::{
            PoolStatus, TaskRequirements, TunerActivity, TunerAllocation, TunerId, TunerState,
            TunerStatus,
        },
    },
};

impl Pool {
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

        // Get device info from device_entities
        let device_entities = match self.device_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to remove device - device entities locked");
                return Err(ScannerError::Custom("Device entities locked".to_string()));
            }
        };

        let device_entity = device_entities
            .get(device_id)
            .ok_or_else(|| ScannerError::DeviceNotFound(device_id.clone()))?;

        let num_tuners = device_entity.num_tuners();
        drop(device_entities);

        // Check tuner allocation status
        let mut entities = match self.tuner_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to remove device - tuner entities locked");
                return Err(ScannerError::Custom("Tuner entities locked".to_string()));
            }
        };

        let has_allocated_tuners = entities
            .iter()
            .any(|e| &e.id().device_id == device_id && e.allocation.is_allocated());

        if has_allocated_tuners {
            debug!(device_id = ?device_id, "Cannot remove device - tuners in use");
            return Err(ScannerError::DeviceInUse(device_id.clone()));
        }

        // Remove all tuner entities for this device
        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);
            entities.remove(&tuner_id);
            info!(tuner_id = ?tuner_id, "Removed TunerEntity");
        }
        drop(entities);

        // Remove DeviceEntity
        let mut device_entities = match self.device_entities.try_lock() {
            Ok(entities) => entities,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to remove device entity - locked");
                return Err(ScannerError::Custom("Device entities locked".to_string()));
            }
        };

        device_entities.remove(device_id);
        debug!(device_id = ?device_id, "Device and all tuners removed, DeviceEntity removed");

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

    /// Create a Tuner handle from a pre-allocated tuner_id
    ///
    /// This method is used when allocation has already been performed by AllocationSystem.
    /// The Pool acts as a Resource here - it trusts that the allocation is valid and
    /// simply creates the hardware handle.
    ///
    /// Returns None if:
    /// - Pool is shutting down
    /// - Cannot spawn subprocess for the device
    pub fn create_tuner_from_allocated(&self, tuner_id: TunerId) -> Option<Tuner> {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!("create_tuner_from_allocated rejected - pool in shutdown mode");
            return None;
        }

        let subprocess_handle = match self.get_or_spawn_subprocess(&tuner_id.device_id) {
            Ok(handle) => handle,
            Err(e) => {
                debug!(
                    tuner_id = ?tuner_id,
                    error = ?e,
                    "Failed to get or spawn subprocess"
                );
                return None;
            }
        };

        debug!(tuner_id = ?tuner_id, "Created tuner from pre-allocated tuner_id");

        Some(Tuner {
            tuner_id,
            backend: crate::hardware::pool::tuner::TunerBackend::Subprocess {
                subprocess: subprocess_handle,
            },
            tuner_entities: Arc::clone(&self.tuner_entities),
            on_return: Box::new({
                let on_state_change = Arc::clone(&self.on_state_change);
                move || {
                    if let Ok(_callbacks) = on_state_change.lock() {
                        // State changed, but we can't easily compute status here
                        // The Tuner Drop will handle deallocating the tuner entity
                    }
                }
            }),
            shutdown_mode: Arc::clone(&self.shutdown_mode),
        })
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
        _inner: &PoolInner,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Option<TunerAllocation> {
        let mut entities = self.tuner_entities.try_lock().ok()?;
        let available_count = entities.iter().filter(|e| e.is_available()).count();

        debug!(
            available_tuners = available_count,
            num_entities = entities.len(),
            requirements = ?requirements,
            "Querying tuner entities for allocation"
        );

        use crate::ecs::Entity;

        let tuner_id = entities
            .iter()
            .find(|e| {
                e.is_available()
                    && (*self.filter).is_allowed(
                        e.id(),
                        &e.device.backend,
                        entities
                            .iter()
                            .filter(|t| t.allocation.is_allocated())
                            .count(),
                        &e.mode,
                    )
                    && e.device
                        .capabilities
                        .supports_frequency(requirements.frequency_hz)
                    && e.device
                        .capabilities
                        .supports_sample_rate(requirements.required_sample_rate)
            })
            .map(|e| e.id().clone())?;

        let entity = entities.get_mut(&tuner_id)?;
        let requester_id = format!(
            "pool_task_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        entity.allocation.allocate(requester_id);
        entity.status.activity = match activity {
            TunerActivity::Scanning => crate::ecs::components::TunerActivity::Scanning,
            TunerActivity::Listening => crate::ecs::components::TunerActivity::Listening,
            TunerActivity::Other => crate::ecs::components::TunerActivity::Other,
        };

        let model = format!(
            "{} channel {}",
            entity.device.device_id, entity.device.channel_index
        );

        Some(TunerAllocation {
            tuner_id,
            backend: entity.device.backend.clone(),
            model,
            capabilities: entity.device.capabilities.clone(),
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
        let tuner_entities = Arc::clone(&self.tuner_entities);
        let device_entities = Arc::clone(&self.device_entities);
        let shutdown_mode_clone = Arc::clone(&self.shutdown_mode);
        let filter = Arc::clone(&self.filter);

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

                // Get device count from device_entities
                let device_count = match device_entities.try_lock() {
                    Ok(hw_entities) => hw_entities.len(),
                    Err(_) => {
                        // Fallback: estimate from tuner count (may be inaccurate for multi-tuner
                        // devices)
                        entities.len()
                    }
                };

                let status = crate::hardware::pool::Pool::build_status_from_entities(
                    &entities,
                    device_count,
                    &filter,
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

        // First, check if subprocess already exists in DeviceEntity
        let device_entities = self
            .device_entities
            .try_lock()
            .map_err(|_| ScannerError::Custom("Hardware entities locked".to_string()))?;

        if let Some(entity) = device_entities.get(device_id) {
            if let Some(handle) = entity.connection.subprocess() {
                debug!(device_id = ?device_id, "Reusing existing subprocess from DeviceEntity");
                return Ok(handle);
            }
        } else {
            return Err(ScannerError::DeviceNotFound(device_id.clone()));
        }

        // Subprocess doesn't exist yet, spawn it
        debug!(
            device_id = ?device_id,
            "Spawning new subprocess (first allocation)"
        );

        drop(device_entities);

        let handle = Arc::new(SubprocessHandle::spawn(
            device_id.clone(),
            Arc::clone(&self.shutdown_mode),
            self.parent_log_file.as_deref(),
        )?);

        // Store subprocess in DeviceEntity
        let mut device_entities = self
            .device_entities
            .try_lock()
            .map_err(|_| ScannerError::Custom("Hardware entities locked".to_string()))?;

        if let Some(entity) = device_entities.get_mut(device_id) {
            entity.connection.attach_subprocess(Arc::clone(&handle));
            debug!(device_id = ?device_id, "Attached subprocess to DeviceEntity");
        }

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
        filter: &PoolFilter,
    ) -> Vec<TunerStatus> {
        // When building status, pass allocated_count=0 so that SingleTuner mode
        // doesn't filter out tuners. We want to show all tuners in status,
        // regardless of allocation state. The allocated_count check is only
        // relevant when deciding whether to ALLOCATE a new tuner, not when
        // REPORTING status.
        entities
            .iter()
            .filter(|entity| {
                filter.is_allowed(entity.id(), &entity.device.backend, 0, &entity.mode)
            })
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
        filter: &PoolFilter,
    ) -> PoolStatus {
        let filtered_tuners = Self::collect_tuner_statuses_from_entities(entities, filter);
        let available_count = filtered_tuners
            .iter()
            .filter(|t| t.state == TunerState::Available)
            .count();
        let allocated_count_filtered = filtered_tuners
            .iter()
            .filter(|t| t.state == TunerState::Allocated)
            .count();

        PoolStatus {
            available_tuner_count: available_count,
            allocated_tuner_count: allocated_count_filtered,
            device_count,
            tuners: filtered_tuners,
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

        // Get device count from device_entities (extract count and immediately drop lock)
        let device_count = match self.device_entities.try_lock() {
            Ok(hw_entities) => hw_entities.len(),
            Err(_) => {
                debug!("Pool status requested but device entities are locked - returning empty");
                return Self::create_empty_pool_status();
            }
        };

        let tuner_count = entities.len();
        debug!(
            tuner_count = tuner_count,
            device_count = device_count,
            "Generating pool status from entities"
        );

        Self::build_status_from_entities(&entities, device_count, &self.filter)
    }
}
