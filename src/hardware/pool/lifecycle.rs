//! Tuner lifecycle operations (add/remove/acquire/release)

use crate::core::types::{Result, ScannerError};
use crate::hardware;
use crate::hardware::pool::state::{Pool, PoolInner};
use crate::hardware::pool::tuner::Tuner;
use crate::hardware::pool::types::{
    AddDeviceResult, AllocationInfo, DeviceEntry, PoolStatus, TaskRequirements, TunerActivity,
    TunerEntry, TunerId, TunerState, TunerStatus,
};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error};

impl Pool {
    /// Add newly discovered device and expose all its tuners
    ///
    /// Checks pool filter before adding device. Returns AddDeviceResult indicating what happened.
    /// During shutdown, returns ShutdownMode without blocking if pool is locked.
    fn check_filter_allows_device(
        &self,
        device_id: &hardware::DeviceId,
        backend_name: &str,
    ) -> Option<AddDeviceResult> {
        let test_tuner_id = TunerId::new(device_id.clone(), 0);

        let allocated_count = match self.pool_ref.try_lock() {
            Ok(inner) => inner.allocated_tuners.len(),
            Err(_) => {
                debug!(device_id = ?device_id, "Add device failed - pool is locked");
                return Some(AddDeviceResult::PoolBusy);
            }
        };

        if !self
            .filter
            .is_allowed(&test_tuner_id, backend_name, allocated_count)
        {
            debug!(
                device_id = ?device_id,
                backend = backend_name,
                "Device rejected by pool filter"
            );
            return Some(AddDeviceResult::FilteredOut {
                device_id: device_id.clone(),
                reason: format!("Filter does not allow backend '{}'", backend_name),
            });
        }

        None
    }

    fn create_and_insert_device_entry(
        &self,
        device: Box<dyn hardware::DeviceTrait>,
        backend_name: &str,
        capabilities: &hardware::Capabilities,
    ) -> Option<hardware::DeviceId> {
        let device_id = device.id().clone();
        let num_tuners = capabilities.channels;

        debug!(
            device_id = ?device_id,
            backend = backend_name,
            num_tuners = num_tuners,
            "Adding device to pool"
        );

        let device_entry = DeviceEntry {
            device: Arc::new(Mutex::new(device)),
            capabilities: capabilities.clone(),
            backend_name: backend_name.to_string(),
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
    ) {
        let mut inner = match self.pool_ref.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                debug!(device_id = ?device_id, "Failed to expose tuners - pool is locked");
                return;
            }
        };

        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);

            debug!(
                tuner_id = ?tuner_id,
                "Exposing tuner {}/{}", channel_index + 1, num_tuners
            );

            let tuner_entry = TunerEntry {
                device_id: device_id.clone(),
                channel_index,
                capabilities: capabilities.clone(),
            };

            inner.available_tuners.insert(tuner_id, tuner_entry);
        }
    }

    pub fn add_device(
        &self,
        device: Box<dyn hardware::DeviceTrait>,
        backend_name: String,
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

        if let Some(result) = self.check_filter_allows_device(&device_id, &backend_name) {
            return result;
        }

        let device_id =
            match self.create_and_insert_device_entry(device, &backend_name, &capabilities) {
                Some(id) => id,
                None => return AddDeviceResult::PoolBusy,
            };

        self.expose_tuners(&device_id, num_tuners, &capabilities);

        AddDeviceResult::Added {
            device_id,
            tuner_count: num_tuners,
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

        let has_allocated_tuners = inner
            .allocated_tuners
            .keys()
            .any(|tuner_id| &tuner_id.device_id == device_id);

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
            inner.available_tuners.remove(&tuner_id);
        }

        inner.devices.remove(device_id);

        debug!(device_id = ?device_id, "Device and all tuners removed");
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

    fn find_best_matching_tuner<'a>(
        &self,
        inner: &'a PoolInner,
        requirements: &TaskRequirements,
        allocated_count: usize,
    ) -> Option<(&'a TunerId, &'a TunerEntry)> {
        inner
            .available_tuners
            .iter()
            .filter_map(|(tuner_id, entry)| {
                let device_entry = match inner.devices.get(&entry.device_id) {
                    Some(entry) => entry,
                    None => {
                        error!(
                            device_id = ?entry.device_id,
                            tuner_id = ?tuner_id,
                            "Device not found for tuner - skipping"
                        );
                        return None;
                    }
                };

                let filter_allowed =
                    self.filter
                        .is_allowed(tuner_id, &device_entry.backend_name, allocated_count);
                let can_handle = entry.capabilities.can_handle_task(requirements);

                debug!(
                    tuner_id = ?tuner_id,
                    backend = &device_entry.backend_name,
                    filter_allowed = filter_allowed,
                    can_handle = can_handle,
                    "Pool acquire: evaluated tuner"
                );

                if filter_allowed && can_handle {
                    Some((tuner_id, entry))
                } else {
                    None
                }
            })
            .min_by_key(|(_, entry)| {
                let range_size = entry
                    .capabilities
                    .rx_frequency_ranges
                    .iter()
                    .map(|(min, max)| (max - min) as u64)
                    .sum::<u64>();

                (range_size, entry.channel_index)
            })
    }

    fn allocate_tuner(
        &self,
        inner: &mut PoolInner,
        tuner_id: &TunerId,
        activity: TunerActivity,
    ) -> Option<Tuner> {
        let entry = match inner.available_tuners.remove(tuner_id) {
            Some(e) => e,
            None => {
                error!(tuner_id = ?tuner_id, "Tuner disappeared during acquisition");
                return None;
            }
        };

        let device_entry = match inner.devices.get(&entry.device_id) {
            Some(d) => d,
            None => {
                error!(
                    device_id = ?entry.device_id,
                    tuner_id = ?tuner_id,
                    "Device not found for tuner during acquisition"
                );
                return None;
            }
        };

        let backend_name = device_entry.backend_name.clone();
        let model = format!("{:?}", device_entry.capabilities.device_id);
        let device = Arc::clone(&device_entry.device);

        inner.allocated_tuners.insert(
            tuner_id.clone(),
            AllocationInfo {
                allocated_at: Instant::now(),
                task_id: None,
                backend_name,
                model,
                activity,
            },
        );

        debug!(tuner_id = ?tuner_id, "Tuner acquired from pool");

        Some(Tuner {
            tuner_id: tuner_id.clone(),
            device,
            pool: Arc::clone(&self.pool_ref),
            shutdown_mode: Arc::clone(&self.shutdown_mode),
        })
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
        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(
                    *state_guard,
                    crate::hardware::pool::state::PoolState::Active(_)
                ) {
                    debug!("Acquire rejected - pool not in Active state");
                    return None;
                }
                drop(state_guard);
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

        let mut inner = match self.pool_ref.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                debug!("Acquire failed - pool is locked");
                return None;
            }
        };

        let allocated_count = inner.allocated_tuners.len();

        debug!(
            available_tuners = inner.available_tuners.len(),
            allocated_count = allocated_count,
            requirements = ?requirements,
            "Pool acquire: checking available tuners"
        );

        match self.find_best_matching_tuner(&inner, requirements, allocated_count) {
            Some((tuner_id, _)) => {
                let tuner_id = tuner_id.clone();
                self.allocate_tuner(&mut inner, &tuner_id, activity)
            }
            None => {
                debug!(requirements = ?requirements, "No tuner available");
                None
            }
        }
    }

    fn create_empty_pool_status() -> PoolStatus {
        PoolStatus {
            available_tuner_count: 0,
            allocated_tuner_count: 0,
            device_count: 0,
            tuners: vec![],
        }
    }

    fn collect_tuner_statuses(inner: &PoolInner) -> Vec<TunerStatus> {
        inner
            .available_tuners
            .iter()
            .filter_map(|(id, entry)| {
                let device = inner.devices.get(&entry.device_id)?;
                Some(TunerStatus {
                    id: id.clone(),
                    model: format!("{:?}", device.capabilities.device_id),
                    backend: device.backend_name.clone(),
                    channel_index: entry.channel_index,
                    state: TunerState::Available,
                    activity: None,
                })
            })
            .chain(inner.allocated_tuners.iter().map(|(id, info)| TunerStatus {
                id: id.clone(),
                model: info.model.clone(),
                backend: info.backend_name.clone(),
                channel_index: id.channel_index,
                state: TunerState::Allocated,
                activity: Some(info.activity.clone()),
            }))
            .collect()
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

        match self.pool_ref.try_lock() {
            Ok(inner) => PoolStatus {
                available_tuner_count: inner.available_tuners.len(),
                allocated_tuner_count: inner.allocated_tuners.len(),
                device_count: inner.devices.len(),
                tuners: Self::collect_tuner_statuses(&inner),
            },
            Err(_) => {
                debug!("Pool status requested but pool is locked - returning empty");
                Self::create_empty_pool_status()
            }
        }
    }
}
