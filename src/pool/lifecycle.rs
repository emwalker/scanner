//! Tuner lifecycle operations (add/remove/acquire/release)

use crate::pool::state::Pool;
use crate::pool::tuner::Tuner;
use crate::pool::types::{
    AddDeviceResult, AllocationInfo, DeviceEntry, PoolStatus, TaskRequirements, TunerActivity,
    TunerEntry, TunerId, TunerState, TunerStatus,
};
use crate::sdr;
use crate::types::{Result, ScannerError};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::debug;

impl Pool {
    /// Add newly discovered device and expose all its tuners
    ///
    /// Checks pool filter before adding device. Returns AddDeviceResult indicating what happened.
    /// During shutdown, returns ShutdownMode without blocking if pool is locked.
    pub fn add_device(
        &self,
        device: Box<dyn sdr::DeviceTrait>,
        backend_name: String,
    ) -> AddDeviceResult {
        let device_id = device.id().clone();
        let capabilities = device.capabilities().clone();
        let num_tuners = capabilities.channels;

        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!(device_id = ?device_id, "Add device skipped - pool in shutdown mode");
            return AddDeviceResult::ShutdownMode;
        }

        // Check filter before adding
        // Create a temporary tuner_id to check filter (use channel 0)
        let test_tuner_id = TunerId::new(device_id.clone(), 0);

        // Use try_lock to get current allocation count for filter check
        let allocated_count = match self.pool_ref.try_lock() {
            Ok(inner) => inner.allocated_tuners.len(),
            Err(_) => {
                debug!(device_id = ?device_id, "Add device failed - pool is locked");
                return AddDeviceResult::PoolBusy;
            }
        };

        if !self
            .filter
            .is_allowed(&test_tuner_id, &backend_name, allocated_count)
        {
            debug!(
                device_id = ?device_id,
                backend = backend_name,
                "Device rejected by pool filter"
            );
            return AddDeviceResult::FilteredOut {
                device_id,
                reason: format!("Filter does not allow backend '{}'", backend_name),
            };
        }

        debug!(
            device_id = ?device_id,
            backend = backend_name,
            num_tuners = num_tuners,
            "Adding device to pool"
        );

        let device_arc = Arc::new(Mutex::new(device));

        let device_entry = DeviceEntry {
            device: Arc::clone(&device_arc),
            capabilities: capabilities.clone(),
            backend_name: backend_name.clone(),
            num_tuners,
            added_at: Instant::now(),
        };

        // Use try_lock to avoid blocking
        let mut inner = match self.pool_ref.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                debug!(device_id = ?device_id, "Add device failed - pool is locked");
                return AddDeviceResult::PoolBusy;
            }
        };
        inner.devices.insert(device_id.clone(), device_entry);

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

        AddDeviceResult::Added {
            device_id,
            tuner_count: num_tuners,
        }
    }

    /// Remove device (hot-unplug)
    ///
    /// Removes device and all its tuners. Returns error if any tuner is allocated.
    /// During shutdown, returns immediately without blocking if pool is locked.
    pub fn remove_device(&self, device_id: &sdr::DeviceId) -> Result<()> {
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
        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!("Acquire rejected - pool in shutdown mode");
            return None;
        }

        // Use try_lock to avoid blocking
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

        let best_match = inner
            .available_tuners
            .iter()
            .filter(|(tuner_id, entry)| {
                let device_entry = inner.devices.get(&entry.device_id).unwrap();
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

                filter_allowed && can_handle
            })
            .min_by_key(|(_, entry)| {
                let range_size = entry
                    .capabilities
                    .rx_frequency_ranges
                    .iter()
                    .map(|(min, max)| (max - min) as u64)
                    .sum::<u64>();

                (range_size, entry.channel_index)
            });

        match best_match {
            Some((tuner_id, _)) => {
                let tuner_id = tuner_id.clone();
                let entry = inner.available_tuners.remove(&tuner_id).unwrap();
                let device_entry = inner.devices.get(&entry.device_id).unwrap();
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
                        activity: activity.clone(),
                    },
                );

                debug!(tuner_id = ?tuner_id, "Tuner acquired from pool");

                Some(Tuner {
                    tuner_id,
                    device,
                    pool: Arc::clone(&self.pool_ref),
                    shutdown_mode: Arc::clone(&self.shutdown_mode),
                })
            }
            None => {
                debug!(requirements = ?requirements, "No tuner available");
                None
            }
        }
    }

    /// Get pool status (for TUI display)
    ///
    /// Returns empty status if in shutdown mode or if pool is locked.
    /// This prevents blocking during shutdown when threads are being torn down.
    pub fn status(&self) -> PoolStatus {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            debug!("Pool status requested during shutdown - returning empty");
            return PoolStatus {
                available_tuner_count: 0,
                allocated_tuner_count: 0,
                device_count: 0,
                tuners: vec![],
            };
        }

        match self.pool_ref.try_lock() {
            Ok(inner) => PoolStatus {
                available_tuner_count: inner.available_tuners.len(),
                allocated_tuner_count: inner.allocated_tuners.len(),
                device_count: inner.devices.len(),
                tuners: inner
                    .available_tuners
                    .iter()
                    .map(|(id, entry)| {
                        let device = inner.devices.get(&entry.device_id).unwrap();
                        TunerStatus {
                            id: id.clone(),
                            model: format!("{:?}", device.capabilities.device_id),
                            backend: device.backend_name.clone(),
                            channel_index: entry.channel_index,
                            state: TunerState::Available,
                            activity: None,
                        }
                    })
                    .chain(inner.allocated_tuners.iter().map(|(id, info)| TunerStatus {
                        id: id.clone(),
                        model: info.model.clone(),
                        backend: info.backend_name.clone(),
                        channel_index: id.channel_index,
                        state: TunerState::Allocated,
                        activity: Some(info.activity.clone()),
                    }))
                    .collect(),
            },
            Err(_) => {
                debug!("Pool status requested but pool is locked - returning empty");
                PoolStatus {
                    available_tuner_count: 0,
                    allocated_tuner_count: 0,
                    device_count: 0,
                    tuners: vec![],
                }
            }
        }
    }
}
