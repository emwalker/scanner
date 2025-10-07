//! Tuner pool with RAII-based resource management
//!
//! This module provides dynamic tuner inventory management for SDR devices.
//! Key features:
//! - RAII guarantees: tuners automatically return to pool when dropped
//! - Multi-tuner devices: exposes all tuners (e.g., RSPduo has 2 tuners)
//! - Capability matching: allocates best tuner for each task
//! - Controlled rollout: PoolFilter enables safe transition to multi-tuner operation

mod pooled_tuner;
mod segment;
mod types;

pub use pooled_tuner::PooledTuner;
pub use segment::PoolSegment;
pub use types::{
    AddDeviceResult, AllocationInfo, DeviceEntry, PoolStatus, TaskPriority, TaskRequirements,
    TunerActivity, TunerEntry, TunerId, TunerState, TunerStatus,
};

use crate::sdr;
use crate::types::{Result, ScannerError};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::debug;

/// Controls which tuners are available for allocation
///
/// Used for gradual rollout of multi-tuner support:
/// - Phase 1: Constrain by backend, driver, or tuning mode
/// - Phase 2+: Gradually relax constraints
/// - Final: allow_all() - full multi-tuner support
pub struct PoolFilter {
    backend: Option<String>,
    driver: Option<String>,
    mode: Option<TuningMode>,
    specific_tuners: Option<HashSet<TunerId>>,
}

/// Tuning mode constraint for filtering
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TuningMode {
    /// Single-tuner mode (ST) - only one tuner can be allocated at a time
    SingleTuner,
    /// Multi-tuner mode (MT) - multiple tuners can be allocated simultaneously
    MultiTuner,
}

impl PoolFilter {
    /// Create a new filter with optional constraints
    ///
    /// # Examples
    /// ```
    /// use scanner::pool::{PoolFilter, TuningMode};
    ///
    /// // Allow only sdrplay devices in single-tuner mode
    /// let filter = PoolFilter::new()
    ///     .with_driver("sdrplay")
    ///     .with_mode(TuningMode::SingleTuner);
    ///
    /// // Allow only soapy backend
    /// let filter = PoolFilter::new().with_backend("soapy");
    /// ```
    pub fn new() -> Self {
        Self {
            backend: None,
            driver: None,
            mode: None,
            specific_tuners: None,
        }
    }

    /// Constrain to specific backend (e.g., "soapy", "rtlsdr")
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Constrain to specific driver (e.g., "sdrplay", "rtlsdr")
    pub fn with_driver(mut self, driver: impl Into<String>) -> Self {
        self.driver = Some(driver.into());
        self
    }

    /// Constrain to specific tuning mode
    pub fn with_mode(mut self, mode: TuningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Constrain to specific tuner IDs (most restrictive)
    pub fn with_tuners(mut self, tuners: Vec<TunerId>) -> Self {
        self.specific_tuners = Some(tuners.into_iter().collect());
        self
    }

    /// Allow all tuners (full multi-tuner mode)
    pub fn allow_all() -> Self {
        Self::new()
    }

    /// Check if a tuner is allowed for allocation
    fn is_allowed(&self, tuner_id: &TunerId, backend_name: &str, allocated_count: usize) -> bool {
        // Check specific tuners first (most restrictive)
        // If specific tuners are set, ONLY those tuners are allowed (skip backend/driver checks)
        if let Some(allowed) = &self.specific_tuners {
            if !allowed.contains(tuner_id) {
                debug!(tuner_id = ?tuner_id, allowed = ?allowed, "Filter rejected: tuner not in allowed set");
                return false;
            }
            // Tuner is in allowed set, skip backend/driver checks and go to mode check
        } else {
            // Check backend (only if specific tuners not set)
            if let Some(allowed_backend) = &self.backend
                && backend_name != allowed_backend
            {
                debug!(
                    tuner_id = ?tuner_id,
                    backend_name = backend_name,
                    allowed_backend = allowed_backend,
                    "Filter rejected: backend mismatch"
                );
                return false;
            }

            // Check driver (case-insensitive, only if specific tuners not set)
            if let Some(allowed_driver) = &self.driver {
                match &tuner_id.device_id {
                    sdr::DeviceId::Backend { backend, .. } => {
                        if !backend.eq_ignore_ascii_case(allowed_driver) {
                            debug!(
                                tuner_id = ?tuner_id,
                                driver = backend,
                                allowed_driver = allowed_driver,
                                "Filter rejected: driver mismatch"
                            );
                            return false;
                        }
                    }
                    sdr::DeviceId::Usb { .. } => {
                        debug!(tuner_id = ?tuner_id, "Filter rejected: USB devices not allowed when driver filter is set");
                        return false;
                    }
                }
            }
        }

        // Check tuning mode
        if let Some(tuning_mode) = &self.mode {
            match tuning_mode {
                TuningMode::SingleTuner => {
                    // Only allow allocation if no tuners are currently allocated
                    if allocated_count > 0 {
                        debug!(
                            tuner_id = ?tuner_id,
                            allocated_count = allocated_count,
                            "Filter rejected: SingleTuner mode and {} tuner(s) already allocated",
                            allocated_count
                        );
                        return false;
                    }
                }
                TuningMode::MultiTuner => {
                    // No restriction on concurrent allocations
                }
            }
        }

        debug!(tuner_id = ?tuner_id, "Filter allowed tuner");
        true
    }
}

impl Default for PoolFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal state (needed for Arc<Mutex<>> pattern)
pub struct PoolInner {
    /// Devices (physical hardware)
    pub devices: HashMap<sdr::DeviceId, DeviceEntry>,

    /// Available tuners (ready for allocation)
    pub available_tuners: HashMap<TunerId, TunerEntry>,

    /// Allocated tuners (in use by tasks)
    pub allocated_tuners: HashMap<TunerId, AllocationInfo>,
}

impl PoolInner {
    /// Internal: return tuner to pool (called by PooledTuner::drop)
    pub fn return_tuner(&mut self, tuner_id: TunerId, shutdown_mode: bool) {
        if shutdown_mode {
            debug!(tuner_id = ?tuner_id, "Tuner return ignored (shutdown mode)");
            return;
        }

        debug!(tuner_id = ?tuner_id, "Tuner returned to pool");

        self.allocated_tuners.remove(&tuner_id);

        if let Some(device_entry) = self.devices.get(&tuner_id.device_id) {
            let tuner_entry = TunerEntry {
                device_id: tuner_id.device_id.clone(),
                channel_index: tuner_id.channel_index,
                capabilities: device_entry.capabilities.clone(),
            };

            self.available_tuners.insert(tuner_id, tuner_entry);
        }
    }
}

/// Dynamic inventory of available tuners
pub struct Pool {
    /// Internal state (Arc<Mutex<>> for thread-safe sharing with PooledTuner)
    pool_ref: Arc<Mutex<PoolInner>>,

    /// Filter controlling which tuners can be allocated
    filter: Arc<PoolFilter>,

    /// Shutdown mode flag (atomic for lock-free access)
    shutdown_mode: Arc<AtomicBool>,
}

impl Pool {
    /// Create new pool with filter
    pub fn new(filter: PoolFilter) -> Self {
        let inner = PoolInner {
            devices: HashMap::new(),
            available_tuners: HashMap::new(),
            allocated_tuners: HashMap::new(),
        };

        Self {
            pool_ref: Arc::new(Mutex::new(inner)),
            filter: Arc::new(filter),
            shutdown_mode: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create new pool allowing all tuners (convenience method)
    pub fn new_unfiltered() -> Self {
        Self::new(PoolFilter::allow_all())
    }

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
    ) -> Result<PooledTuner> {
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
    ) -> Option<PooledTuner> {
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

                Some(PooledTuner {
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

    /// Enter shutdown mode - makes all operations fail-fast
    ///
    /// After calling this, the pool will:
    /// - Reject new allocations (acquire/try_acquire return None/Err)
    /// - Ignore tuner returns (return_tuner is a no-op)
    /// - Allow queries to continue (status still works)
    ///
    /// This prevents deadlocks during shutdown when threads are being torn down.
    /// Uses atomic flag so it never blocks.
    pub fn shutdown(&self) {
        debug!("Pool entering shutdown mode");
        self.shutdown_mode.store(true, Ordering::SeqCst);
    }

    /// Check if pool is in shutdown mode
    ///
    /// This is a lock-free operation that can be called from any thread
    /// to check if shutdown has been initiated.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_mode.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdr;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_pooled_tuner_drop_doesnt_block_when_pool_locked() {
        let mut pool = Pool::new_unfiltered();
        let pool_arc = pool.pool_ref.clone();

        let device_id = sdr::DeviceId::from_serial("mock", "test001");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        let _pool_lock = pool_arc.lock().unwrap();

        let handle = thread::spawn(move || {
            drop(tuner);
        });

        let result = handle.join();
        assert!(result.is_ok());

        drop(_pool_lock);
    }

    #[test]
    fn test_pooled_tuner_drop_during_shutdown() {
        let mut pool = Arc::new(Pool::new_unfiltered());

        let device_id = sdr::DeviceId::from_serial("mock", "test002");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        Arc::get_mut(&mut pool)
            .unwrap()
            .add_device(device, "mock".to_string())
            .unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let tuner = pool_clone
                .try_acquire(&requirements, TunerActivity::Scanning)
                .unwrap();
            thread::sleep(Duration::from_millis(50));
            drop(tuner);
        });

        thread::sleep(Duration::from_millis(10));

        let _status = pool.status();

        handle.join().unwrap();

        let final_status = pool.status();
        assert_eq!(final_status.allocated_tuner_count, 0);
    }

    #[test]
    fn test_shutdown_mode() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test003");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        let status_before = pool.status();
        assert_eq!(status_before.allocated_tuner_count, 1);

        pool.shutdown();

        drop(tuner);

        thread::sleep(Duration::from_millis(10));

        let status_after = pool.status();
        assert_eq!(status_after.allocated_tuner_count, 0);
    }

    #[test]
    fn test_shutdown_never_blocks() {
        let mut pool = Arc::new(Pool::new_unfiltered());

        let device_id = sdr::DeviceId::from_serial("mock", "test004");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        Arc::get_mut(&mut pool)
            .unwrap()
            .add_device(device, "mock".to_string())
            .unwrap();

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            pool_clone.shutdown();
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_status_during_shutdown() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test005");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let _tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        let status_before = pool.status();
        assert_eq!(status_before.allocated_tuner_count, 1);

        pool.shutdown();

        let status_after = pool.status();
        assert_eq!(status_after.allocated_tuner_count, 0);
        assert_eq!(status_after.available_tuner_count, 0);
        assert_eq!(status_after.device_count, 0);
    }

    #[test]
    fn test_status_never_blocks_when_pool_locked() {
        let mut pool = Arc::new(Pool::new_unfiltered());

        let device_id = sdr::DeviceId::from_serial("mock", "test006");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        Arc::get_mut(&mut pool)
            .unwrap()
            .add_device(device, "mock".to_string())
            .unwrap();

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let status = pool_clone.status();
            assert_eq!(status.device_count, 0);
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_acquire_rejected_during_shutdown() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test007");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner_before = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner_before.is_some());
        drop(tuner_before);

        pool.shutdown();

        let tuner_after = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(
            tuner_after.is_none(),
            "Should not acquire tuner during shutdown"
        );
    }

    #[test]
    fn test_acquire_never_blocks_when_pool_locked() {
        use std::time::Duration;

        let mut pool = Arc::new(Pool::new_unfiltered());

        let device_id = sdr::DeviceId::from_serial("mock", "test008");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        Arc::get_mut(&mut pool)
            .unwrap()
            .add_device(device, "mock".to_string())
            .unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let start = Instant::now();
            let result = pool_clone.try_acquire(&requirements, TunerActivity::Scanning);
            let elapsed = start.elapsed();

            assert!(result.is_none(), "Should return None when pool is locked");
            assert!(
                elapsed < Duration::from_millis(100),
                "Should return immediately, took {:?}",
                elapsed
            );
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_device_rejected_during_shutdown() {
        let pool = Pool::new_unfiltered();

        pool.shutdown();

        let device_id = sdr::DeviceId::from_serial("mock", "test009");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        let result = pool.add_device(device, "mock".to_string());

        assert!(
            matches!(result, AddDeviceResult::ShutdownMode),
            "Should return ShutdownMode"
        );

        let status = pool.status();
        assert_eq!(
            status.device_count, 0,
            "Device should not be added during shutdown"
        );
    }

    #[test]
    fn test_remove_device_rejected_during_shutdown() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test010");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        pool.shutdown();

        let result = pool.remove_device(&device_id);
        assert!(
            result.is_ok(),
            "Should succeed but skip removing device during shutdown"
        );
    }

    #[test]
    fn test_add_device_never_blocks_when_pool_locked() {
        use std::time::Duration;

        let pool = Pool::new_unfiltered();
        let pool_arc = pool.pool_ref.clone();

        // Hold the lock in this thread
        let _lock = pool_arc.lock().unwrap();

        // Try to add device from another thread
        let handle = thread::spawn(move || {
            let mut pool_in_thread = pool;
            let start = Instant::now();
            let device_id = sdr::DeviceId::from_serial("mock", "test011");
            let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
            let result = pool_in_thread.add_device(device, "mock".to_string());
            let elapsed = start.elapsed();

            assert!(result.is_err(), "Should return error when pool is locked");
            assert!(
                elapsed < Duration::from_millis(100),
                "Should return immediately, took {:?}",
                elapsed
            );
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_device_never_blocks_when_pool_locked() {
        use std::time::Duration;

        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test012");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let pool_arc = pool.pool_ref.clone();
        let _lock = pool_arc.lock().unwrap();

        let handle = thread::spawn(move || {
            let mut pool_in_thread = pool;
            let start = Instant::now();
            let result = pool_in_thread.remove_device(&device_id);
            let elapsed = start.elapsed();

            assert!(result.is_err(), "Should return error when pool is locked");
            assert!(
                elapsed < Duration::from_millis(100),
                "Should return immediately, took {:?}",
                elapsed
            );
        });

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tuner_operations_rejected_during_shutdown() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test013");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let mut tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        pool.shutdown();

        let tune_result = tuner.tune(100.0e6);
        assert!(
            tune_result.is_err(),
            "Tune should fail during shutdown: {:?}",
            tune_result
        );

        let gain_result = tuner.set_gain(30.0);
        assert!(
            gain_result.is_err(),
            "Set gain should fail during shutdown: {:?}",
            gain_result
        );
    }

    #[test]
    fn test_is_shutdown() {
        let pool = Pool::new_unfiltered();

        assert!(
            !pool.is_shutdown(),
            "Pool should not be in shutdown mode initially"
        );

        pool.shutdown();

        assert!(
            pool.is_shutdown(),
            "Pool should be in shutdown mode after shutdown()"
        );
    }

    #[test]
    fn test_is_shutdown_thread_safe() {
        let pool = Arc::new(Pool::new_unfiltered());

        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            // Check from another thread
            assert!(!pool_clone.is_shutdown());

            // Wait for main thread to trigger shutdown
            thread::sleep(Duration::from_millis(50));

            assert!(pool_clone.is_shutdown());
        });

        thread::sleep(Duration::from_millis(10));
        pool.shutdown();

        handle.join().unwrap();
    }

    #[test]
    fn test_activity_tracking() {
        let mut pool = Pool::new_unfiltered();

        let device_id = sdr::DeviceId::from_serial("mock", "test014");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let _scanning_tuner = pool
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();

        let status = pool.status();
        assert_eq!(status.allocated_tuner_count, 1);

        let allocated_tuner = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .expect("Should have one allocated tuner");

        assert_eq!(allocated_tuner.activity, Some(TunerActivity::Scanning));

        drop(_scanning_tuner);

        let _listening_tuner = pool
            .try_acquire(&requirements, TunerActivity::Listening)
            .unwrap();

        let status = pool.status();
        let allocated_tuner = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .expect("Should have one allocated tuner");

        assert_eq!(allocated_tuner.activity, Some(TunerActivity::Listening));
    }

    #[test]
    fn test_filter_by_backend() {
        let pool = Pool::new(PoolFilter::new().with_backend("soapy"));

        let soapy_id = sdr::DeviceId::from_serial("sdrplay", "test015");
        let soapy_device = Box::new(sdr::mock::MockDevice::new(soapy_id.clone(), false));
        pool.add_device(soapy_device, "soapy".to_string()).unwrap();

        let rtlsdr_id = sdr::DeviceId::from_serial("rtlsdr", "test016");
        let rtlsdr_device = Box::new(sdr::mock::MockDevice::new(rtlsdr_id.clone(), false));
        // Different backend should be filtered out
        let result = pool.add_device(rtlsdr_device, "rtlsdr".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner.is_some(), "Should acquire tuner from soapy backend");

        let status = pool.status();
        assert_eq!(
            status.allocated_tuner_count, 1,
            "Should have one allocated tuner"
        );
        assert_eq!(
            status.available_tuner_count, 0,
            "Only soapy tuner was added, it's now allocated"
        );
    }

    #[test]
    fn test_filter_by_driver() {
        let pool = Pool::new(PoolFilter::new().with_driver("sdrplay"));

        let sdrplay_id = sdr::DeviceId::from_serial("sdrplay", "test017");
        let sdrplay_device = Box::new(sdr::mock::MockDevice::new(sdrplay_id.clone(), false));
        pool.add_device(sdrplay_device, "soapy".to_string())
            .unwrap();

        let rtlsdr_id = sdr::DeviceId::from_serial("rtlsdr", "test018");
        let rtlsdr_device = Box::new(sdr::mock::MockDevice::new(rtlsdr_id.clone(), false));
        // RTL-SDR should be filtered out by driver check
        let result = pool.add_device(rtlsdr_device, "soapy".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner.is_some(), "Should acquire sdrplay tuner");

        let status = pool.status();
        let allocated = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .unwrap();
        assert!(
            format!("{:?}", allocated.id.device_id).contains("sdrplay"),
            "Allocated tuner should be from sdrplay driver"
        );
    }

    #[test]
    fn test_filter_allow_tuners() {
        let mut pool = Pool::new_unfiltered();

        let device1_id = sdr::DeviceId::from_serial("mock", "test019");
        let device1 = Box::new(sdr::mock::MockDevice::new(device1_id.clone(), false));
        pool.add_device(device1, "mock".to_string()).unwrap();

        let device2_id = sdr::DeviceId::from_serial("mock", "test020");
        let device2 = Box::new(sdr::mock::MockDevice::new(device2_id.clone(), false));
        pool.add_device(device2, "mock".to_string()).unwrap();

        drop(pool);

        let tuner1 = TunerId::new(device1_id.clone(), 0);
        let mut pool_filtered = Pool::new(PoolFilter::new().with_tuners(vec![tuner1.clone()]));

        let device1_again = Box::new(sdr::mock::MockDevice::new(device1_id.clone(), false));
        pool_filtered
            .add_device(device1_again, "mock".to_string())
            .unwrap();

        let device2_again = Box::new(sdr::mock::MockDevice::new(device2_id.clone(), false));
        // Device2 should be filtered out (only tuner1 is allowed)
        let result = pool_filtered.add_device(device2_again, "mock".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner = pool_filtered
            .try_acquire(&requirements, TunerActivity::Scanning)
            .unwrap();
        assert_eq!(tuner.id(), &tuner1, "Should only acquire allowed tuner");
    }

    #[test]
    fn test_filter_single_tuner_mode() {
        let mut pool = Pool::new(PoolFilter::new().with_mode(TuningMode::SingleTuner));

        let device_id = sdr::DeviceId::from_serial("mock", "test021");
        let device = Box::new(sdr::mock::MockDevice::new(device_id.clone(), false));
        pool.add_device(device, "mock".to_string()).unwrap();

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner1.is_some(), "Should acquire first tuner");

        let tuner2 = pool.try_acquire(&requirements, TunerActivity::Listening);
        assert!(
            tuner2.is_none(),
            "Should not acquire second tuner in SingleTuner mode"
        );

        drop(tuner1);

        let tuner3 = pool.try_acquire(&requirements, TunerActivity::Listening);
        assert!(
            tuner3.is_some(),
            "Should acquire tuner after first is released"
        );
    }

    #[test]
    fn test_filter_combined_driver_and_mode() {
        let pool = Pool::new(
            PoolFilter::new()
                .with_driver("sdrplay")
                .with_mode(TuningMode::SingleTuner),
        );

        let sdrplay_id = sdr::DeviceId::from_serial("sdrplay", "test022");
        let sdrplay_device = Box::new(sdr::mock::MockDevice::new(sdrplay_id.clone(), false));
        pool.add_device(sdrplay_device, "soapy".to_string())
            .unwrap();

        let rtlsdr_id = sdr::DeviceId::from_serial("rtlsdr", "test023");
        let rtlsdr_device = Box::new(sdr::mock::MockDevice::new(rtlsdr_id.clone(), false));
        // RTL-SDR should be filtered out by driver check
        let result = pool.add_device(rtlsdr_device, "rtlsdr".to_string());
        assert!(matches!(result, AddDeviceResult::FilteredOut { .. }));

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        let tuner1 = pool.try_acquire(&requirements, TunerActivity::Scanning);
        assert!(tuner1.is_some(), "Should acquire sdrplay tuner");

        let status = pool.status();
        let allocated = status
            .tuners
            .iter()
            .find(|t| t.state == TunerState::Allocated)
            .unwrap();
        assert!(
            format!("{:?}", allocated.id.device_id).contains("sdrplay"),
            "Should allocate from sdrplay driver"
        );

        let tuner2 = pool.try_acquire(&requirements, TunerActivity::Listening);
        assert!(
            tuner2.is_none(),
            "Should not allocate second tuner in SingleTuner mode"
        );
    }
}
