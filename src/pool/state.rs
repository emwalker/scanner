//! Core pool state structures

use crate::pool::filter::PoolFilter;
use crate::pool::types::{AllocationInfo, DeviceEntry, TunerEntry, TunerId};
use crate::sdr;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::debug;

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
    /// Internal: return tuner to pool (called by Tuner::drop)
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
    /// Internal state (Arc<Mutex<>> for thread-safe sharing with Tuner)
    pub(crate) pool_ref: Arc<Mutex<PoolInner>>,

    /// Filter controlling which tuners can be allocated
    pub(crate) filter: Arc<PoolFilter>,

    /// Shutdown mode flag (atomic for lock-free access)
    pub(crate) shutdown_mode: Arc<AtomicBool>,
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

    /// Enter shutdown mode (makes pool reject all future operations)
    pub fn shutdown(&self) {
        self.shutdown_mode.store(true, Ordering::SeqCst);
        debug!("Pool entered shutdown mode");
    }

    /// Check if pool is in shutdown mode
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_mode.load(Ordering::SeqCst)
    }
}
