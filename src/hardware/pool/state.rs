//! Core pool state structures

use crate::hardware;
use crate::hardware::pool::SubprocessHandle;
use crate::hardware::pool::filter::PoolFilter;
use crate::hardware::pool::types::{DeviceEntry, PoolStatus};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::debug;

/// Callback invoked when tuner state changes
pub type StateChangeCallback = Arc<Mutex<Vec<Box<dyn Fn(PoolStatus) + Send + Sync>>>>;

/// State: Pool is active and can allocate tuners
#[derive(Debug, Clone, PartialEq)]
pub struct Active;

/// State: Pool is shutting down, no new allocations allowed
#[derive(Debug, Clone, PartialEq)]
pub struct ShuttingDown;

/// Pool lifecycle state
///
/// This enum wraps typestate structs, allowing compile-time type safety
/// for state transitions while maintaining runtime flexibility for
/// dynamic event handling.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolState {
    /// Pool is active and can allocate tuners
    Active(Active),
    /// Pool is shutting down, no new allocations allowed
    ShuttingDown(ShuttingDown),
}

/// Internal state (needed for Arc<Mutex<>> pattern)
pub struct PoolInner {
    /// Devices (physical hardware)
    pub devices: HashMap<hardware::DeviceId, DeviceEntry>,
}

/// Dynamic inventory of available tuners
pub struct Pool {
    /// Current lifecycle state
    pub(crate) state: Mutex<PoolState>,

    /// Internal state (Arc<Mutex<>> for thread-safe sharing with Tuner)
    pub(crate) pool_ref: Arc<Mutex<PoolInner>>,

    /// Filter controlling which tuners can be allocated
    pub(crate) filter: Arc<PoolFilter>,

    /// Shutdown mode flag (atomic for lock-free access in hot paths and Drop)
    pub(crate) shutdown_mode: Arc<AtomicBool>,

    /// Callbacks invoked when tuner state changes (acquire/release)
    pub(crate) on_state_change: StateChangeCallback,

    /// Device worker subprocesses (one per device, lazily spawned)
    pub(crate) subprocesses: Mutex<HashMap<hardware::DeviceId, Arc<SubprocessHandle>>>,

    /// Parent process log file path (used to derive worker log paths)
    pub(crate) parent_log_file: Option<String>,

    /// ECS tuner entities (authoritative source of tuner state)
    pub(crate) tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,

    /// ECS allocation system (drives tuner allocation decisions)
    pub(crate) allocation_system: Mutex<crate::ecs::systems::AllocationSystem>,
}

impl Pool {
    /// Create new pool with filter and optional parent log file
    pub fn new(filter: PoolFilter, parent_log_file: Option<String>) -> Self {
        let inner = PoolInner {
            devices: HashMap::new(),
        };

        Self {
            state: Mutex::new(PoolState::Active(Active)),
            pool_ref: Arc::new(Mutex::new(inner)),
            filter: Arc::new(filter),
            shutdown_mode: Arc::new(AtomicBool::new(false)),
            on_state_change: Arc::new(Mutex::new(Vec::new())),
            subprocesses: Mutex::new(HashMap::new()),
            parent_log_file,
            tuner_entities: Arc::new(Mutex::new(crate::ecs::EntityWorld::new())),
            allocation_system: Mutex::new(crate::ecs::systems::AllocationSystem::new()),
        }
    }

    /// Create new pool allowing all tuners (convenience method)
    pub fn new_unfiltered() -> Self {
        Self::new(PoolFilter::allow_all(), None)
    }

    /// Get parent log file path for worker log derivation
    pub fn parent_log_file(&self) -> Option<String> {
        self.parent_log_file.clone()
    }

    /// Enter shutdown mode (makes pool reject all future operations)
    ///
    /// Transitions from Active → ShuttingDown. Idempotent if already shutting down.
    pub fn shutdown(&self) {
        if let Ok(mut state) = self.state.lock()
            && matches!(*state, PoolState::Active(_))
        {
            *state = PoolState::ShuttingDown(ShuttingDown);
            debug!("Pool state transitioned to ShuttingDown");
        }

        self.shutdown_mode.store(true, Ordering::SeqCst);
        debug!("Pool entered shutdown mode");

        if let Ok(mut subprocesses) = self.subprocesses.lock() {
            let count = subprocesses.len();
            if count > 0 {
                debug!(
                    subprocess_count = count,
                    "Shutting down device subprocesses"
                );
            }

            for (device_id, handle) in subprocesses.iter_mut() {
                debug!(device_id = ?device_id, "Shutting down subprocess");
                if let Some(handle) = Arc::get_mut(handle) {
                    let _ = handle.shutdown();
                }
            }

            subprocesses.clear();
        }
    }

    /// Check if pool is in shutdown mode
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_mode.load(Ordering::SeqCst)
    }

    /// Check if pool is in Active state
    pub fn is_active(&self) -> bool {
        self.state
            .lock()
            .map(|state| matches!(*state, PoolState::Active(_)))
            .unwrap_or(false)
    }

    /// Check if pool is in ShuttingDown state
    pub fn is_shutting_down(&self) -> bool {
        self.state
            .lock()
            .map(|state| matches!(*state, PoolState::ShuttingDown(_)))
            .unwrap_or(false)
    }

    /// Register a callback to be invoked when tuner state changes
    pub fn add_state_change_callback(&self, callback: Box<dyn Fn(PoolStatus) + Send + Sync>) {
        if let Ok(mut callbacks) = self.on_state_change.lock() {
            callbacks.push(callback);
        }
    }

    /// Invoke all registered state change callbacks
    pub(crate) fn notify_state_change(&self) {
        let status = self.status();
        if let Ok(callbacks) = self.on_state_change.lock() {
            for callback in callbacks.iter() {
                callback(status.clone());
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        debug!("Pool being dropped, calling shutdown");
        self.shutdown();
    }
}
