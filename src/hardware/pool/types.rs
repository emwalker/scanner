//! Types for tuner pool management

use crate::hardware;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Result of attempting to add a device to the pool
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddDeviceResult {
    /// Device was successfully added to pool
    Added {
        device_id: hardware::DeviceId,
        tuner_count: usize,
    },
    /// Device was rejected by pool filter
    FilteredOut {
        device_id: hardware::DeviceId,
        reason: String,
    },
    /// Pool is in shutdown mode
    ShutdownMode,
    /// Pool lock could not be acquired
    PoolBusy,
}

impl AddDeviceResult {
    /// Returns true if the result is Added
    pub fn is_ok(&self) -> bool {
        matches!(self, AddDeviceResult::Added { .. })
    }

    /// Returns true if the result is not Added
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /// Unwrap for tests - panics if not Added
    #[cfg(test)]
    pub fn unwrap(self) {
        match self {
            AddDeviceResult::Added { .. } => {}
            other => panic!("Expected Added, got {:?}", other),
        }
    }
}

/// Tuner identifier: composite of device ID + channel index
///
/// This type identifies a specific tuner (RX channel) within a device.
/// Multi-tuner devices like the SDRplay RSPduo have multiple tuners,
/// each identified by a unique channel index.
#[derive(
    Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub struct TunerId {
    pub device_id: hardware::DeviceId,
    pub channel_index: usize,
}

impl TunerId {
    pub fn new(device_id: hardware::DeviceId, channel_index: usize) -> Self {
        Self {
            device_id,
            channel_index,
        }
    }
}

/// Device entry (physical SDR hardware)
pub struct DeviceEntry {
    /// Shared reference to the device
    /// Multiple tuners from the same device share this
    /// None when using subprocess mode (device opened in subprocess instead)
    pub device: Option<Arc<Mutex<Box<dyn hardware::DeviceTrait>>>>,

    /// Device-level capabilities
    pub capabilities: hardware::Capabilities,

    /// Backend that provides this device
    pub backend: hardware::types::Backend,

    /// Number of tuners/channels this device has
    pub num_tuners: usize,

    /// When device was added to pool
    pub added_at: Instant,
}

/// Tuner entry (individual RX channel within a device)
pub struct TunerEntry {
    /// Which device this tuner belongs to
    pub device_id: hardware::DeviceId,

    /// Channel index (0 for first tuner, 1 for second, etc.)
    pub channel_index: usize,

    /// Tuner-specific capabilities (may differ from device-level)
    pub capabilities: hardware::Capabilities,
}

/// Allocation tracking
pub struct AllocationInfo {
    pub allocated_at: Instant,
    pub task_id: Option<String>,
    pub backend: hardware::types::Backend,
    pub model: String,
    pub activity: TunerActivity,
}

/// Intermediate data from tuner allocation, used during subprocess spawning
#[allow(dead_code)]
pub(crate) struct TunerAllocation {
    pub tuner_id: TunerId,
    pub backend: hardware::types::Backend,
    pub model: String,
    pub capabilities: hardware::Capabilities,
    pub activity: TunerActivity,
}

/// What a tuner is currently doing (replaces ActiveTuners.scanning/listening)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TunerActivity {
    Scanning,
    Listening,
    Other,
}

/// Task requirements for capability matching
#[derive(Clone, Debug)]
pub struct TaskRequirements {
    pub frequency_hz: f64,
    pub bandwidth_hz: f64,
    pub required_sample_rate: f64,
    pub priority: TaskPriority,
}

/// Task priority (reserved for future priority-based scheduling)
#[derive(Clone, Debug)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
}

/// Pool status for TUI display
#[derive(Clone, Debug)]
pub struct PoolStatus {
    pub available_tuner_count: usize,
    pub allocated_tuner_count: usize,
    pub device_count: usize,
    pub tuners: Vec<TunerStatus>,
}

/// Individual tuner status
#[derive(Clone, Debug)]
pub struct TunerStatus {
    pub id: TunerId,
    pub state: TunerState,
    pub activity: Option<TunerActivity>,
}

/// Tuner state
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TunerState {
    Available,
    Allocated,
}
