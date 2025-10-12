//! Tuner pool with RAII-based resource management
//!
//! This module provides dynamic tuner inventory management for SDR devices.
//! Key features:
//! - RAII guarantees: tuners automatically return to pool when dropped
//! - Multi-tuner devices: exposes all tuners (e.g., RSPduo has 2 tuners)
//! - Capability matching: allocates best tuner for each task
//! - Controlled rollout: PoolFilter enables safe transition to multi-tuner operation

mod filter;
mod lifecycle;
mod provider;
mod segment;
mod state;
mod subprocess;
mod subprocess_source;
mod tuner;
mod types;

use tokio::sync::broadcast;

/// Trait for sample segment providers
///
/// This trait defines the interface for objects that provide audio samples
/// via a broadcast channel. Both legacy (SoapySdrManager) and pool-based
/// (pool::Segment) implementations use this trait.
pub trait SegmentTrait {
    fn audio_subscriber(&self) -> broadcast::Receiver<crate::broadcast::SamplePacket>;
}

pub use filter::{PoolFilter, TuningMode};
pub use provider::TunerProvider;
pub use segment::Segment;
pub use state::Pool;
pub use subprocess::SubprocessHandle;
pub use subprocess_source::SubprocessSource;
pub use tuner::Tuner;
pub use types::{
    AddDeviceResult, AllocationInfo, DeviceEntry, PoolStatus, TaskPriority, TaskRequirements,
    TunerActivity, TunerEntry, TunerId, TunerState, TunerStatus,
};

#[cfg(test)]
mod tests;
