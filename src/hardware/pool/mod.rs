//! Tuner allocation and filtering
//!
//! The Pool is an **allocation mechanism only**. Its sole responsibility is tracking
//! which tuners are available vs allocated, and deciding which tuner to allocate based
//! on requirements and filters.
//!
//! **Device discovery is the authoritative source for tuners.** The discovery service
//! enumerates hardware and populates tuner entities via DeviceEnumerationTask. The pool
//! references these entities (via Arc<Mutex<EntityWorld>>) but does not create or own them.
//!
//! **Ownership model:**
//! - Scan level (src/cli/scan.rs) creates and owns EntityWorlds
//! - Discovery (src/task/enumeration.rs) writes to EntityWorlds (adds/removes devices)
//! - Pool (src/hardware/pool) reads from EntityWorlds (queries for allocation)
//!
//! Key features:
//! - RAII guarantees: tuners automatically return when dropped
//! - Capability matching: allocates best tuner for each task
//! - PoolFilter: constrains which tuners can be allocated (driver, mode, channel)
//! - Subprocess management: spawns/reuses worker processes per device

mod filter;
mod lifecycle;
mod provider;
mod segment;
mod state;
mod subprocess;
mod subprocess_source;
pub mod test_utils;
mod tuner;
mod types;

use tokio::sync::broadcast;

/// Trait for sample segment providers
///
/// This trait defines the interface for objects that provide audio samples
/// via a broadcast channel. Both legacy (SoapySdrManager) and pool-based
/// (pool::Segment) implementations use this trait.
pub trait SegmentTrait: Send {
    fn audio_subscriber(&self) -> broadcast::Receiver<crate::broadcast::SamplePacket>;
}

pub use filter::{PoolFilter, TuningMode};
pub use provider::TunerProvider;
pub use segment::{Segment, detect_peaks_with_temp_graph};
pub use state::Pool;
pub use subprocess::SubprocessHandle;
pub use subprocess_source::SubprocessSource;
pub use tuner::Tuner;
pub use types::{
    AllocationInfo, DeviceEntry, PoolStatus, TaskPriority, TaskRequirements, TunerActivity,
    TunerEntry, TunerId, TunerState, TunerStatus,
};

#[cfg(test)]
mod tests;
