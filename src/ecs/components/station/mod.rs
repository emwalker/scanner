//! Station components for ECS

mod discovery;
mod history;
mod info;
mod tune_request;

pub use discovery::StationDiscoveryComponent;
pub use history::StationHistoryComponent;
pub use info::StationInfoComponent;
pub use tune_request::TuneRequestComponent;

use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a station
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StationId(u64);

static NEXT_STATION_ID: AtomicU64 = AtomicU64::new(1);

impl StationId {
    /// Create a new unique station ID
    pub fn new() -> Self {
        Self(NEXT_STATION_ID.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for StationId {
    fn default() -> Self {
        Self::new()
    }
}
