//! User interface components and event types

pub mod display;
pub mod format;
pub mod tracking;
pub mod tui;

pub use format::{frequency_hz_label, frequency_hz_tabular};

/// Events that can be sent to the TUI for display
#[derive(Debug, Clone)]
pub enum TuiEvent {
    /// Tuner discovered and added (from discovery service)
    TunerAdded(crate::hardware::DeviceInfo),
    /// Tuner removed/disconnected (from discovery service)
    TunerRemoved(crate::hardware::DeviceId),
    /// Scanner has been paused and is ready for browsing
    Paused {
        tuner_id: crate::hardware::pool::TunerId,
    },
    /// Active tuners state has been updated
    ActiveTunersUpdated {
        status: crate::hardware::pool::PoolStatus,
    },
}
