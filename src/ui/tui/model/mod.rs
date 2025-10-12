//! TUI data model using The Elm Architecture pattern
//!
//! This module is being refactored from a single 3,657-line file into focused submodules.
//! During migration, we maintain backward compatibility by re-exporting everything.

pub mod devices;
pub mod navigation;
pub mod queries;
pub mod state;
pub mod types;
pub mod updates;

#[cfg(test)]
mod tests;

// Re-export all types for backward compatibility
pub use state::{Model, TunerInfo};
pub use types::{
    CandidateProgress, CandidateStatus, FocusState, SelectedCandidateInfo, TunerDisplayInfo,
    TunerDisplayState, TunerState, UiMode, WindowProgress,
};
