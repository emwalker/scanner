//! ECS Systems - behavior logic that operates on entities and components

pub mod audio;
pub mod device;
pub mod scan;
pub mod station;
pub mod tuner;
pub mod ui;

#[cfg(test)]
mod integration_tests;

pub use audio::{
    CoordinationSystem as AudioCoordinationSystem, ManagementSystem,
    PlaybackSystem as AudioPlaybackSystem,
};
pub use device::DiscoverySystem;
pub use scan::{
    CoordinationSystem as ScanCoordinationSystem,
    RequestProcessorSystem as ScanRequestProcessorSystem, WindowProcessingSystem,
};
pub use station::TuneTransitionSystem;
pub use tuner::{AllocationRequest, AllocationSystem};
pub use ui::UIUpdateSystem;
