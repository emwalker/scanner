//! ECS Systems - behavior logic that operates on entities and components

pub mod audio;
pub mod device;
pub mod scan;
pub mod signal;
pub mod station;
pub mod task;
pub mod tuner;
pub mod ui;
pub mod window;

#[cfg(test)]
mod integration_tests;

pub use audio::{
    AudioSpawnSystem, CoordinationSystem as AudioCoordinationSystem, ManagementSystem,
    PlaybackSystem as AudioPlaybackSystem,
};
pub use device::DiscoverySystem;
pub use scan::{
    CoordinationSystem as ScanCoordinationSystem,
    RequestProcessorSystem as ScanRequestProcessorSystem, WindowProcessingSystem,
};
pub use signal::{PeakAnalysisSystem, SignalAnalysisSpawnSystem, SignalAnalysisSystem};
pub use station::{
    TuneRequestSystem, TuneTransitionSystem, TunerAllocationSystem, TuningCoordinationSystem,
};
pub use task::TaskCoordinationSystem;
pub use tuner::{AllocationRequest, AllocationSystem};
pub use ui::UIUpdateSystem;
pub use window::{
    PeakCompletionSystem, PeakDetectionSystem, WindowTimeoutSystem, WindowWorkerCompletionSystem,
    WindowWorkerSpawnSystem,
};
