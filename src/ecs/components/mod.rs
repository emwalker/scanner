//! Components for ECS entities
//!
//! Components are pure data structures representing different aspects of entity state.
//! They contain no behavior - systems operate on components to implement logic.

mod allocation;
mod constraint;
mod display_name;
mod priority;
mod status;
mod tuner_device;

pub mod analysis;
pub mod audio;
pub mod device;
pub mod scan;
pub mod signal;
pub mod station;
pub mod task;
pub mod ui;
pub mod window;

pub use allocation::{AllocationComponent, AllocationState};
pub use analysis::AnalysisInputComponent;
pub use audio::{
    AudioAllocationComponent, AudioId, AudioPlaybackComponent, AudioTuningComponent,
    StopListeningRequestComponent,
};
pub use constraint::ConstraintComponent;
pub use device::{
    DeviceConnectionComponent, DeviceConnectionState, DeviceInfoComponent, DeviceLifecycleComponent,
};
pub use display_name::DisplayNameComponent;
pub use priority::{Priority, PriorityComponent};
pub use scan::{
    PauseRequestComponent, ResumeRequestComponent, ScanConfigComponent, ScanId,
    ScanLifecycleComponent, ScanPauseState, ScanProgressComponent, ScanResultsComponent, ScanType,
};
pub use signal::{AnalysisResults, AnalysisStatus, PlaybackState, SignalId};
pub use station::{
    StationDiscoveryComponent, StationHistoryComponent, StationId, StationInfoComponent,
    StationPlaybackComponent, StationPlaybackState, TuneState,
};
pub use status::{StatusComponent, TunerActivity};
pub use task::{
    TaskProgressComponent, TaskResult, TaskResultComponent, TaskResultValue, TaskState,
    TaskStateComponent,
};
pub use tuner_device::DeviceComponent;
pub use window::{
    SegmentComponent, WindowAllocationComponent, WindowId, WindowProgressComponent,
    WindowProgressState,
};
