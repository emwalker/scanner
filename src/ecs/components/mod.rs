//! Components for ECS entities
//!
//! Components are pure data structures representing different aspects of entity state.
//! They contain no behavior - systems operate on components to implement logic.

mod allocation;
mod constraint;
mod device;
mod priority;
mod status;

pub mod audio;
pub mod candidate;
pub mod hardware;
pub mod scan;
pub mod station;
pub mod window;

pub use allocation::{AllocationComponent, AllocationState};
pub use audio::{
    AudioAllocationComponent, AudioId, AudioPlaybackComponent, AudioTuningComponent,
    StopListeningRequestComponent,
};
pub use candidate::{
    CandidateId, CandidateInfoComponent, CandidateLifecycleComponent, CandidateProgressComponent,
    CandidateState,
};
pub use constraint::ConstraintComponent;
pub use device::DeviceComponent;
pub use hardware::{
    HardwareConnectionComponent, HardwareConnectionState, HardwareInfoComponent,
    HardwareLifecycleComponent,
};
pub use priority::{Priority, PriorityComponent};
pub use scan::{
    PauseRequestComponent, ResumeRequestComponent, ScanConfigComponent, ScanId,
    ScanLifecycleComponent, ScanPauseState, ScanProgressComponent, ScanResultsComponent, ScanType,
};
pub use station::{
    StationDiscoveryComponent, StationHistoryComponent, StationId, StationInfoComponent,
    StationPlaybackComponent, StationPlaybackState,
};
pub use status::{StatusComponent, TunerActivity};
pub use window::{
    SegmentComponent, WindowAllocationComponent, WindowId, WindowProgressComponent,
    WindowProgressState,
};
