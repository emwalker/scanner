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
pub mod scan;
pub mod station;

pub use allocation::{AllocationComponent, AllocationState};
pub use audio::{AudioAllocationComponent, AudioId, AudioPlaybackComponent, AudioTuningComponent};
pub use constraint::ConstraintComponent;
pub use device::DeviceComponent;
pub use priority::{Priority, PriorityComponent};
pub use scan::{
    ScanConfigComponent, ScanId, ScanLifecycleComponent, ScanPauseState, ScanProgressComponent,
    ScanResultsComponent, ScanType,
};
pub use station::{
    StationDiscoveryComponent, StationHistoryComponent, StationId, StationInfoComponent,
};
pub use status::{StatusComponent, TunerActivity};
