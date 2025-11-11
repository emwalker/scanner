//! ECS Resources - global shared state

pub mod clock;
pub mod filesystem;
pub mod global_pause;
pub mod location;

pub use clock::{Clock, DurationExt, MockClock, SystemClock};
pub use filesystem::{FileMetadata, FileSystem, MockFileSystem, StdFileSystem};
pub use global_pause::{GlobalPauseResource, GlobalPauseState, PlayingStationInfo};
pub use location::{
    DetectedLocation, LocationConfidence, LocationError, LocationResource, LocationSource,
    new_location_resource,
};
