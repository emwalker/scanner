//! Entity definitions for ECS
//!
//! Entities are composites of components that represent complete objects
//! in the system (tuners, scans, stations, audio sessions, etc.)

mod audio;
mod device;
mod scan;
mod signal;
mod station;
mod task;
mod task_components;
mod tuner;
mod window;

pub use audio::AudioEntity;
pub use device::DeviceEntity;
pub use scan::ScanEntity;
pub use signal::SignalEntity;
pub use station::StationEntity;
pub use task::{ScanTaskData, TaskEntity, TaskId, TaskKind, TaskWindowCell};
pub use task_components::TaskComponents;
pub use tuner::TunerEntity;
pub use window::WindowEntity;
