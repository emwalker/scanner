//! Entity definitions for ECS
//!
//! Entities are composites of components that represent complete objects
//! in the system (tuners, scans, stations, audio sessions, etc.)

mod audio;
mod candidate;
mod hardware;
mod scan;
mod station;
mod tuner;
mod window;

pub use audio::AudioEntity;
pub use candidate::CandidateEntity;
pub use hardware::HardwareEntity;
pub use scan::ScanEntity;
pub use station::StationEntity;
pub use tuner::TunerEntity;
pub use window::WindowEntity;
