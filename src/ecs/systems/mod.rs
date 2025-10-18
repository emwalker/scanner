//! ECS Systems - behavior logic that operates on entities and components

pub mod audio;
pub mod device;
pub mod scan;
pub mod tuner;

#[cfg(test)]
mod integration_tests;

pub use audio::ManagementSystem;
pub use device::DiscoverySystem;
pub use scan::CoordinationSystem;
pub use tuner::{AllocationRequest, AllocationSystem};
