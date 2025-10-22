//! Entity-Component-System (ECS) architecture
//!
//! This module provides a lightweight ECS implementation inspired by game engines
//! but adapted for SDR scanning use cases. Unlike full ECS frameworks like Bevy,
//! this implementation uses simple HashMap-based storage since we have dozens of
//! entities rather than thousands.
//!
//! Core concepts:
//! - **Entities**: Lightweight identifiers that tie together components
//! - **Components**: Pure data structures representing different aspects of entity state
//! - **Systems**: Pure functions that operate on components to implement behavior
//! - **World**: Storage container for entities of a specific type

pub mod components;
pub mod entities;
pub mod queue;
pub mod resources;
pub mod systems;

mod coordinator;
mod entity;
mod schedule;
mod system;
mod world;

pub use coordinator::Coordinator;
pub use entity::Entity;
pub use queue::{PauseRequest, PauseRequestQueue};
pub use resources::{GlobalPauseResource, GlobalPauseState};
pub use schedule::Scheduler;
pub use system::{Resource, System, SystemContext};
pub use world::EntityWorld;

// Type alias for shared entity worlds
pub type Entities<T> = std::sync::Arc<std::sync::RwLock<EntityWorld<T>>>;

// Re-export commonly used types for convenience
pub use components::scan::WindowAllocationRequest;
pub use components::{
    AllocationComponent, AllocationState, AudioAllocationComponent, AudioId,
    AudioPlaybackComponent, AudioTuningComponent, CandidateId, CandidateInfoComponent,
    CandidateLifecycleComponent, CandidateProgressComponent, CandidateState, ConstraintComponent,
    DeviceComponent, HardwareConnectionComponent, HardwareConnectionState, HardwareInfoComponent,
    HardwareLifecycleComponent, Priority, PriorityComponent, ScanConfigComponent, ScanId,
    ScanLifecycleComponent, ScanPauseState, ScanProgressComponent, ScanResultsComponent, ScanType,
    SegmentComponent, StationDiscoveryComponent, StationHistoryComponent, StationId,
    StationInfoComponent, StationPlaybackComponent, StationPlaybackState, StatusComponent,
    WindowAllocationComponent, WindowId, WindowProgressComponent, WindowProgressState,
};
pub use entities::{
    AudioEntity, CandidateEntity, HardwareEntity, ScanEntity, StationEntity, TunerEntity,
    WindowEntity,
};
