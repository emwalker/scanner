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

// Test helpers module - available for both unit and integration tests
pub mod test_helpers;

mod coordinator;
mod entity;
mod schedule;
mod system;
mod world;

pub use coordinator::Coordinator;
pub use entity::Entity;
pub use queue::{
    PauseAndTuneRequest, PauseRequestQueue, TunerAllocationQueue, TunerAllocationRequest,
    TunerRequester,
};
pub use resources::{GlobalPauseResource, GlobalPauseState, PlayingStationInfo};
pub use schedule::Scheduler;
pub use system::{Resource, System, SystemContext};
pub use world::EntityWorld;

// Type alias for shared entity worlds
pub type Entities<T> = std::sync::Arc<std::sync::RwLock<EntityWorld<T>>>;

// Re-export commonly used types for convenience
pub use components::{
    AllocationComponent, AllocationState, AudioAllocationComponent, AudioId,
    AudioPlaybackComponent, AudioTuningComponent, ConstraintComponent, DeviceComponent,
    DeviceConnectionComponent, DeviceConnectionState, DeviceInfoComponent,
    DeviceLifecycleComponent, Priority, PriorityComponent, ScanConfigComponent, ScanId,
    ScanLifecycleComponent, ScanPauseState, ScanProgressComponent, ScanResultsComponent, ScanType,
    SegmentComponent, SignalId, StationDiscoveryComponent, StationHistoryComponent, StationId,
    StationInfoComponent, StationPlaybackComponent, StationPlaybackState, StatusComponent,
    TaskProgressComponent, TaskResult, TaskResultComponent, TaskResultValue, TaskState,
    TaskStateComponent, TuneState, WindowAllocationComponent, WindowId, WindowProgressComponent,
    WindowProgressState, scan::WindowAllocationRequest,
};
pub use entities::{
    AudioEntity, DeviceEntity, ScanEntity, ScanTaskData, SignalEntity, StationEntity,
    TaskComponents, TaskEntity, TaskId, TaskKind, TunerEntity, WindowEntity,
};
