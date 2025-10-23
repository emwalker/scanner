//! Task abstraction for SDR operations
//!
//! This module provides a task-based architecture for SDR operations,
//! enabling parallel execution across multiple devices and backends.
//!
//! Key features:
//! - Backend-safe task scheduling with serialized API access
//! - RAII-based tuner management
//! - Per-task cancellation
//! - Device enumeration tasks

mod enumeration;
mod scheduler;
mod types;

// Public exports
pub use enumeration::DeviceEnumerationTask;
pub use scheduler::TaskScheduler;
pub use types::{
    Task, TaskContinuation, TaskError, TaskHandle, TaskId, TaskPriority, TaskStatus, TaskType,
};

#[cfg(test)]
mod tests;
