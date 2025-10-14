//! Core types for task abstraction

use crate::core::types::{Result, ScannerError};
use crate::hardware::{pool, types::Backend};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Task wrapper using enum dispatch (faster than dyn trait)
#[allow(dead_code)]
pub enum Task {
    ScanBand(super::ScanBandTask),
    ScanStations(super::ScanStationsTask),
    Audio(super::AudioTask),
    DeviceEnumeration(super::DeviceEnumerationTask),
}

impl Task {
    /// Run the task with shutdown signal
    ///
    /// All tasks have the same signature - they manage their own resource acquisition.
    #[allow(dead_code)]
    pub fn run(&mut self, shutdown: CancellationToken) -> Result<()> {
        match self {
            Task::ScanBand(t) => t.run(shutdown),
            Task::ScanStations(t) => t.run(shutdown),
            Task::Audio(t) => t.run(shutdown),
            Task::DeviceEnumeration(t) => t.run(shutdown),
        }
    }

    /// Backend this task operates on
    #[allow(dead_code)]
    pub fn backend(&self) -> Backend {
        match self {
            Task::ScanBand(t) => t.backend(),
            Task::ScanStations(t) => t.backend(),
            Task::Audio(t) => t.backend(),
            Task::DeviceEnumeration(t) => t.backend().clone(),
        }
    }

    /// Task type identifier
    #[allow(dead_code)]
    pub fn task_type(&self) -> TaskType {
        match self {
            Task::ScanBand(_) => TaskType::ScanningBand,
            Task::ScanStations(_) => TaskType::ScanningStations,
            Task::Audio(_) => TaskType::Audio,
            Task::DeviceEnumeration(_) => TaskType::DeviceEnumeration,
        }
    }

    /// Human-readable description for TUI
    #[allow(dead_code)]
    pub fn description(&self) -> String {
        match self {
            Task::ScanBand(t) => t.description(),
            Task::ScanStations(t) => t.description(),
            Task::Audio(t) => t.description(),
            Task::DeviceEnumeration(t) => t.description(),
        }
    }

    /// Lifecycle hook: called when task starts
    #[allow(dead_code)]
    pub fn on_start(&mut self) {
        match self {
            Task::ScanBand(t) => t.on_start(),
            Task::ScanStations(t) => t.on_start(),
            Task::Audio(t) => t.on_start(),
            Task::DeviceEnumeration(t) => t.on_start(),
        }
    }

    /// Lifecycle hook: called when task completes successfully
    #[allow(dead_code)]
    pub fn on_complete(&mut self) {
        match self {
            Task::ScanBand(t) => t.on_complete(),
            Task::ScanStations(t) => t.on_complete(),
            Task::Audio(t) => t.on_complete(),
            Task::DeviceEnumeration(t) => t.on_complete(),
        }
    }

    /// Lifecycle hook: called when task encounters an error
    #[allow(dead_code)]
    pub fn on_error(&mut self, error: &ScannerError) {
        match self {
            Task::ScanBand(t) => t.on_error(error),
            Task::ScanStations(t) => t.on_error(error),
            Task::Audio(t) => t.on_error(error),
            Task::DeviceEnumeration(t) => t.on_error(error),
        }
    }
}

/// Task type identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TaskType {
    ScanningBand,
    ScanningStations,
    Audio,
    DeviceEnumeration,
}

/// Task priority for scheduling
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
}

/// Task ID for tracking
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[allow(dead_code)]
pub struct TaskId(u64);

impl TaskId {
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for controlling a running task
#[derive(Clone)]
#[allow(dead_code)]
pub struct TaskHandle {
    pub task_id: TaskId,
    cancel_token: CancellationToken,
}

impl TaskHandle {
    #[allow(dead_code)]
    pub fn new(task_id: TaskId, cancel_token: CancellationToken) -> Self {
        Self {
            task_id,
            cancel_token,
        }
    }

    /// Cancel this specific task without affecting others
    #[allow(dead_code)]
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Check if task has been cancelled
    #[allow(dead_code)]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }
}

/// Error information from failed tasks
#[derive(Debug)]
#[allow(dead_code)]
pub struct TaskError {
    pub task_id: TaskId,
    pub task_type: TaskType,
    pub error: ScannerError,
    pub occurred_at: Instant,
}

/// Status of a running task
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TaskStatus {
    pub task_id: TaskId,
    pub task_type: TaskType,
    pub description: String,
    pub tuner_id: Option<pool::TunerId>,
    pub running_duration: Duration,
}
