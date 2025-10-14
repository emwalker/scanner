# Plan 009: Task Abstraction

**Date**: October 2025
**Status**: ✅ Complete (All Steps 1-9 Complete, Phase 2 State Machine Migration Done)
**Dependencies**: ✅ `005-backend-abstraction.md`, `007-device-pool.md`
**Related Plans**: `004-multi-sdr.md` (parent plan), ✅ `003-structured-concurrency-shutdown.md`
**Enables**: Plan 010

## Prerequisites

- ✅ Plan 005: Backend abstraction complete
- ✅ Plan 007: Device pool complete (Phases 1a-1e)
- ✅ Plan 003: Structured concurrency complete

## Executive Summary

Transform operations (scanning, audio playback) from monolithic functions into independent, reusable tasks.

## Task Architecture

All tasks implement a unified interface:

```rust
pub fn run(&mut self, shutdown: CancellationToken) -> Result<()>
```

Tasks are self-contained units that manage their own resource acquisition.

### Task Classifications

**Short-lived tuner holders** (AudioTask):
- Acquire tuner at start of run() from pool
- Hold tuner for entire duration
- Release automatically when Segment/Tuner drops
- Duration: seconds to minutes

**Coordinators** (ScanBandTask, ScanStationsTask, DeviceEnumerationTask):
- Don't acquire tuners themselves
- Delegate to sub-operations that acquire tuners (scan tasks → windows)
- Or don't need tuners at all (DeviceEnumerationTask)
- Duration: seconds to minutes

### Why Coordinators Don't Hold Tuners

Long-running scans (5-10 minutes) would block backend access if they held a tuner for the entire duration. Per-window allocation (0.5 seconds per window) allows other tasks to access the backend during gaps.

Timeline:
```
0s:    ScanTask starts
0.1s:  Window 1 acquires tuner → processes (0.5s) → releases
0.6s:  DeviceEnumerationTask can run (0.2s)
0.8s:  Window 2 acquires tuner → processes (0.5s) → releases
```

### Benefits

- Simplified TaskScheduler - no tuner acquisition logic needed
- All tasks have same signature
- Tasks control their own resource lifecycle
- Consistent lifecycle hooks across all task types
- Tasks can run in parallel on different devices automatically

## Problem Statement

Operations were embedded in `MainThread`, tightly coupled and unable to run in parallel or be tested in isolation.

## Goal

Operations as independent, composable tasks:

```rust
// Create tasks
let scan_task = Task::ScanBand(ScanBandTask::new(config, band));
let audio_task = Task::Audio(AudioTask::new(station_freq, config));

// Submit to scheduler
scheduler.submit(scan_task)?;
scheduler.submit(audio_task)?;

// Both run in parallel on separate tuners
```

## Core Types

```rust
pub enum Task {
    ScanBand(ScanBandTask),
    ScanStations(ScanStationsTask),
    Audio(AudioTask),
    DeviceEnumeration(DeviceEnumerationTask),
}

pub enum TaskType {
    ScanningBand,
    ScanningStations,
    Audio,
    DeviceEnumeration,
}

pub enum TaskPriority {
    Low,       // Background scanning
    Normal,    // Regular audio
    High,      // Future: P25 control channel
}

pub struct TaskHandle {
    pub task_id: TaskId,
    cancel_token: CancellationToken,
}

pub struct TaskStatus {
    pub task_id: TaskId,
    pub task_type: TaskType,
    pub description: String,
    pub tuner_id: Option<pool::TunerId>,
    pub running_duration: Duration,
}
```

## Task Implementations

### ScanBandTask

Coordinator task that scans entire bands. Each window acquires its own tuner, processes, and releases it. Supports both Phase 1 (simple scanning) and Phase 2 (full state machine with pause, tune, listen).

```rust
pub struct ScanBandTask {
    config: ScanningConfig,
    band: Band,
    progress_reporter: Arc<dyn ProgressReporter>,
    pause_signal: PauseSignal,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    command_receiver: Option<Receiver<ScannerCommand>>,
    tui_event_sender: Option<Sender<TuiEvent>>,
}
```

### ScanStationsTask

Similar to ScanBandTask but scans specific frequencies from a list. Shares the same state machine implementation via `ScanContext`.

```rust
pub struct ScanStationsTask {
    config: ScanningConfig,
    stations: Vec<f64>,  // Specific frequencies to scan
    // ... same fields as ScanBandTask
}
```

### DeviceEnumerationTask

Discovers available SDR devices for a backend. Serialized through the backend queue to prevent concurrent API access.

```rust
pub struct DeviceEnumerationTask {
    backend: crate::hardware::Backend,
    pool: Arc<Pool>,
    discovery_tx: mpsc::Sender<crate::discovery::Event>,
}
```

When scheduled:
1. Queries backend API for available devices
2. Adds devices to pool via `add_device_metadata()`
3. Emits `discovery::Event::Added` for successfully added devices

Integration with discovery service:
```
USB Event/Timer → Discovery Service
                ↓
                Submit DeviceEnumerationTask(Backend::Soapy, pool, discovery_tx)
                ↓
                Backend serialization (1 task per backend at a time)
                ↓
                Enumerate devices → Add to Pool → Emit events → TUI updates
```

### AudioTask

Acquires and holds a tuner for its entire duration. Creates an AudioSession for persistent audio streaming.

```rust
pub struct AudioTask {
    station_freq: f64,
    config: ScanningConfig,
    pool: Arc<Pool>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}
```

## TaskScheduler

Schedules tasks with backend API serialization:

```rust
pub struct TaskScheduler {
    sdr_pool: Arc<Pool>,
    running_tasks: Arc<DashMap<TaskId, RunningTaskInfo>>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    backend_queues: Arc<Mutex<HashMap<Backend, VecDeque<PrioritizedTask>>>>,
    backend_semaphores: Arc<Mutex<HashMap<Backend, Arc<tokio::sync::Semaphore>>>>,
}
```

Key operations:
- `submit(task)` - Submit task for execution, returns `TaskHandle`
- `stop(task_id)` - Cancel specific task
- `status()` - Query all running tasks
- `shutdown()` - Shutdown all tasks

## Backend Serialization

Backend APIs (Soapy, RTL-SDR) can be unsafe when accessed concurrently. The task scheduler uses per-backend semaphores:

- Each backend gets its own semaphore (default: 1 permit = serialized access)
- Tasks acquire a permit before accessing tuners from that backend
- Permit is held for the task's entire lifetime
- Prevents concurrent API calls to the same backend
- Device enumeration tasks also go through the backend queue

With two SDRPlay devices, only one task can access the SDRPlay API at a time, even though both devices are available in the pool.

## Implementation Progress

### Step 1: Create Module Structure ✅
- `src/task/mod.rs` - Module exports
- `src/task/types.rs` - Core types
- `src/task/scan/` - Scan task implementations (band, stations, context)
- `src/task/audio.rs` - AudioTask implementation
- `src/task/enumeration.rs` - DeviceEnumerationTask implementation
- `src/task/scheduler.rs` - TaskScheduler implementation

### Step 2: Define Core Types ✅
- Task enum with ScanBand, ScanStations, Audio, and DeviceEnumeration variants
- TaskType, TaskPriority enums
- TaskHandle, TaskError, TaskStatus types
- Task::run() takes only shutdown: CancellationToken (unified interface)
- Task::backend() method for backend determination

### Step 3: Implement Backend Serialization ✅
- Backend queue infrastructure in TaskScheduler
- Per-backend semaphore support
- determine_backend() method
- acquire_backend_permit() method

### Step 4: Implement DeviceEnumerationTask ✅
- Accepts backend, pool, and discovery_tx
- run() method enumerates devices for Soapy and Mock backends
- Maps Backend enum to concrete backend implementations
- Uses Capabilities::for_device() for device capabilities
- Calls pool.add_device_metadata() for each discovered device
- Emits discovery::Event::Added only for successfully added devices
- Handles all AddDeviceResult variants
- Tests passing (mock, shutdown, unknown backend, USB backend)

### Step 5: Implement ScanBandTask ✅
- Coordinator pattern: doesn't hold tuner, windows acquire per-operation
- run() method with window loop iterating over Band::windows()
- Each window acquires its own tuner via Window::process_with_pool()
- Pause signal polling, shutdown checks
- Phase 2: Full state machine with ScanContext (pause, tune, listen)
- Lifecycle hooks (on_start, on_complete, on_error)

### Step 6: Implement ScanStationsTask ✅
- Similar structure to ScanBandTask
- Takes Vec<f64> frequencies instead of Band
- Creates windows from frequency list (one per station)
- Shares ScanContext state machine with ScanBandTask
- Supports both Phase 1 (simple) and Phase 2 (full state machine)

### Step 7: Implement AudioTask ✅
- Tuner-holder pattern: holds tuner for entire duration
- run() method acquires tuner at start
- Creates AudioSession for persistent audio stream
- Acquires tuner via Segment::new() from pool
- Streams audio until shutdown
- RAII cleanup: AudioSession drop waits for audio graph, then drops segment

### Step 8: Implement TaskScheduler ✅
- submit() with backend serialization
- Backend semaphores ensure only one task per backend at a time
- Tasks spawn on threads with lifecycle hooks
- stop() for per-task cancellation
- status() for querying running tasks
- shutdown() for canceling all tasks
- Per-task cancellation tokens (child of coordinator token)

### Step 9: MainThread Integration & Testing ✅
- MainThread::scan_band() creates ScanBandTask and submits via scheduler
- MainThread::scan_stations() creates ScanStationsTask and submits via scheduler
- Comprehensive test coverage (298 tests passing):
  - Basic task functionality
  - Parallel execution
  - Backend serialization
  - Device enumeration
  - Per-task cancellation
  - Shutdown safety
  - Error handling

## Usage Patterns

### Single Task
```rust
let scan_task = ScanBandTask::new(config, band, progress, pool, shutdown);
let handle = scheduler.submit(Task::ScanBand(scan_task))?;
scheduler.stop(handle.task_id)?;
```

### Parallel Tasks
```rust
let scan_task = Task::ScanBand(ScanBandTask::new(/* ... */));
let audio_task = Task::Audio(AudioTask::new(/* ... */));

scheduler.submit(scan_task)?;
scheduler.submit(audio_task)?;
// Both run in parallel on different tuners
```

### Per-Task Cancellation
```rust
let handle = scheduler.submit(task)?;
handle.cancel();  // Cancel this specific task without affecting others
```

## File Structure

```
src/task/
  mod.rs                   # Module exports
  types.rs                 # Task enum, TaskType, TaskPriority
  scan/
    mod.rs                 # Re-exports
    band.rs                # ScanBandTask
    stations.rs            # ScanStationsTask
    context.rs             # Shared ScanContext state machine
  audio.rs                 # AudioTask
  enumeration.rs           # DeviceEnumerationTask
  scheduler.rs             # TaskScheduler with backend serialization
  tests.rs                 # Integration tests
```

## Key Design Decisions

### Enum Dispatch Over Trait Objects
Uses `Task` enum instead of `dyn SdrTask` trait for ~10x better performance. Enum dispatch eliminates vtable lookups while maintaining clean code organization.

### Backend API Serialization
Critical safety feature using per-backend semaphores to prevent unsafe concurrent API access. Default: 1 concurrent task per backend (safest).

### Per-Task Cancellation
Each task gets its own cancellation token (child of coordinator token). Allows stopping individual tasks without global shutdown.

### Shared State Machine
ScanBandTask and ScanStationsTask share ScanContext implementation, avoiding code duplication while providing different window generation strategies.

## Current Status

All steps 1-9 complete. The unified task architecture has been fully implemented, tested, and validated:

- Task enum with unified `run(shutdown: CancellationToken)` interface
- ScanBandTask and ScanStationsTask with shared state machine
- AudioTask and DeviceEnumerationTask fully functional
- TaskScheduler with backend serialization
- MainThread integration complete
- 298 tests passing
- Production-ready architecture

The refactoring successfully moved operations from MainThread into independent tasks. Both coordinator (scan tasks) and tuner-holder (audio task) patterns work through the scheduler with proper resource management, backend serialization, and state machine support.

## Next Steps

1. **Plan 010**: Multi-SDR Orchestration
2. **Future**: P25 trunking tasks (P25ControlTask, P25VoiceTask)
