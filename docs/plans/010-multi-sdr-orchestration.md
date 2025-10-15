# Plan 010: Multi-SDR Orchestration

**Date**: October 2025
**Status**: ✅ Core Complete - Integration tests added
**Dependencies**: All previous plans (005-009) ✅ Complete
**Related Plans**: `004-multi-sdr.md` (parent plan)

## Executive Summary

Bring all multi-SDR components together into working system.

**Components to integrate**:
- Backend Abstraction (Plan 005)
- Device Discovery (Plan 006)
- Device Pool (Plan 007)
- Subprocess IPC (Plan 008)
- Task Abstraction (Plan 009)

**Result**: "Plug in device → automatically discovered → assigned to task → parallel operation"

## Goal

Complete integration demonstrating:
1. **Hot-plug**: Plug in RTL-SDR → automatically available
2. **Parallel ops**: Scan on device #1, listen on device #2
3. **Dynamic TUI**: Real-time device and task status
4. **Graceful degradation**: Works with 1 device, scales to N devices

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│ Main Process                                                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Discovery Thread (Plan 006)                              │ │
│  │   ├─ Linux: udev events                                  │ │
│  │   └─ Other OS: polling                                   │ │
│  └─────────────────┬────────────────────────────────────────┘ │
│                    │ discovery::Event::Added/Removed           │
│                    ↓                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ pool::Pool (Plan 007)                                    │ │
│  │   ├─ Backend Abstraction (Plan 005)                      │ │
│  │   ├─ Device Capabilities                                 │ │
│  │   └─ RAII pool::PooledDevice                             │ │
│  └─────────────────┬────────────────────────────────────────┘ │
│                    │ acquire/release                           │
│                    ↓                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ task::Scheduler (Plan 009)                               │ │
│  │   ├─ task::ScanTask                                      │ │
│  │   └─ task::AudioTask                                     │ │
│  └─────────────────┬────────────────────────────────────────┘ │
│                    │ spawn via ShutdownCoordinator             │
│                    ↓                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Task Threads                                             │ │
│  │   ├─ Uses SubprocessDevice (Plan 008)                    │ │
│  │   └─ Auto-returns device via RAII                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ TUI Display                                              │ │
│  │   ├─ Shows pool status                                   │ │
│  │   ├─ Shows running tasks                                 │ │
│  │   └─ Real-time updates                                   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘

Worker Subprocesses (one per device, Plan 008)
  ├─ RTL-SDR subprocess ──→ Unix socket IPC
  ├─ SDRplay subprocess ──→ Unix socket IPC
  └─ HackRF subprocess ──→ Unix socket IPC
```

## Implementation Steps

### Step 1a: TaskContinuation Pattern ✅ Complete

**Changes**:
- Added `TaskContinuation` enum with `Complete` and `Resubmit` variants
- Updated `Task::run()` signature to return `Result<TaskContinuation>`
- Updated all task implementations to return `TaskContinuation::Complete`
- Updated `TaskScheduler` to handle continuation loop (drops permit on `Resubmit`, reacquires, continues)
- Added `test_task_continuation_pattern` demonstrating continuation pattern works
- All 299 tests pass

**Current state**: Infrastructure in place, all tasks return `Complete` (no yielding yet). Test validates pattern enables state preservation between yields.

### Step 1b: Integration Validation ✅ Complete

**Changes**:
- Added `test_integration_pool_scheduler_discovery` validating pool + scheduler + DeviceEnumerationTask work together
- Test proves DeviceEnumerationTask adds devices to pool and emits discovery events
- MainThread already uses scheduler to submit tasks
- All 300 tests pass

**Current state**: Core multi-SDR components integrated and working. Discovery service exists but not yet started in MainThread.

### Step 1c: Start Discovery Service ✅ Complete (Alternative Implementation)

**Changes**:
- Implemented in `src/cli/discovery.rs` as `start_discovery_service()` function (not as MainThread method)
- Called from `src/cli/scan.rs` in `run_tui_mode()` during application setup
- Discovery service submits DeviceEnumerationTask via TaskScheduler
- Discovery events forwarded to TUI via `TuiEvent::TunerAdded` and `TuiEvent::TunerRemoved`
- All 310 tests pass

**Implementation differs from plan**: Rather than adding methods to MainThread, discovery is wired up in CLI layer. Functionally equivalent but simpler integration.

**Current state**: Discovery service runs automatically on application startup. When devices are plugged in, discovery events trigger DeviceEnumerationTask submission via the scheduler, which updates the pool with new devices under backend serialization.

### Step 1d: Enable Cooperative Yielding in ScanBandTask ✅ Complete

**Changes**:
- Added persistent state fields to `ScanBandTask`: `scanner_state`, `current_playing`, `window_centers`, `windows_to_process`, `window_index`
- Modified `run()` to initialize state on first call and preserve it across yields
- Changed `run()` to process ONE state machine iteration per call instead of looping
- Return `TaskContinuation::Resubmit` to yield between iterations, `Complete` on `Break`
- Added `test_scan_band_cooperative_yielding` proving tasks interleave correctly
- All 301 tests pass

**Current state**: ScanBandTask now yields after each state machine iteration (typically one window). This enables DeviceEnumerationTask and other tasks to interleave during long scans. The backend semaphore is released on yield, allowing fair scheduling.

### Step 1e: Integration Tests ✅ Complete

**File**: `tests/multi_sdr_orchestration_test.rs`

**Tests added** (8 total, all passing):
1. `test_single_device_backward_compatibility` - Validates single-device operation
2. `test_parallel_scan_and_audio` - Multiple tasks run concurrently with 2+ devices
3. `test_cooperative_yielding_allows_interleaving` - Proves DeviceEnumerationTask completes during scan (validates yielding)
4. `test_discovery_to_allocation_flow` - End-to-end: discovery → enumeration → pool → allocation
5. `test_task_scheduler_fairness` - 3 tasks submitted rapidly, all complete without starvation
6. `test_shutdown_during_active_tasks` - Graceful shutdown with multiple active tasks
7. `test_device_enumeration_updates_pool` - DeviceEnumerationTask properly updates pool
8. `test_multiple_scans_sequential` - Sequential scan submission and cancellation

**Test infrastructure**:
- `TestEnv` helper struct for test setup
- Uses `Backend::Mock` (provides 2 fake devices)
- No real hardware required (CI-friendly)
- Fast execution: 1.4 seconds for all 8 tests

**Current state**: All 318 tests pass (310 lib + 8 integration). Core multi-SDR orchestration validated.

---

## What Was Completed

### Core Functionality ✅
1. **TaskContinuation pattern** - Cooperative task yielding (Step 1a)
2. **Pool + TaskScheduler integration** - Components work together (Step 1b)
3. **Discovery service integration** - Automatic device detection (Step 1c, different approach)
4. **Cooperative yielding in ScanBandTask** - Releases backend permit between windows (Step 1d)
5. **Integration tests** - 8 comprehensive tests validating end-to-end behavior (Step 1e)

### Architecture Decisions ✅
- **No Orchestrator struct** - Components wired directly in CLI layer (`src/cli/scan.rs`)
- **Simpler than planned** - Pool, TaskScheduler, and Discovery created separately and passed to MainThread
- **Same functionality** - All planned capabilities work, just without the wrapper struct

### TUI Integration ✅ (Partial)
- `TuiEvent::TunerAdded`, `TuiEvent::TunerRemoved`, `TuiEvent::ActiveTunersUpdated` exist
- Pool has state change callbacks that send events to TUI
- TUI model handles device add/remove events
- Device list updates work

---

## What Was Skipped

### Step 2: Orchestrator Struct ⏭️ Skipped

**Reason**: Unnecessary abstraction. Direct wiring in CLI layer is simpler and equally functional.

**Original plan**: Create `multi::Orchestrator` struct wrapping Pool + TaskScheduler + Discovery

**What exists instead**:
```rust
// src/cli/scan.rs - actual implementation
let pool = Arc::new(Pool::new(filter, log_file));
let scheduler = Arc::new(TaskScheduler::new(pool.clone(), shutdown.clone()));
let discovery = start_discovery_service(tui_tx, shutdown.clone(), scheduler.clone(), pool.clone());
let main_thread = MainThread::new_with_progress(..., pool, scheduler, ...);
```

This achieves the same result without an additional abstraction layer.

### Step 1c: Original Design (Reference Only)

Original design from planning phase (not implemented):

```rust
/// Orchestrates all multi-SDR components
pub struct Orchestrator {
    /// Device pool (RAII)
    pool: Arc<pool::Pool>,

    /// Task scheduler
    scheduler: Arc<task::Scheduler>,

    /// Discovery service
    discovery: Box<dyn discovery::Service>,

    /// Discovery event channel
    discovery_events: (mpsc::Sender<discovery::Event>, mpsc::Receiver<discovery::Event>),

    /// Backend (SoapySDR for now)
    backend: Box<dyn sdr::Backend>,

    /// Shutdown coordination
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

impl Orchestrator {
    pub fn new(shutdown_coordinator: Arc<ShutdownCoordinator>) -> Result<Self> {
        // Create backend
        let backend: Box<dyn sdr::Backend> = Box::new(sdr::Soapy);

        // Create pool
        let pool = pool::Pool::new();

        // Create scheduler
        let scheduler = Arc::new(task::Scheduler::new(
            Arc::clone(&pool),
            Arc::clone(&shutdown_coordinator),
        ));

        // Create discovery service
        let backends = vec![Box::new(sdr::Soapy) as Box<dyn sdr::Backend>];
        let discovery = discovery::create(backends);

        let discovery_events = mpsc::channel();

        Ok(Self {
            pool,
            scheduler,
            discovery,
            discovery_events,
            backend,
            shutdown_coordinator,
        })
    }

    /// Start background services (discovery, etc.)
    pub fn start(&mut self) -> Result<()> {
        // Start discovery service
        let event_tx = self.discovery_events.0.clone();
        let mut discovery = std::mem::replace(
            &mut self.discovery,
            discovery::create(vec![])
        );

        self.shutdown_coordinator.spawn_sdr_thread(move |cancel| {
            discovery.run(event_tx, cancel);
        })?;

        // Start discovery event handler
        let pool = Arc::clone(&self.pool);
        let backend = Box::new(sdr::Soapy) as Box<dyn sdr::Backend>;
        let event_rx = self.discovery_events.1.clone();

        self.shutdown_coordinator.spawn_sdr_thread(move |cancel| {
            Self::handle_discovery_events(pool, backend, event_rx, cancel);
        })?;

        Ok(())
    }

    fn handle_discovery_events(
        pool: Arc<pool::Pool>,
        backend: Box<dyn sdr::Backend>,
        event_rx: mpsc::Receiver<discovery::Event>,
        cancel: CancellationToken,
    ) {
        while !cancel.is_cancelled() {
            match event_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(discovery::Event::Added(info)) => {
                    debug!(device_id = ?info.id, label = %info.label, "Device added");

                    // Open device via backend
                    match backend.open_device(&info.id) {
                        Ok(device) => {
                            // Extract backend name from DeviceId
                            let backend_name = match &info.id {
                                sdr::DeviceId::Backend { backend, .. } => backend.clone(),
                                sdr::DeviceId::Usb { .. } => "USB".to_string(),
                            };
                            if let Err(e) = pool.add_device(device, backend_name) {
                                debug!(error = ?e, "Failed to add device to pool");
                            }
                        }
                        Err(e) => {
                            debug!(error = ?e, "Failed to open device");
                        }
                    }
                }

                Ok(discovery::Event::Removed(id)) => {
                    debug!(device_id = ?id, "Device removed");
                    pool.remove_device(&id);
                }

                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Normal timeout, continue
                }

                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Submit a task for execution
    /// Task will be executed cooperatively, yielding to other tasks as needed
    pub fn submit_task(&self, task: Box<dyn task::Task>) -> Result<task::TaskHandle> {
        self.scheduler.submit(task)
    }

    /// Stop a running task
    pub fn stop_task(&self, task_id: task::TaskId) -> Result<()> {
        self.scheduler.stop(task_id)
    }

    /// Get pool status (for TUI)
    pub fn pool_status(&self) -> pool::PoolStatus {
        self.pool.status()
    }

    /// Get task status (for TUI)
    pub fn task_status(&self) -> Vec<task::TaskStatus> {
        self.scheduler.status()
    }
}
```

### Step 1b: Integrate Discovery with TaskScheduler

**Goal**: Wire Discovery Service to use TaskScheduler for backend enumeration while maintaining discovery event flow

**Current State**:
- Discovery Service has MultiEnumerator that calls backends directly
- No serialization of backend API access
- Unsafe for concurrent discovery events
- Discovery events flow to TUI for device list updates

**New Architecture**:
```rust
impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        // When USB event detected:

        // OLD: self.enumerator.enumerate() → calls backends directly
        // OLD: manually emit discovery events

        // NEW: Submit DeviceEnumerationTask for each backend
        // Pass the discovery event channel to the task so it can emit events
        for backend in [Backend::Soapy, Backend::Mock] {
            let task = Task::DeviceEnumeration(
                DeviceEnumerationTask::new(
                    backend,
                    self.pool.clone(),
                    event_tx.clone(),  // ← Task will emit discovery events
                )
            );
            self.task_scheduler.submit(task)?;
        }

        // Task handles:
        // 1. Pool updates via add_device_metadata()
        // 2. Emitting discovery::Event::Added for successful additions
        // 3. Backend serialization through TaskScheduler
    }
}
```

**Changes**:
1. Discovery Service holds `Arc<TaskScheduler>` reference
2. On discovery trigger (USB event or timer), submit DeviceEnumerationTask per backend
3. Pass `event_tx` (discovery event channel) to DeviceEnumerationTask constructor
4. DeviceEnumerationTask adds devices to pool AND emits discovery events
5. Remove direct backend calls from MultiEnumerator (or keep only for USB inspection)
6. Backend serialization now protects all enumeration
7. Discovery event flow maintained: Task → event_tx → TUI handler → Device list update

**Event Flow**:
```
USB Event → Discovery Service
          ↓
          Submit DeviceEnumerationTask(backend, pool, event_tx)
          ↓
          Task runs (serialized by backend)
          ↓
          For each discovered device:
            1. Add to pool → AddDeviceResult::Added
            2. Emit discovery::Event::Added(device_info)
          ↓
          TUI receives event → Updates device list
```

**Key Insight**: DeviceEnumerationTask bridges two worlds:
- Pool management (allocation state)
- Discovery events (TUI device list)

This maintains clean separation while ensuring both systems stay in sync.

**DeviceEnumerationTask Implementation**:

```rust
impl Task for DeviceEnumerationTask {
    fn run(&mut self, _cancel: CancellationToken) -> Result<TaskContinuation> {
        // Enumerate all devices for this backend
        let devices = self.backend.enumerate_devices()?;

        for device_info in devices {
            // Add to pool
            match self.pool.add_device_metadata(device_info.clone()) {
                Ok(AddDeviceResult::Added) => {
                    // Emit discovery event for TUI
                    self.event_tx.send(Event::Added(device_info))?;
                }
                Ok(AddDeviceResult::AlreadyExists) => {
                    // Skip - already in pool
                }
                Err(e) => {
                    debug!(error = ?e, "Failed to add device to pool");
                }
            }
        }

        // Enumeration is a one-shot task
        Ok(TaskContinuation::Complete)
    }

    fn backend(&self) -> Backend {
        self.backend_type
    }
}
```

**Alternative**: Keep MultiEnumerator for non-backend enumerators (USB inspection), use TaskScheduler only for backend enumeration.

### Step 1c: TaskScheduler with Continuation Pattern

**Goal**: Enable task interleaving through cooperative yielding

**Key insight**: Long-running tasks (like ScanTask) must yield between work units to allow other tasks to run. This is achieved through the continuation pattern where tasks can request resubmission.

**Task Continuation API**:

```rust
/// Controls task execution flow
pub enum TaskContinuation {
    /// Task completed successfully
    Complete,

    /// Task has more work - resubmit to allow other tasks to run
    Resubmit,
}

pub trait Task {
    /// Execute one unit of work
    /// Returns whether task should continue or is complete
    fn run(&mut self, cancel: CancellationToken) -> Result<TaskContinuation>;

    /// Called when task starts
    fn on_start(&mut self) {}

    /// Called when task completes
    fn on_complete(&mut self) {}

    /// Called on error
    fn on_error(&mut self, error: &Error) {}

    /// Backend this task needs (for serialization)
    fn backend(&self) -> Backend;
}
```

**TaskScheduler Implementation**:

```rust
impl TaskScheduler {
    pub fn submit(&self, task: Task) -> Result<TaskHandle> {
        let task_id = TaskId::new();
        let cancel_token = CancellationToken::new();

        // Determine backend for serialization
        let backend = task.backend();

        // Get backend semaphore
        let semaphore = self.backend_semaphores.get(&backend)
            .ok_or("Unknown backend")?
            .clone();

        // Spawn task thread
        let mut task = task;
        let cancel = cancel_token.clone();
        std::thread::spawn(move || {
            task.on_start();

            // Task execution loop - allows cooperative yielding
            loop {
                // Acquire backend permit (blocks if other task using backend)
                let _permit = semaphore.acquire_blocking();

                // Run one unit of work
                match task.run(cancel.clone()) {
                    Ok(TaskContinuation::Complete) => {
                        // Task done - permit released automatically via Drop
                        task.on_complete();
                        break;
                    }
                    Ok(TaskContinuation::Resubmit) => {
                        // Task has more work but yields to allow fairness
                        // Drop permit explicitly to release backend immediately
                        drop(_permit);

                        // Check for cancellation before reacquiring
                        if cancel.is_cancelled() {
                            break;
                        }

                        // Continue loop - will reacquire permit and run again
                        // Other tasks can acquire the permit in between
                        continue;
                    }
                    Err(e) => {
                        task.on_error(&e);
                        break;
                    }
                }
            }
        });

        Ok(TaskHandle::new(task_id, cancel_token))
    }
}
```

**Example: ScanTask with Cooperative Yielding**:

```rust
impl Task for ScanTask {
    fn run(&mut self, cancel: CancellationToken) -> Result<TaskContinuation> {
        // Process ONE window per call
        let window = &self.windows[self.current_window_index];

        // Acquire device from pool
        let device = self.pool.acquire()?;

        // Process this window
        self.process_window(device, window)?;

        // Release device back to pool
        drop(device);

        // Move to next window
        self.current_window_index += 1;

        // More windows to scan?
        if self.current_window_index < self.windows.len() {
            Ok(TaskContinuation::Resubmit)  // Yield, will continue later
        } else {
            Ok(TaskContinuation::Complete)  // All done
        }
    }

    fn backend(&self) -> Backend {
        Backend::Soapy
    }
}
```

**Benefits**:
- **Fairness**: Tasks yield between work units, preventing starvation
- **Interleaving**: DeviceEnumerationTask can run between scan windows
- **Simplicity**: No complex continuation management or async runtime needed
- **Natural state preservation**: Task struct maintains state between yields
- **Backend serialization**: Semaphores ensure only one task per backend at a time

**How Interleaving Works**:

```
Timeline with cooperative yielding:

0.0s:  ScanTask acquires Soapy permit
0.0s:    → Processes window 1 (0.5s)
0.5s:    → Returns TaskContinuation::Resubmit
0.5s:    → Releases Soapy permit
0.5s:    → Loops back to reacquire permit (blocks in queue)

0.5s:  DeviceEnumerationTask acquires Soapy permit (gets it first!)
0.5s:    → Enumerates devices (0.2s)
0.7s:    → Returns TaskContinuation::Complete
0.7s:    → Releases Soapy permit

0.7s:  ScanTask reacquires Soapy permit
0.7s:    → Processes window 2 (0.5s)
1.2s:    → Returns TaskContinuation::Resubmit
1.2s:    → Releases Soapy permit

1.2s:  ScanTask reacquires Soapy permit (no competition)
1.2s:    → Processes window 3 (0.5s)
1.7s:    → Returns TaskContinuation::Complete
1.7s:    → Task ends
```

The key: ScanTask releases the backend semaphore after each window, allowing other Soapy tasks to interleave. Without this, the entire 5-10 minute scan would block all other Soapy operations.

### Step 3: TUI Display for Multi-SDR ⏭️ Partially Complete

**Completed**:
- TUI events exist (`TunerAdded`, `TunerRemoved`, `ActiveTunersUpdated`)
- Pool state callbacks notify TUI of changes
- TUI model tracks device state
- Basic device list functionality works

**Not completed**:
- Enhanced renderer showing device model, task description, running duration
- Sophisticated multi-device status display as described below
- Real-time task progress indicators per device

**Would need**: Updates to TUI renderers to display richer device + task state

### Steps 4-6: Testing, Documentation, Examples ⏭️ Skipped

**Step 4 - Integration testing**: ✅ **DONE** (added 8 comprehensive tests in Step 1e)

**Step 5 - Documentation**: ⏭️ Skipped
- No README updates
- No usage examples (`examples/multi_sdr_scan.rs`, etc.)
- No troubleshooting guide

**Step 6 - Performance testing**: ⏭️ Skipped
- No subprocess overhead measurements
- No parallel throughput validation
- No benchmarking

---

## Original Step 2 Design (Reference Only)

Update MainThread to use Orchestrator:

```rust
pub struct MainThread {
    // Remove old fields:
    // devices: Vec<soapy::Device>,  ❌
    // audio_session: Option<AudioSession>,  ❌

    // Add orchestrator:
    orchestrator: multi::Orchestrator,  ✅

    // Keep essential fields:
    config: ScanningConfig,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    progress_display: Arc<TuiProgressDisplay>,
}

impl MainThread {
    pub fn new(
        config: ScanningConfig,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let progress_display = Arc::new(TuiProgressDisplay::new()?);

        let mut orchestrator = multi::Orchestrator::new(
            Arc::clone(&shutdown_coordinator)
        )?;

        orchestrator.start()?;

        Ok(Self {
            orchestrator,
            config,
            shutdown_coordinator,
            progress_display,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        // Start scanning task
        // Note: ScanTask will yield between windows via TaskContinuation::Resubmit
        let scan_task = task::ScanTask::new(
            self.config.clone(),
            FrequencyBand::fm(),
            Arc::clone(&self.progress_display) as Arc<dyn ProgressReporter>,
        );

        let scan_handle = self.orchestrator.submit_task(Box::new(scan_task))?;

        // Main loop - handle user input
        loop {
            // Update TUI with current status
            self.update_tui()?;

            // Check for user input
            if let Some(station_freq) = self.check_for_tune_request() {
                // User wants to listen to a station
                let audio_task = task::AudioTask::new(
                    station_freq,
                    self.config.clone(),
                    Arc::clone(&self.shutdown_coordinator),
                );

                // Submit audio task (will use second device if available)
                self.orchestrator.submit_task(Box::new(audio_task))?;
            }

            // Check for shutdown
            if self.shutdown_coordinator.token().is_cancelled() {
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        Ok(())
    }

    fn update_tui(&self) -> Result<()> {
        let pool_status = self.orchestrator.pool_status();
        let task_status = self.orchestrator.task_status();

        self.progress_display.update_multi_sdr_status(pool_status, task_status);

        Ok(())
    }
}
```

---

## Original Step 3 Design (Reference Only)

Update TUI for Multi-SDR Display:

```rust
impl TuiProgressDisplay {
    pub fn update_multi_sdr_status(
        &self,
        pool_status: PoolStatus,
        task_status: Vec<TaskStatus>,
    ) {
        let mut tuner_infos = Vec::new();

        // Show devices with allocated tasks
        for task in &task_status {
            let device = pool_status.devices.iter()
                .find(|d| d.id == task.device_id);

            if let Some(device) = device {
                tuner_infos.push(TunerInfo {
                    name: device.model.clone(),
                    frequency: format!("{}", task.description),
                    status: format!("Running: {:.1}s", task.running_duration.as_secs_f64()),
                    signal_strength: None,
                });
            }
        }

        // Show available devices
        for device in &pool_status.devices {
            if device.state == DeviceState::Available {
                tuner_infos.push(TunerInfo {
                    name: device.model.clone(),
                    frequency: "Idle".to_string(),
                    status: format!("Available ({})", device.backend),
                    signal_strength: None,
                });
            }
        }

        self.set_tuner_info(tuner_infos);
    }
}
```

### Step 4: Integration Testing
**Time**: 3 hours

```rust
#[test]
fn test_single_device_operation() {
    // Backward compatibility: should work like before
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let mut orchestrator = multi::Orchestrator::new(coordinator).unwrap();

    orchestrator.start().unwrap();

    // Wait for discovery
    std::thread::sleep(Duration::from_secs(1));

    // Should have at least one device
    assert!(orchestrator.pool_status().available_count > 0);

    // Submit scan task
    let scan_task = task::ScanTask::new(/* ... */);
    let task_id = orchestrator.submit_task(Box::new(scan_task)).unwrap();

    // Should be running
    assert_eq!(orchestrator.task_status().len(), 1);

    // Stop task
    orchestrator.stop_task(task_id).unwrap();
}

#[test]
fn test_parallel_operation() {
    // Requires 2+ devices connected
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let mut orchestrator = multi::Orchestrator::new(coordinator).unwrap();

    orchestrator.start().unwrap();
    std::thread::sleep(Duration::from_secs(1));

    let pool_status = orchestrator.pool_status();
    if pool_status.available_count < 2 {
        println!("Skipping: requires 2+ devices");
        return;
    }

    // Submit scan task
    let scan_task = task::ScanTask::new(/* ... */);
    let scan_id = orchestrator.submit_task(Box::new(scan_task)).unwrap();

    // Submit audio task
    let audio_task = task::AudioTask::new(88.9e6, /* ... */);
    let audio_id = orchestrator.submit_task(Box::new(audio_task)).unwrap();

    // Both should be running
    assert_eq!(orchestrator.task_status().len(), 2);

    // Should be on different devices
    let tasks = orchestrator.task_status();
    let scan_device = tasks.iter().find(|t| t.task_id == scan_id).unwrap().device_id.clone();
    let audio_device = tasks.iter().find(|t| t.task_id == audio_id).unwrap().device_id.clone();

    assert_ne!(scan_device, audio_device);
}

#[test]
#[ignore]  // Manual test: cargo test test_hotplug -- --ignored
fn test_hotplug() {
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let mut orchestrator = multi::Orchestrator::new(coordinator).unwrap();

    orchestrator.start().unwrap();

    println!("Initial devices:");
    let initial = orchestrator.pool_status();
    println!("  Available: {}", initial.available_count);

    println!("\nPlug in a new device...");
    std::thread::sleep(Duration::from_secs(10));

    let after_add = orchestrator.pool_status();
    println!("After plug:");
    println!("  Available: {}", after_add.available_count);

    assert!(after_add.available_count > initial.available_count);

    println!("\nUnplug the device...");
    std::thread::sleep(Duration::from_secs(10));

    let after_remove = orchestrator.pool_status();
    println!("After unplug:");
    println!("  Available: {}", after_remove.available_count);

    assert_eq!(after_remove.available_count, initial.available_count);
}

#[test]
fn test_task_continuation_interleaving() {
    // Test that tasks properly yield via TaskContinuation::Resubmit
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let mut orchestrator = multi::Orchestrator::new(coordinator).unwrap();

    orchestrator.start().unwrap();
    std::thread::sleep(Duration::from_secs(1));

    // Track execution order
    let execution_log = Arc::new(Mutex::new(Vec::new()));

    // Create a multi-window scan task that yields between windows
    let log1 = Arc::clone(&execution_log);
    let scan_task = MockScanTask::new(3, move |window| {
        log1.lock().unwrap().push(format!("scan-window-{}", window));
    });

    // Create a quick enumeration task
    let log2 = Arc::clone(&execution_log);
    let enum_task = MockEnumerationTask::new(move || {
        log2.lock().unwrap().push("enumeration".to_string());
    });

    // Submit scan task first
    let scan_handle = orchestrator.submit_task(Box::new(scan_task)).unwrap();

    // Wait for first window to complete
    std::thread::sleep(Duration::from_millis(100));

    // Submit enumeration task - should interleave between scan windows
    let enum_handle = orchestrator.submit_task(Box::new(enum_task)).unwrap();

    // Wait for both to complete
    scan_handle.wait().unwrap();
    enum_handle.wait().unwrap();

    // Check execution log
    let log = execution_log.lock().unwrap();
    println!("Execution order: {:?}", log);

    // Should see interleaving:
    // scan-window-0, enumeration, scan-window-1, scan-window-2
    // (or other interleaved patterns)
    assert_eq!(log.len(), 4);
    assert!(log.contains(&"scan-window-0".to_string()));
    assert!(log.contains(&"scan-window-1".to_string()));
    assert!(log.contains(&"scan-window-2".to_string()));
    assert!(log.contains(&"enumeration".to_string()));

    // Enumeration should NOT be at the end (would indicate no interleaving)
    assert_ne!(log[3], "enumeration");
}
```

### Step 5: Documentation and Examples
**Time**: 2 hours

1. Update README with multi-SDR usage
2. Add examples:
   - `examples/multi_sdr_scan.rs` - Basic multi-device scanning
   - `examples/parallel_scan_audio.rs` - Scan + audio simultaneously
3. Document behavior with different device counts
4. Add troubleshooting guide

### Step 6: Performance Testing
**Time**: 2 hours

Verify performance with multiple devices:

```rust
#[test]
fn test_subprocess_overhead() {
    // Measure latency overhead of subprocess IPC
    let device = SubprocessDevice::new(/* ... */).unwrap();

    let start = Instant::now();
    for _ in 0..1000 {
        device.read_samples().unwrap();
    }
    let elapsed = start.elapsed();

    let avg_latency = elapsed / 1000;
    println!("Average I/Q read latency: {:?}", avg_latency);

    // Should be < 100μs
    assert!(avg_latency < Duration::from_micros(100));
}

#[test]
fn test_parallel_throughput() {
    // Verify multiple devices don't interfere
    let device1 = SubprocessDevice::new(/* RTL-SDR */).unwrap();
    let device2 = SubprocessDevice::new(/* SDRplay */).unwrap();

    let (tx1, rx1) = mpsc::channel();
    let (tx2, rx2) = mpsc::channel();

    // Read from both simultaneously
    std::thread::spawn(move || {
        for _ in 0..100 {
            tx1.send(device1.read_samples().unwrap()).unwrap();
        }
    });

    std::thread::spawn(move || {
        for _ in 0..100 {
            tx2.send(device2.read_samples().unwrap()).unwrap();
        }
    });

    // Both should complete without blocking each other
    let mut count1 = 0;
    let mut count2 = 0;

    loop {
        select! {
            recv(rx1) -> _ => count1 += 1,
            recv(rx2) -> _ => count2 += 1,
        }

        if count1 >= 100 && count2 >= 100 {
            break;
        }
    }

    assert_eq!(count1, 100);
    assert_eq!(count2, 100);
}
```

## Expected Behaviors

### With 1 Device
- Same as current behavior
- Scanning and audio are mutually exclusive (pause scan to listen)
- Backward compatible

### With 2 Devices
- **New capability**: Scan on device #1, listen on device #2
- No pause required
- Both operations continue simultaneously

### With 3+ Devices
- Can run multiple tasks: scan + multiple audio streams
- Or: P25 control channel + voice channels (future)
- Limited only by available devices

## User Experience

### Startup
```
Initializing scanner...
Discovering devices...
Found 2 devices:
  - SDRplay RSPduo (Tuner 1)
  - RTL-SDR (Serial: 00000001)
Starting discovery service...
Ready.
```

### Hot-plug
```
[During operation]
Device added: HackRF One (Serial: 0000000000...)
  Added to pool (3 devices available)

[User unplugs RTL-SDR]
Device removed: RTL-SDR (Serial: 00000001)
  Removed from pool (2 devices available)
```

### TUI Display
```
╭─ SDRplay RSPduo (Tuner 1) ─────────────────────────────╮
│ Scanning: FM Band (88-108 MHz)                        │
│ Status: Window 15/20 • Running 45.2s                   │
╰────────────────────────────────────────────────────────╯

╭─ RTL-SDR (Serial: 00000001) ───────────────────────────╮
│ Audio: 88.9 MHz FM                                     │
│ Status: Playing • Running 12.5s                        │
│ Signal: ▂▃▅▇ 85%                                       │
╰────────────────────────────────────────────────────────╯

╭─ HackRF One (Serial: 0000000...) ──────────────────────╮
│ Idle                                                   │
│ Status: Available                                      │
╰────────────────────────────────────────────────────────╯
```

## Actual File Structure

```
src/
  task/
    scheduler.rs               # TaskScheduler with backend serialization
    scan/band.rs              # ScanBandTask with cooperative yielding
    enumeration.rs            # DeviceEnumerationTask
  cli/
    discovery.rs              # start_discovery_service() function
    scan.rs                   # Wires Pool + TaskScheduler + Discovery
  hardware/pool/              # Device pool with RAII
  discovery/                  # Discovery service implementations

tests/
  multi_sdr_orchestration_test.rs  # 8 integration tests
```

No `src/multi/` directory - components wired in CLI layer instead.
No examples - would need to be added for documentation.

## Success Criteria - Status

✅ **Hot-plug works** - Discovery service running, DeviceEnumerationTask updates pool
✅ **Parallel operation** - TaskScheduler supports concurrent tasks on different devices
✅ **Backward compatible** - Single device works as before (validated by tests)
✅ **TUI events** - Device add/remove events flow to TUI model
⚠️ **TUI display** - Basic events work, but no rich multi-device renderer
✅ **All tests pass** - 318 tests (310 lib + 8 integration)
⏭️ **Performance benchmarks** - Not measured (subprocess overhead untested)
✅ **Cleanup verified** - Pool shutdown handling tested, subprocess cleanup via existing tests

---

## Final Summary

### What This Plan Achieved

The multi-SDR orchestration is **functionally complete**:

1. **Core Architecture** ✅
   - TaskScheduler with backend serialization working
   - Pool provides RAII-managed tuners to tasks
   - Discovery service automatically detects and enumerates devices
   - Cooperative yielding enables fair task scheduling

2. **Integration** ✅
   - Components wired together in CLI layer (simpler than planned Orchestrator struct)
   - DeviceEnumerationTask updates pool when discovery events occur
   - TUI receives device add/remove events
   - MainThread submits tasks via TaskScheduler

3. **Testing** ✅
   - 8 comprehensive integration tests validate end-to-end behavior
   - Tests use Backend::Mock (no hardware required, CI-friendly)
   - All 318 tests pass

4. **Key Innovation** ✅
   - TaskContinuation pattern enables cooperative multitasking without async runtime
   - ScanBandTask yields between windows, allowing device enumeration to interleave
   - Backend semaphores prevent API conflicts while allowing parallelism

### Architectural Simplification

Rather than implementing the planned `Orchestrator` struct, components are wired directly:
```rust
// In src/cli/scan.rs
let pool = Arc::new(Pool::new(filter, log_file));
let scheduler = Arc::new(TaskScheduler::new(pool, shutdown));
let discovery = start_discovery_service(tui_tx, shutdown, scheduler, pool);
```

This achieves the same functionality with less indirection.

### What Remains

**For production use**:
- Enhanced TUI renderer showing device + task state
- Usage documentation and examples
- Performance benchmarking

**For advanced features**:
- Task priority scheduling (infrastructure exists, not used)
- Parallel audio on multiple devices
- Advanced scheduling policies

### Key Takeaway

The multi-SDR architecture is **working and validated**. All planned components exist and integrate correctly. The system can discover devices, allocate them to tasks, run multiple tasks concurrently, and handle graceful shutdown. Integration tests prove these behaviors work correctly.
