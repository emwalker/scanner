# Plan 009: Task Abstraction

**Date**: October 2025
**Status**: Not Started
**Dependencies**: ✅ `005-backend-abstraction.md`, `007-device-pool.md`
**Related Plans**: `004-multi-sdr.md` (parent plan), ✅ `003-structured-concurrency-shutdown.md`
**Enables**: Plan 010

## Prerequisites Status

- ✅ Plan 005: Backend abstraction complete
- ⏸️  Plan 007: Device pool not started (required for this plan)
- ✅ Plan 003: Structured concurrency complete

Waiting on Plan 007 (Device Pool) before starting.

## Executive Summary

Transform operations (scanning, audio playback) from monolithic functions into independent, reusable tasks.

**Key benefit**: Tasks can run in parallel on different devices automatically.

## Problem Statement

Current architecture has operations embedded in `MainThread`:
```rust
impl MainThread {
    fn scan_band(&mut self, device: &soapy::Device) -> Result<()> {
        // 200+ lines of scanning logic hardcoded here
        // Tightly coupled to MainThread state
        // Can't run on different device
        // Can't run in parallel with other operations
    }
}
```

**Issues**:
- Operations tightly coupled to `MainThread`
- Can't run multiple operations in parallel
- Hard to test in isolation
- Device assignment is hardcoded

## Goal

Operations as independent, composable tasks:
```rust
// Create tasks
let scan_task = ScanTask::new(config, band);
let audio_task = AudioTask::new(station_freq, config);

// Submit to scheduler (automatically acquires devices)
scheduler.submit(scan_task)?;
scheduler.submit(audio_task)?;

// Both run in parallel on separate devices!
```

## Design

### Core Trait

```rust
/// Abstraction for operations that use SDR devices
pub trait SdrTask: Send {
    /// Requirements this task needs from a device
    fn requirements(&self) -> TaskRequirements;

    /// Run the task with provided device
    fn run(&mut self, device: PooledDevice, shutdown: CancellationToken) -> Result<()>;

    /// Task type identifier
    fn task_type(&self) -> TaskType;

    /// Human-readable description for TUI
    fn description(&self) -> String;
}

/// Requirements for device acquisition
#[derive(Clone, Debug)]
pub struct TaskRequirements {
    pub frequency_hz: f64,
    pub bandwidth_hz: f64,
    pub required_sample_rate: f64,
    pub priority: TaskPriority,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskType {
    Scanning,
    Audio,
    P25Control,   // Future
    P25Voice,     // Future
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,       // Background scanning
    Normal,    // Regular audio
    High,      // P25 control channel (critical)
}
```

### ScanTask Implementation

```rust
/// Band scanning task (extracted from MainThread::scan_band)
pub struct ScanTask {
    config: ScanningConfig,
    band: FrequencyBand,
    progress_reporter: Arc<dyn ProgressReporter>,
    scanner_state: ScannerState,
    pause_signal: PauseSignal,
}

impl ScanTask {
    pub fn new(
        config: ScanningConfig,
        band: FrequencyBand,
        progress_reporter: Arc<dyn ProgressReporter>,
    ) -> Self {
        Self {
            config,
            band,
            progress_reporter,
            scanner_state: ScannerState::new(),
            pause_signal: PauseSignal::new(),
        }
    }

    /// Access to pause signal (for external control)
    pub fn pause_signal(&self) -> &PauseSignal {
        &self.pause_signal
    }
}

impl SdrTask for ScanTask {
    fn requirements(&self) -> TaskRequirements {
        TaskRequirements {
            frequency_hz: self.band.center_frequency(),
            bandwidth_hz: self.config.bandwidth,
            required_sample_rate: self.config.samp_rate,
            priority: TaskPriority::Low,  // Background scanning
        }
    }

    fn run(&mut self, device: PooledDevice, shutdown: CancellationToken) -> Result<()> {
        debug!("Starting scan task on device: {}", device.capabilities().model);

        // Start scanning
        self.scanner_state.start_window(0);

        loop {
            // Check for shutdown
            if shutdown.is_cancelled() {
                self.scanner_state.shutdown();
            }

            // Check for pause
            if self.pause_signal.is_paused() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            // Process current state
            match &self.scanner_state.mode {
                ScanMode::ShuttingDown => break,

                ScanMode::Scanning => {
                    // Scan current window
                    let window_idx = self.scanner_state.current_window_index();
                    let window = self.band.window(window_idx);

                    // Tune device to window center
                    device.as_device_mut().tune(window.center_freq)?;

                    // Create segment for this window
                    let segment = /* ... create from device ... */;

                    // Process window (existing logic)
                    let mut window_processor = Window::new(
                        window,
                        &self.config,
                        &self.scanner_state,
                    );

                    window_processor.process(segment)?;

                    // Move to next window
                    self.scanner_state.complete_window();
                }

                _ => {}
            }
        }

        debug!("Scan task completed");
        Ok(())
    }

    fn task_type(&self) -> TaskType {
        TaskType::Scanning
    }

    fn description(&self) -> String {
        format!("Scanning: {} ({:.1}-{:.1} MHz)",
            self.band.name(),
            self.band.start_freq() / 1e6,
            self.band.end_freq() / 1e6,
        )
    }
}
```

### AudioTask Implementation

```rust
/// Audio streaming task (extracted from AudioSession)
pub struct AudioTask {
    station_freq: f64,
    config: ScanningConfig,
    audio_session: Option<AudioSession>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

impl AudioTask {
    pub fn new(
        station_freq: f64,
        config: ScanningConfig,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            station_freq,
            config,
            audio_session: None,
            shutdown_coordinator,
        }
    }
}

impl SdrTask for AudioTask {
    fn requirements(&self) -> TaskRequirements {
        TaskRequirements {
            frequency_hz: self.station_freq,
            bandwidth_hz: 200_000.0,  // FM bandwidth
            required_sample_rate: self.config.samp_rate,
            priority: TaskPriority::Normal,
        }
    }

    fn run(&mut self, device: PooledDevice, shutdown: CancellationToken) -> Result<()> {
        debug!(
            freq_mhz = self.station_freq / 1e6,
            device = device.capabilities().model,
            "Starting audio task"
        );

        // Tune device to station
        device.as_device_mut().tune(self.station_freq)?;

        // Create segment for audio
        let segment = /* ... create from device ... */;

        // Create audio session
        let mut session = AudioSession::new(
            &self.config,
            Arc::clone(&self.shutdown_coordinator),
        )?;

        session.tune_to_station(segment)?;

        // Keep playing until shutdown
        while !shutdown.is_cancelled() {
            std::thread::sleep(Duration::from_millis(100));
        }

        debug!("Audio task stopping");
        Ok(())
    }

    fn task_type(&self) -> TaskType {
        TaskType::Audio
    }

    fn description(&self) -> String {
        format!("Audio: {:.1} MHz FM", self.station_freq / 1e6)
    }
}
```

### TaskScheduler (Simple Version)

```rust
/// Schedules tasks to available devices
pub struct TaskScheduler {
    sdr_pool: Arc<SdrPool>,
    running_tasks: Arc<Mutex<HashMap<TaskId, RunningTaskInfo>>>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

/// Task ID for tracking
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

/// Information about a running task
struct RunningTaskInfo {
    task_type: TaskType,
    description: String,
    device_id: DeviceId,
    started_at: Instant,
}

impl TaskScheduler {
    pub fn new(
        sdr_pool: Arc<SdrPool>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            sdr_pool,
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            shutdown_coordinator,
        }
    }

    /// Submit task for execution
    pub fn submit(&self, mut task: Box<dyn SdrTask>) -> Result<TaskId> {
        let requirements = task.requirements();
        let task_type = task.task_type();
        let description = task.description();

        debug!(
            task_type = ?task_type,
            description = description,
            "Submitting task"
        );

        // Acquire suitable device from pool
        let device = self.sdr_pool.acquire(&requirements)
            .map_err(|_| ScannerError::NoAvailableDevice(requirements.clone()))?;

        let device_id = device.capabilities().device_id.clone();
        let task_id = TaskId::new();

        // Track task
        let running_tasks = Arc::clone(&self.running_tasks);
        running_tasks.lock().unwrap().insert(task_id, RunningTaskInfo {
            task_type,
            description: description.clone(),
            device_id: device_id.clone(),
            started_at: Instant::now(),
        });

        // Spawn task thread
        let running_tasks_clone = Arc::clone(&running_tasks);
        self.shutdown_coordinator.spawn_sdr_thread(move |shutdown| {
            debug!(task_id = ?task_id, "Task thread started");

            match task.run(device, shutdown) {
                Ok(_) => {
                    debug!(task_id = ?task_id, "Task completed successfully");
                }
                Err(e) => {
                    debug!(task_id = ?task_id, error = ?e, "Task failed");
                }
            }

            // Remove from running tasks
            running_tasks_clone.lock().unwrap().remove(&task_id);

            debug!(task_id = ?task_id, "Task thread exited");
        })?;

        debug!(task_id = ?task_id, device = ?device_id, "Task started");
        Ok(task_id)
    }

    /// Stop a running task
    pub fn stop(&self, task_id: TaskId) -> Result<()> {
        // Task will be stopped via shutdown coordinator
        // Device will be auto-returned via RAII when task exits

        debug!(task_id = ?task_id, "Stopping task");

        // Note: In a more sophisticated implementation, we could send a
        // task-specific cancellation token, but for now we rely on
        // the ShutdownCoordinator's global shutdown

        Ok(())
    }

    /// Get status of all running tasks
    pub fn status(&self) -> Vec<TaskStatus> {
        self.running_tasks.lock().unwrap()
            .iter()
            .map(|(id, info)| TaskStatus {
                task_id: *id,
                task_type: info.task_type,
                description: info.description.clone(),
                device_id: info.device_id.clone(),
                running_duration: info.started_at.elapsed(),
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct TaskStatus {
    pub task_id: TaskId,
    pub task_type: TaskType,
    pub description: String,
    pub device_id: DeviceId,
    pub running_duration: Duration,
}
```

## Implementation Steps

### Step 1: Create Module Structure
**Time**: 30 minutes

1. Create `src/task/mod.rs`
2. Create `src/task/traits.rs` - `SdrTask` trait
3. Create `src/task/scan_task.rs` - `ScanTask` implementation
4. Create `src/task/audio_task.rs` - `AudioTask` implementation
5. Create `src/task/scheduler.rs` - `TaskScheduler`

### Step 2: Define SdrTask Trait
**Time**: 1 hour

1. Define `SdrTask` trait
2. Define `TaskRequirements`, `TaskType`, `TaskPriority`
3. Add documentation and examples

### Step 3: Extract ScanTask
**Time**: 3 hours

This is the most complex step - extracting 200+ lines from `MainThread::scan_band()`.

1. Create `ScanTask` struct
2. Move scanning logic from `MainThread` to `ScanTask::run()`
3. Update to use provided `PooledDevice` instead of `self.devices[0]`
4. Add pause signal handling
5. Integrate with `ScannerState`
6. Test single-device scanning still works

### Step 4: Extract AudioTask
**Time**: 2 hours

1. Create `AudioTask` struct
2. Extract audio logic from `AudioSession` usage
3. Update to use provided `PooledDevice`
4. Test single-device audio playback

### Step 5: Implement TaskScheduler
**Time**: 2 hours

1. Create `TaskScheduler` struct
2. Implement `submit()` - acquire device, spawn thread
3. Implement `stop()` - cancel task
4. Implement `status()` - query running tasks
5. Integrate with `ShutdownCoordinator`

### Step 6: Update MainThread
**Time**: 2 hours

Replace direct calls with task submission:

**Before**:
```rust
impl MainThread {
    fn scan_band(&mut self, device: &soapy::Device) -> Result<()> {
        // ... 200 lines of scanning logic
    }
}
```

**After**:
```rust
impl MainThread {
    fn start_scanning(&mut self) -> Result<TaskId> {
        let task = ScanTask::new(
            self.config.clone(),
            self.band.clone(),
            self.progress_reporter.clone(),
        );

        self.scheduler.submit(Box::new(task))
    }

    fn start_audio(&mut self, station_freq: f64) -> Result<TaskId> {
        let task = AudioTask::new(
            station_freq,
            self.config.clone(),
            Arc::clone(&self.shutdown_coordinator),
        );

        self.scheduler.submit(Box::new(task))
    }
}
```

### Step 7: Testing
**Time**: 2 hours

```rust
#[test]
fn test_scan_task() {
    let config = test_config();
    let band = FrequencyBand::fm();
    let progress = Arc::new(NullProgressReporter);

    let mut task = ScanTask::new(config, band, progress);
    let device = create_test_device();
    let cancel = CancellationToken::new();

    // Run task in background
    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        task.run(device, cancel_clone).unwrap();
    });

    // Let it run a bit
    std::thread::sleep(Duration::from_millis(500));

    // Cancel
    cancel.cancel();
}

#[test]
fn test_audio_task() {
    let config = test_config();
    let coordinator = Arc::new(ShutdownCoordinator::new());

    let mut task = AudioTask::new(88.9e6, config, coordinator);
    let device = create_test_device();
    let cancel = CancellationToken::new();

    // Run task
    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        task.run(device, cancel_clone).unwrap();
    });

    std::thread::sleep(Duration::from_millis(500));
    cancel.cancel();
}

#[test]
fn test_task_scheduler() {
    let pool = create_test_pool();
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool, coordinator);

    // Submit scan task
    let scan_task = Box::new(ScanTask::new(/* ... */));
    let task_id = scheduler.submit(scan_task).unwrap();

    // Should be running
    let status = scheduler.status();
    assert_eq!(status.len(), 1);
    assert_eq!(status[0].task_id, task_id);
    assert_eq!(status[0].task_type, TaskType::Scanning);

    // Stop task
    scheduler.stop(task_id).unwrap();

    // Give it time to stop
    std::thread::sleep(Duration::from_millis(200));

    // Should no longer be running
    assert_eq!(scheduler.status().len(), 0);
}

#[test]
fn test_parallel_tasks() {
    let pool = create_test_pool_with_2_devices();
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let scheduler = TaskScheduler::new(pool, coordinator);

    // Submit scan task
    let scan_task = Box::new(ScanTask::new(/* ... */));
    let scan_id = scheduler.submit(scan_task).unwrap();

    // Submit audio task (should get different device)
    let audio_task = Box::new(AudioTask::new(88.9e6, /* ... */));
    let audio_id = scheduler.submit(audio_task).unwrap();

    // Both should be running
    let status = scheduler.status();
    assert_eq!(status.len(), 2);

    // Verify they're on different devices
    let scan_device = status.iter().find(|s| s.task_id == scan_id).unwrap().device_id.clone();
    let audio_device = status.iter().find(|s| s.task_id == audio_id).unwrap().device_id.clone();

    assert_ne!(scan_device, audio_device);
}
```

## Benefits

### Architectural
✅ **Separation of concerns** - Tasks independent of MainThread
✅ **Reusability** - Tasks can be used in different contexts
✅ **Testability** - Tasks testable in isolation
✅ **Composability** - Multiple tasks can run together

### Parallel Execution
✅ **Automatic** - Scheduler handles device acquisition
✅ **Transparent** - No manual device management
✅ **Efficient** - Best device selected for each task

### Maintainability
✅ **Smaller files** - Each task in own file
✅ **Clearer code** - Task responsibility is clear
✅ **Easier debugging** - Isolated task behavior

## Usage Patterns

### Single Task
```rust
let scan_task = ScanTask::new(config, band, progress);
let task_id = scheduler.submit(Box::new(scan_task))?;

// Task runs until completed or stopped
scheduler.stop(task_id)?;
```

### Parallel Tasks
```rust
// With 2+ devices: both run simultaneously
let scan_id = scheduler.submit(Box::new(scan_task))?;
let audio_id = scheduler.submit(Box::new(audio_task))?;

// With 1 device: second task waits for device to be available
```

### Fallback to Sequential (1 Device)
```rust
if pool.available_count() == 1 {
    // Pause scan before starting audio (current behavior)
    scan_task.pause_signal().pause();
    let audio_id = scheduler.submit(Box::new(audio_task))?;
    // Resume scan when audio stops
}
```

## File Structure

```
src/
  task/
    mod.rs              # Module exports
    traits.rs           # SdrTask trait, requirements
    scan_task.rs       # ScanTask implementation
    audio_task.rs      # AudioTask implementation
    scheduler.rs       # TaskScheduler
    p25_control.rs     # Future: P25 control channel task
    p25_voice.rs       # Future: P25 voice channel task
```

## Estimated Time

**Total**: 12-13 hours

- Step 1: Module structure (30 min)
- Step 2: SdrTask trait (1 hr)
- Step 3: Extract ScanTask (3 hrs)
- Step 4: Extract AudioTask (2 hrs)
- Step 5: TaskScheduler (2 hrs)
- Step 6: Update MainThread (2 hrs)
- Step 7: Testing (2 hrs)

## Success Criteria

✅ `ScanTask` extracted from `MainThread`
✅ `AudioTask` extracted from audio logic
✅ `TaskScheduler` submits and tracks tasks
✅ Single-device operation still works (backward compatible)
✅ Parallel execution works with 2+ devices
✅ RAII device cleanup verified
✅ All existing tests pass

## Next Steps

After completing this plan:
1. **Plan 010**: Multi-SDR Orchestration (ties everything together)
2. **Future**: P25 trunking tasks (P25ControlTask, P25VoiceTask)
