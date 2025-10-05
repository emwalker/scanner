# Plan 010: Multi-SDR Orchestration

**Date**: October 2025
**Status**: Not Started
**Dependencies**: All previous plans (005-009)
**Related Plans**: `004-multi-sdr.md` (parent plan)

## Prerequisites Status

- ✅ Plan 005: Backend abstraction complete
- ✅ Plan 006: Device discovery complete
- ⏸️  Plan 007: Device pool not started
- ⏸️  Plan 008: Subprocess IPC not started
- ⏸️  Plan 009: Task abstraction not started

Waiting on Plans 007, 008, and 009 before starting integration.

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

### Step 1: Create Orchestration Layer
**Time**: 2 hours

Create `src/multi/mod.rs`:

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
    pub fn submit_task(&self, task: Box<dyn task::Task>) -> Result<task::TaskId> {
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

### Step 2: Update MainThread to Use Orchestrator
**Time**: 2 hours

Replace device management with orchestrator:

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
        let scan_task = task::ScanTask::new(
            self.config.clone(),
            FrequencyBand::fm(),
            Arc::clone(&self.progress_display) as Arc<dyn ProgressReporter>,
        );

        let scan_task_id = self.orchestrator.submit_task(Box::new(scan_task))?;

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

### Step 3: Update TUI for Multi-SDR Display
**Time**: 3 hours

Update `TuiProgressDisplay` to show real device/task state:

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

## File Structure

```
src/
  multi/
    mod.rs                     # Orchestrator integration layer

examples/
  multi_sdr_scan.rs            # Basic multi-device example
  parallel_scan_audio.rs       # Parallel operation demo
```

## Estimated Time

**Total**: 14-15 hours

- Step 1: Orchestration layer (2 hrs)
- Step 2: Update MainThread (2 hrs)
- Step 3: Update TUI (3 hrs)
- Step 4: Integration testing (3 hrs)
- Step 5: Documentation (2 hrs)
- Step 6: Performance testing (2 hrs)

## Success Criteria

✅ Hot-plug works (devices appear/disappear in TUI)
✅ Parallel operation works with 2+ devices
✅ Single-device operation unchanged (backward compatible)
✅ TUI shows real-time device and task status
✅ All previous tests still pass
✅ Performance acceptable (<100μs latency overhead)
✅ Subprocess cleanup verified (no orphans)

## Migration Path

### Phase 1: Enable Multi-SDR (This Plan)
- All components integrated
- Parallel operation working
- TUI updated

### Phase 2: Optimize (Future)
- Add Seify backend for native RTL-SDR support
- Benchmark vs SoapySDR
- Optimize IPC if needed

### Phase 3: Advanced Features (Future)
- P25 trunking (control + voice channels)
- Direction finding
- Network SDRs
- Advanced scheduling (priority, quality metrics)

## Total Implementation Time

**All Plans Combined**:
- Plan 005: Backend Abstraction (4-6 hrs)
- Plan 006: Device Discovery (5-6 hrs)
- Plan 007: Device Pool (7-8 hrs)
- Plan 008: Subprocess IPC (9-10 hrs)
- Plan 009: Task Abstraction (12-13 hrs)
- Plan 010: Orchestration (14-15 hrs)

**Total**: ~51-58 hours (~2 weeks full-time, ~4-6 weeks part-time)

## Conclusion

This plan completes the multi-SDR architecture by integrating all components into a working system that:

1. **Automatically discovers** devices at runtime
2. **Manages device pool** with RAII guarantees
3. **Isolates devices** in subprocesses for reliability
4. **Schedules tasks** to available devices
5. **Updates TUI** with real-time status

The result is a robust, scalable architecture that works with 1 device (backward compatible) and automatically scales to N devices with parallel operations.
