# Plan 011: Hot-Plug Device Support

**Date**: October 2025
**Status**: Not Started
**Dependencies**: ✅ `007-device-pool.md`, ✅ `006-device-discovery.md`
**Related Plans**: `004-multi-sdr.md`, `007-device-pool.md`

## Executive Summary

Implement true hot-plug support for SDR devices, allowing devices to be added and removed at runtime while the application is running. This builds on the discovery service (Plan 006) and device pool (Plan 007) to provide seamless device management.

**Key Challenge**: Distinguishing initial device enumeration from true hot-plug events.

## Problem Statement

The current implementation has discovery infrastructure and pool management, but hot-plug integration revealed design issues that need to be addressed:

### 1. Initial Enumeration vs Hot-Plug Events

**Problem**: Discovery service fires events for ALL devices found during initial enumeration
- These aren't "hot-plug" events - they're baseline inventory
- Current code treats them as "new" devices and tries to add them to pool
- But the selected device is already in pool from startup!
- **Result**: Duplicate device addition attempts, conflicts with SingleTuner mode

**Example**:
```
Startup: Add SDRplay RSPduo (driver=sdrplay, mode=DT) to pool
Discovery initial scan fires Event::Added for:
  - RSPduo (driver=sdrplay, mode=DT)  ← Duplicate of startup device!
  - RSPduo (driver=sdrplay, mode=MA)
  - RSPduo (driver=sdrplay, mode=MA8)
  - RSPduo (driver=sdrplay, mode=ST)
  - RTL-SDR (driver=rtlsdr)

Pool tries to add all 5 devices, but:
  - First one is duplicate
  - Pool filter rejects RTL-SDR (only sdrplay allowed)
  - SingleTuner mode rejects variants (already allocated)
```

### 2. SingleTuner Mode Conflicts

**Problem**: Pool configured with `.with_mode(TuningMode::SingleTuner)`
- Initial device added to pool and allocated for scanning
- Discovery then fires events for 4 more SDRplay variants (same hardware, different modes)
- Pool correctly rejects them: "SingleTuner mode and 1 tuner already allocated"
- **But**: We're opening each device before checking if it should be added
- **Result**: Wasted resources opening devices that will be immediately rejected

### 3. Filter Duplication Problem

**Problem**: Two places checking the same filter logic
- Pool has filter: `PoolFilter::new().with_driver("sdrplay").with_mode(TuningMode::SingleTuner)`
- Discovery handler was manually checking `backend.eq_ignore_ascii_case("sdrplay")`
- This duplicates filter logic and can get out of sync
- **Better approach**: Let pool check filter, but check BEFORE opening device

### 4. Device Opening is Expensive

**Problem**: Opening devices that will be filtered out wastes resources
- Each `backend.open_device()` call initializes hardware
- Pool filter check happens AFTER device is opened
- Example: Opening RTL-SDR just to have it rejected by filter
- **Better approach**: Pre-filter based on DeviceId before opening

### 5. Duplicate Device Variants

**Problem**: SDRplay RSPduo appears as 4 different "devices" (DT, MA, MA8, ST)
- These are different tuning modes of the same physical hardware
- Pool needs to handle: "already have this hardware, skip duplicate"
- Or: treat each mode as separate device (current behavior)
- Need clear policy on handling variants

## Design

### Event Flow Architecture

With the integration of DeviceEnumerationTask (Plan 009), discovery events can come from two sources:

1. **Direct Discovery Events**: USB hotplug detection (udev on Linux)
2. **Backend Enumeration Events**: Emitted by DeviceEnumerationTask after polling backend APIs

Both flow through the same `mpsc::Sender<discovery::Event>` channel to the TUI, maintaining unified event handling.

**Key Principle**: Discovery events update the TUI device list. Pool allocation changes (via `add_device_metadata()` or `acquire()`) update tuner status but do NOT directly trigger TUI updates. This maintains clean separation:
- Discovery events → TUI device list (add/remove devices)
- Pool state changes → TUI tuner status (idle/busy/etc.)

### 1. Discovery Event Distinction

Add flag to distinguish initial enumeration from hot-plug:

```rust
pub enum Event {
    Added {
        info: DeviceInfo,
        is_initial_scan: bool,  // ← NEW: true during startup enumeration
    },
    Removed {
        id: DeviceId,
    },
}
```

**Alternative approach**: Complete discovery BEFORE main thread starts
```rust
// In main.rs
let discovery_service = DiscoveryService::new();
let initial_devices = discovery_service.enumerate_blocking()?;  // Blocks until complete

// Now start hot-plug monitoring
let (event_tx, event_rx) = mpsc::channel();
discovery_service.start_monitoring(event_tx);  // Only fires for actual changes
```

**Note on DeviceEnumerationTask**: When Discovery Service submits DeviceEnumerationTask, it passes the discovery event channel. The task emits `Event::Added` for successfully added devices, maintaining the unified event flow.

### 2. Pre-filter Before Opening

Add method to check filter without opening device:

```rust
impl Pool {
    /// Check if a device should be added without opening it
    ///
    /// This allows discovery to pre-filter before expensive device opening.
    pub fn should_add_device(
        &self,
        device_id: &DeviceId,
        backend_name: &str
    ) -> ShouldAddResult {
        // Check shutdown first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return ShouldAddResult::ShutdownMode;
        }

        // Check if device already in pool
        if let Ok(inner) = self.pool_ref.try_lock() {
            if inner.devices.contains_key(device_id) {
                return ShouldAddResult::AlreadyExists;
            }

            // Check filter (without opening device)
            let test_tuner_id = TunerId::new(device_id.clone(), 0);
            let allocated_count = inner.allocated_tuners.len();

            if !self.filter.is_allowed(&test_tuner_id, backend_name, allocated_count) {
                return ShouldAddResult::FilteredOut {
                    reason: format!("Filter does not allow backend '{}'", backend_name),
                };
            }

            ShouldAddResult::ShouldAdd
        } else {
            ShouldAddResult::PoolBusy
        }
    }
}

pub enum ShouldAddResult {
    ShouldAdd,
    AlreadyExists,
    FilteredOut { reason: String },
    ShutdownMode,
    PoolBusy,
}
```

**Usage in discovery handler**:
```rust
match event {
    Event::Added { info, is_initial_scan } => {
        // Skip if initial scan and device already in pool
        if is_initial_scan {
            if let ShouldAddResult::AlreadyExists = pool.should_add_device(&info.id, &backend_name) {
                debug!(device_id = ?info.id, "Skipping initial scan duplicate");
                continue;
            }
        }

        // Pre-filter before opening
        match pool.should_add_device(&info.id, &backend_name) {
            ShouldAddResult::ShouldAdd => {
                // Only open device if it will be added
                if let Ok(device) = backend.open_device(&info.id) {
                    pool.add_device(device, backend_name)?;
                }
            }
            ShouldAddResult::FilteredOut { reason } => {
                debug!(device_id = ?info.id, reason = reason, "Device pre-filtered");
            }
            ShouldAddResult::AlreadyExists => {
                debug!(device_id = ?info.id, "Device already in pool");
            }
            _ => {}
        }
    }
}
```

### 3. Duplicate Device Variant Handling

**Option A: One Device Per Mode** (current behavior)
- Treat each RSPduo mode (DT, MA, MA8, ST) as separate device
- User can choose which mode to use via CLI
- Pool manages all variants independently

**Option B: Deduplicate by Hardware**
- Detect that DT/MA/MA8/ST are same physical hardware
- Only add first variant discovered
- Later variants get `AlreadyExists` result

**Recommendation**: Start with Option A (simpler), add Option B later if needed

### 4. Hot-Plug Event Flow

**Complete flow with all checks**:

There are two paths for discovery events reaching the TUI:

**Path 1: Direct USB Events** (handled by Discovery Service)
```rust
fn handle_discovery_event(
    event: Event,
    pool: &Arc<Pool>,
    backend: &Arc<dyn Backend>,
) -> Result<()> {
    match event {
        Event::Added { info, is_initial_scan } => {
            let backend_name = match &info.id {
                DeviceId::Backend { backend, .. } => backend.clone(),
                DeviceId::Usb { .. } => "USB".to_string(),
            };

            // Step 1: Check if we should add this device (pre-filter)
            match pool.should_add_device(&info.id, &backend_name) {
                ShouldAddResult::ShouldAdd => {
                    // Step 2: Open device (expensive operation)
                    match backend.open_device(&info.id) {
                        Ok(device) => {
                            // Step 3: Add to pool (with filter check)
                            match pool.add_device(device, backend_name) {
                                AddDeviceResult::Added { device_id, tuner_count } => {
                                    info!(
                                        device_id = ?device_id,
                                        tuner_count = tuner_count,
                                        is_hot_plug = !is_initial_scan,
                                        "Device added to pool"
                                    );
                                }
                                other => {
                                    debug!(result = ?other, "Device not added");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(device_id = ?info.id, error = ?e, "Failed to open device");
                        }
                    }
                }
                ShouldAddResult::AlreadyExists => {
                    if is_initial_scan {
                        debug!(device_id = ?info.id, "Skipping duplicate from initial scan");
                    } else {
                        warn!(device_id = ?info.id, "Hot-plug device already in pool");
                    }
                }
                ShouldAddResult::FilteredOut { reason } => {
                    debug!(
                        device_id = ?info.id,
                        reason = reason,
                        "Device pre-filtered (not opened)"
                    );
                }
                ShouldAddResult::ShutdownMode => {
                    debug!("Ignoring device add - pool in shutdown");
                }
                ShouldAddResult::PoolBusy => {
                    debug!("Pool busy - skipping device add");
                }
            }
        }
        Event::Removed { id } => {
            match pool.remove_device(&id) {
                Ok(()) => {
                    info!(device_id = ?id, "Device removed from pool");
                }
                Err(e) => {
                    debug!(device_id = ?id, error = ?e, "Device removal failed or skipped");
                }
            }
        }
    }

    Ok(())
}
```

**Path 2: Backend Enumeration via DeviceEnumerationTask** (handled by task, see Plan 009)
```rust
// In DeviceEnumerationTask::run()
for device_info in discovered_devices {
    let capabilities = Capabilities::for_device(&device_info.id);

    let result = self.pool.add_device_metadata(
        device_info.id.clone(),
        capabilities,
        self.backend.clone(),
    );

    match result {
        AddDeviceResult::Added { device_id, tuner_count } => {
            // Emit discovery event - flows to same TUI handler as Path 1
            let _ = self.discovery_tx.send(discovery::Event::Added(device_info));
        }
        // Other cases don't emit events
        _ => {}
    }
}
```

**Unified Event Handling**: Both paths send events to the same channel, which forwards to the TUI. This ensures:
- TUI device list stays synchronized with pool state
- Same event handling logic for both discovery mechanisms
- Clean separation: pool manages allocation, events manage UI updates

### 5. SingleTuner Mode Behavior

**Question**: When pool is in SingleTuner mode and a tuner is already allocated, what should hot-plug do?

**Option A: Reject Immediately** (current behavior)
- New device gets `FilteredOut` because mode doesn't allow multiple allocations
- Simple, predictable
- User sees device in discovery but it's not available

**Option B: Queue for Later**
- Store device info but don't allocate
- When current tuner is released, queued devices become available
- More complex state management

**Option C: Allow Addition, Reject Allocation**
- Add device to pool's internal inventory
- Filter prevents allocation (not addition)
- User can see all discovered devices via `pool.status()`
- Allocation fails with "SingleTuner mode and 1 tuner already allocated"

**Recommendation**: Option C - separate device inventory from allocation policy

## Implementation Steps

### Step 1: Update Discovery Service

1. Add `is_initial_scan` flag to `Event::Added`
2. Modify `run()` to mark initial enumeration events
3. Once initial scan completes, mark all future events as `is_initial_scan: false`

### Step 2: Implement Pre-Filter

1. Add `should_add_device()` method to Pool
2. Add `ShouldAddResult` enum
3. Add `AlreadyExists` check in pool
4. Add unit tests for pre-filter logic

### Step 3: Update Discovery Handler

1. Check `should_add_device()` before opening device
2. Skip initial scan duplicates
3. Only open devices that will be added
4. Update logging to distinguish hot-plug from initial scan

### Step 4: Handle Removal

1. Test hot-unplug scenarios
2. Verify allocated device removal fails gracefully
3. Ensure removed device can be re-added if plugged back in

### Step 5: Integration Testing

1. Test initial scan with multiple devices
2. Test hot-plug add during scanning
3. Test hot-unplug of unused device
4. Test hot-unplug of allocated device (should fail)
5. Test re-plug of same device

## Success Criteria

✅ Initial scan doesn't add duplicate devices
✅ Hot-plug adds new devices at runtime
✅ Hot-unplug removes unused devices
✅ Hot-unplug of allocated device fails gracefully
✅ Pre-filter prevents opening devices that will be rejected
✅ No filter logic duplication between pool and discovery
✅ SingleTuner mode behavior is clear and consistent
✅ All device variants handled correctly (RSPduo modes, etc.)

## Current Workaround

Until this plan is implemented:
- Discovery events forwarded to TUI only (no pool updates)
- Initial device added to pool at startup (working correctly)
- Pool filter prevents unwanted allocations
- No runtime device add/remove via discovery

## Related Documentation

- Plan 006: Discovery service provides Event::Added / Event::Removed
- Plan 007: Pool infrastructure with AddDeviceResult enum
- Plan 007: Pool filter with backend/driver/mode/tuner constraints
