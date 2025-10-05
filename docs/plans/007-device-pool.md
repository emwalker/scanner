# Plan 007: Device Pool with RAII

**Date**: October 2025
**Status**: Not Started
**Dependencies**: ✅ `005-backend-abstraction.md`, ✅ `006-device-discovery.md`
**Related Plans**: `004-multi-sdr.md` (parent plan)
**Enables**: Plans 009, 010

## Prerequisites Complete

All dependencies are now complete:
- ✅ Plan 005: Backend abstraction (`Backend`, `DeviceTrait`) is implemented
- ✅ Plan 006: Discovery service provides `Event::Added` / `Event::Removed`

This plan is ready to implement.

## Executive Summary

Implement dynamic device inventory using Rust's RAII (Resource Acquisition Is Initialization) pattern to guarantee proper resource management.

**Key innovation**: Devices automatically return to pool when dropped - impossible to leak.

## Problem Statement

Current approach uses raw device handles:
```rust
let device = devices[0].clone();  // ← Easy to leak
// ... use device ...
// Oops, forgot to return it!
```

**Issues**:
- Manual device management (easy to forget to release)
- No dynamic inventory (devices fixed at startup)
- No capability-based allocation
- Resource leaks possible

## Goal

Self-managing device pool with guaranteed cleanup:
```rust
{
    let device = pool.acquire(requirements)?;  // Returns pool::PooledDevice
    // ... use device ...
} // ← Automatically returned to pool when out of scope!
```

## Design

### Core Types

```rust
/// Dynamic inventory of available devices
pub struct Pool {
    /// Devices available for allocation
    available: HashMap<sdr::DeviceId, Entry>,

    /// Devices currently allocated to tasks
    allocated: HashMap<sdr::DeviceId, AllocationInfo>,

    /// Self-reference for RAII pattern
    pool_ref: Arc<Mutex<PoolInner>>,
}

/// Internal state (needed for Arc<Mutex<>> pattern)
struct PoolInner {
    available: HashMap<sdr::DeviceId, Entry>,
    allocated: HashMap<sdr::DeviceId, AllocationInfo>,
}

/// Device entry in pool inventory
struct Entry {
    device: Box<dyn sdr::Device>,
    capabilities: sdr::Capabilities,
    backend_name: String,
    added_at: Instant,
}

/// Allocation tracking
struct AllocationInfo {
    allocated_at: Instant,
    task_id: Option<TaskId>,  // Future use
}

/// Smart pointer that auto-returns device on drop (RAII)
pub struct PooledDevice {
    /// The actual device (Option for Drop impl)
    device: Option<Box<dyn sdr::Device>>,

    /// Pool reference for auto-return
    pool: Arc<Mutex<PoolInner>>,

    /// Device ID for return
    device_id: sdr::DeviceId,
}

impl PooledDevice {
    /// Access to device capabilities
    pub fn capabilities(&self) -> &sdr::Capabilities {
        self.device.as_ref().unwrap().capabilities()
    }

    /// Access to underlying device
    pub fn as_device(&self) -> &dyn sdr::Device {
        self.device.as_ref().unwrap().as_ref()
    }

    /// Mutable access to underlying device
    pub fn as_device_mut(&mut self) -> &mut dyn sdr::Device {
        self.device.as_mut().unwrap().as_mut()
    }
}

impl Drop for PooledDevice {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            let mut pool = self.pool.lock().unwrap();
            pool.return_device(self.device_id.clone(), device);
        }
    }
}
```

### Pool Operations

```rust
impl Pool {
    pub fn new() -> Arc<Self> {
        let inner = PoolInner {
            available: HashMap::new(),
            allocated: HashMap::new(),
        };

        let pool_ref = Arc::new(Mutex::new(inner));

        Arc::new(Self {
            available: HashMap::new(),
            allocated: HashMap::new(),
            pool_ref: pool_ref.clone(),
        })
    }

    /// Add newly discovered device
    pub fn add_device(
        &mut self,
        device: Box<dyn sdr::Device>,
        backend_name: String,
    ) -> Result<()> {
        let device_id = device.device_id().clone();
        let capabilities = device.capabilities().clone();

        debug!(
            device_id = ?device_id,
            model = capabilities.model,
            backend = backend_name,
            "Adding device to pool"
        );

        let entry = Entry {
            device,
            capabilities,
            backend_name,
            added_at: Instant::now(),
        };

        let mut inner = self.pool_ref.lock().unwrap();
        inner.available.insert(device_id, entry);

        Ok(())
    }

    /// Remove device (hot-unplug)
    pub fn remove_device(&mut self, device_id: &sdr::DeviceId) -> Option<Entry> {
        let mut inner = self.pool_ref.lock().unwrap();

        if inner.allocated.contains_key(device_id) {
            // Device is currently in use - can't remove yet
            // Task will handle device failure when it tries to use it
            debug!(device_id = ?device_id, "Cannot remove allocated device");
            return None;
        }

        inner.available.remove(device_id)
    }

    /// Acquire device matching requirements
    pub fn acquire(&self, requirements: &TaskRequirements) -> Result<PooledDevice> {
        let mut inner = self.pool_ref.lock().unwrap();

        // Find best matching device
        let best_match = inner.available
            .iter()
            .filter(|(_, entry)| entry.capabilities.can_handle_task(requirements))
            .min_by_key(|(_, entry)| {
                // Prefer device with smallest freq range that still fits
                // (save wide-range devices for tasks that need them)
                let range_size = entry.capabilities.freq_range_hz.1
                              - entry.capabilities.freq_range_hz.0;

                // Secondary sort: prefer older devices (FIFO)
                (range_size as u64, entry.added_at)
            });

        match best_match {
            Some((device_id, _)) => {
                let device_id = device_id.clone();
                let entry = inner.available.remove(&device_id).unwrap();

                // Mark as allocated
                inner.allocated.insert(device_id.clone(), AllocationInfo {
                    allocated_at: Instant::now(),
                    task_id: None,
                });

                debug!(
                    device_id = ?device_id,
                    model = entry.capabilities.model,
                    "Device acquired from pool"
                );

                Ok(PooledDevice {
                    device: Some(entry.device),
                    pool: Arc::clone(&self.pool_ref),
                    device_id,
                })
            }
            None => Err(ScannerError::NoAvailableDevice(requirements.clone())),
        }
    }

    /// Get pool status (for TUI display)
    pub fn status(&self) -> PoolStatus {
        let inner = self.pool_ref.lock().unwrap();

        PoolStatus {
            available_count: inner.available.len(),
            allocated_count: inner.allocated.len(),
            devices: inner.available.values()
                .map(|entry| DeviceStatus {
                    id: entry.device.device_id().clone(),
                    model: entry.capabilities.model.clone(),
                    backend: entry.backend_name.clone(),
                    state: DeviceState::Available,
                })
                .chain(
                    inner.allocated.keys().map(|id| DeviceStatus {
                        id: id.clone(),
                        model: "allocated".to_string(),
                        backend: "".to_string(),
                        state: DeviceState::Allocated,
                    })
                )
                .collect(),
        }
    }
}

impl PoolInner {
    /// Internal: return device to pool (called by PooledDevice::drop)
    fn return_device(&mut self, device_id: sdr::DeviceId, device: Box<dyn sdr::Device>) {
        debug!(device_id = ?device_id, "Device returned to pool");

        // Remove from allocated
        self.allocated.remove(&device_id);

        // Re-query capabilities (might have changed)
        let capabilities = device.capabilities().clone();

        // Add back to available
        let entry = Entry {
            device,
            capabilities,
            backend_name: "SoapySDR".to_string(),  // TODO: Store backend name
            added_at: Instant::now(),
        };

        self.available.insert(device_id, entry);
    }
}
```

### Capability Matching

```rust
/// Device capabilities for matching to tasks
/// (Note: This is defined in src/sdr/types.rs as sdr::Capabilities)
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub device_id: sdr::DeviceId,
    pub serial_number: String,
    pub model: String,

    // Frequency capabilities
    pub freq_range_hz: (f64, f64),

    // Sample rate capabilities
    pub supported_sample_rates: Vec<f64>,
    pub max_sample_rate: f64,

    // Other capabilities
    pub num_rx_channels: usize,
    pub has_agc: bool,
    pub gain_range: (f64, f64),
}

impl Capabilities {
    /// Check if this device can handle a specific task
    pub fn can_handle_task(&self, task: &TaskRequirements) -> bool {
        // Frequency range check
        if task.frequency_hz < self.freq_range_hz.0
            || task.frequency_hz > self.freq_range_hz.1 {
            return false;
        }

        // Sample rate check
        if task.required_sample_rate > self.max_sample_rate {
            return false;
        }

        // Could add more checks:
        // - Bandwidth requirements
        // - Channel count
        // - Specific features (AGC, etc.)

        true
    }

    /// Query from SoapySDR device
    pub fn from_soapy_device(device: &soapysdr::Device) -> Result<Self> {
        let hardware_info = device.get_hardware_info();

        let serial = hardware_info
            .get("serial")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let model = hardware_info
            .get("label")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let driver = hardware_info
            .get("driver")
            .cloned()
            .unwrap_or_else(|| "soapy".to_string());

        // Query frequency range
        let freq_ranges = device.get_frequency_range(soapysdr::Direction::Rx, 0)?;
        let freq_range_hz = if !freq_ranges.is_empty() {
            let min = freq_ranges.iter().map(|r| r.minimum()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = freq_ranges.iter().map(|r| r.maximum()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            (min, max)
        } else {
            (0.0, 6e9)  // Default wide range
        };

        // Query sample rates
        let sample_rates = device.get_sample_rate_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| r.maximum())
            .collect::<Vec<_>>();

        let max_sample_rate = sample_rates.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .cloned()
            .unwrap_or(10e6);

        // Query other capabilities
        let num_rx_channels = device.get_num_channels(soapysdr::Direction::Rx)?;
        let has_agc = device.has_gain_mode(soapysdr::Direction::Rx, 0)?;

        let gain_range = device.get_gain_range(soapysdr::Direction::Rx, 0)?
            .map(|r| (r.minimum(), r.maximum()))
            .unwrap_or((0.0, 48.0));

        Ok(Self {
            device_id: sdr::DeviceId::Backend {
                backend: driver.clone(),
                serial: serial.clone()
            },
            serial_number: serial,
            model,
            freq_range_hz,
            supported_sample_rates: sample_rates,
            max_sample_rate,
            num_rx_channels,
            has_agc,
            gain_range,
        })
    }
}

/// Task requirements for capability matching
#[derive(Clone, Debug)]
pub struct TaskRequirements {
    pub frequency_hz: f64,
    pub bandwidth_hz: f64,
    pub required_sample_rate: f64,
    pub priority: TaskPriority,
}

#[derive(Clone, Debug)]
pub enum TaskPriority {
    Low,       // Background scanning
    Normal,    // Regular audio
    High,      // P25 control channel
}
```

### Integration with Discovery

```rust
// In main thread or orchestration layer
let pool = pool::Pool::new();
let (event_tx, event_rx) = mpsc::channel();

// Start discovery service
coordinator.spawn_sdr_thread(|cancel| {
    discovery_service.run(event_tx, cancel);
});

// Handle discovery events
coordinator.spawn_sdr_thread(|cancel| {
    while !cancel.is_cancelled() {
        match event_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(discovery::Event::Added(info)) => {
                // Open device via backend
                if let Ok(device) = backend.open_device(&info.id) {
                    // Extract backend name from DeviceId
                    let backend_name = match &info.id {
                        sdr::DeviceId::Backend { backend, .. } => backend.clone(),
                        sdr::DeviceId::Usb { .. } => "USB".to_string(),
                    };
                    pool.add_device(device, backend_name);
                }
            }
            Ok(discovery::Event::Removed(id)) => {
                pool.remove_device(&id);
            }
            Err(_) => {
                // Timeout or channel closed
            }
        }
    }
});
```

## Implementation Steps

### Step 1: Create Module Structure
**Time**: 30 minutes

1. Create `src/pool/mod.rs`
2. Create `src/pool/pool.rs` - Pool implementation
3. Create `src/pool/pooled_device.rs` - PooledDevice RAII wrapper
4. Create `src/pool/types.rs` - Supporting types (TaskRequirements, PoolStatus, etc.)

### Step 2: Implement Capability Matching
**Time**: 1 hour

1. Add capability matching to `sdr::Capabilities` (in Plan 005 types)
2. Implement `can_handle_task()` matching logic
3. Add unit tests for capability matching

```rust
#[test]
fn test_capability_matching() {
    let caps = sdr::Capabilities {
        device_id: sdr::DeviceId::Backend {
            backend: "test".to_string(),
            serial: "001".to_string()
        },
        serial_number: "001".into(),
        model: "Test SDR".into(),
        freq_range_hz: (1e6, 1e9),
        supported_sample_rates: vec![2e6],
        max_sample_rate: 2e6,
        num_rx_channels: 1,
        has_agc: true,
        gain_range: (0.0, 48.0),
    };

    let fm_task = TaskRequirements {
        frequency_hz: 88.9e6,  // FM radio
        bandwidth_hz: 200e3,
        required_sample_rate: 2e6,
        priority: TaskPriority::Normal,
    };

    assert!(caps.can_handle_task(&fm_task));

    let out_of_range = TaskRequirements {
        frequency_hz: 5e9,  // Too high
        bandwidth_hz: 200e3,
        required_sample_rate: 2e6,
        priority: TaskPriority::Normal,
    };

    assert!(!caps.can_handle_task(&out_of_range));
}
```

### Step 3: Implement Pool (Core)
**Time**: 2 hours

1. Implement `Pool::new()`
2. Implement `add_device()`
3. Implement `remove_device()`
4. Implement `acquire()` with capability matching
5. Add pool status query for TUI

### Step 4: Implement RAII PooledDevice
**Time**: 1.5 hours

1. Define `PooledDevice` struct
2. Implement `Deref` and `DerefMut` for ergonomic access
3. Implement `Drop` for auto-return
4. Add tests verifying auto-return

```rust
#[test]
fn test_pooled_device_auto_return() {
    let pool = pool::Pool::new();

    // Add a test device
    let device = create_test_device();
    let device_id = device.device_id().clone();
    pool.add_device(device, "test".into()).unwrap();

    // Verify available
    assert_eq!(pool.status().available_count, 1);

    {
        let _pooled = pool.acquire(&test_requirements()).unwrap();
        // Device is now allocated
        assert_eq!(pool.status().allocated_count, 1);
        assert_eq!(pool.status().available_count, 0);
    } // ← Drop happens here

    // Device should be automatically returned
    assert_eq!(pool.status().available_count, 1);
    assert_eq!(pool.status().allocated_count, 0);
}

#[test]
fn test_pooled_device_explicit_drop() {
    let pool = pool::Pool::new();
    pool.add_device(create_test_device(), "test".into()).unwrap();

    let pooled = pool.acquire(&test_requirements()).unwrap();
    drop(pooled);  // Explicit drop

    // Should return to pool
    assert_eq!(pool.status().available_count, 1);
}
```

### Step 5: Allocation Strategy
**Time**: 1 hour

Implement smart device selection:
1. Filter by capability match
2. Sort by frequency range (prefer narrow-band for specific tasks)
3. Secondary sort by age (FIFO - oldest first)

```rust
fn find_best_device(&self, requirements: &TaskRequirements) -> Option<&sdr::DeviceId> {
    self.available
        .iter()
        .filter(|(_, entry)| entry.capabilities.can_handle_task(requirements))
        .min_by_key(|(_, entry)| {
            let range_size = entry.capabilities.freq_range_hz.1
                          - entry.capabilities.freq_range_hz.0;

            // Prefer narrow-range devices (save wide-range for tasks that need them)
            // Break ties with age (FIFO)
            (range_size as u64, entry.added_at)
        })
        .map(|(id, _)| id)
}
```

### Step 6: Integration Testing
**Time**: 1 hour

```rust
#[test]
fn test_pool_with_discovery() {
    let pool = pool::Pool::new();
    let backend = sdr::Soapy;

    // Simulate discovery events
    let devices = backend.enumerate_devices().unwrap();

    for device_info in devices {
        let device = backend.open_device(&device_info.id).unwrap();
        // Extract backend name from DeviceId
        let backend_name = match &device_info.id {
            sdr::DeviceId::Backend { backend, .. } => backend.clone(),
            sdr::DeviceId::Usb { .. } => "USB".to_string(),
        };
        pool.add_device(device, backend_name);
    }

    // Should have devices in pool
    assert!(pool.status().available_count > 0);

    // Acquire a device
    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2e6,
        priority: TaskPriority::Normal,
    };

    let device = pool.acquire(&requirements).unwrap();
    assert_eq!(pool.status().allocated_count, 1);

    drop(device);
    assert_eq!(pool.status().allocated_count, 0);
}

#[test]
fn test_no_available_device() {
    let pool = pool::Pool::new();

    let requirements = TaskRequirements {
        frequency_hz: 88.9e6,
        bandwidth_hz: 200e3,
        required_sample_rate: 2e6,
        priority: TaskPriority::Normal,
    };

    // Should fail - no devices
    let result = pool.acquire(&requirements);
    assert!(result.is_err());
}

#[test]
fn test_hot_remove_allocated_device() {
    let pool = pool::Pool::new();

    let device = create_test_device();
    let device_id = device.device_id().clone();
    pool.add_device(device, "test".into()).unwrap();

    // Acquire device
    let _pooled = pool.acquire(&test_requirements()).unwrap();

    // Try to remove allocated device
    let removed = pool.remove_device(&device_id);
    assert!(removed.is_none());  // Can't remove allocated device

    // Device is still allocated
    assert_eq!(pool.status().allocated_count, 1);
}
```

## Benefits

### RAII Guarantees
✅ **Impossible to leak** - Devices always returned on drop
✅ **Compiler enforced** - Rust ownership prevents forgetting cleanup
✅ **Exception safe** - Returns device even if panic occurs
✅ **Scoped lifetime** - Device lifetime matches usage scope

### Dynamic Inventory
✅ **Hot-plug support** - Add devices at runtime
✅ **Hot-remove handling** - Graceful device removal
✅ **Capability-aware** - Automatic device matching

### Pool Management
✅ **Smart allocation** - Best-fit device selection
✅ **Status visibility** - Query pool state for TUI
✅ **Thread-safe** - Arc<Mutex<>> for concurrent access

## Usage Patterns

### Basic Acquisition
```rust
let pool = pool::Pool::new();
// ... add devices ...

let requirements = pool::TaskRequirements {
    frequency_hz: 88.9e6,
    bandwidth_hz: 200e3,
    required_sample_rate: 2e6,
    priority: pool::TaskPriority::Normal,
};

let device = pool.acquire(&requirements)?;
// Use device...
// Automatically returned when out of scope
```

### Multiple Devices
```rust
let scan_device = pool.acquire(&scan_requirements)?;
let audio_device = pool.acquire(&audio_requirements)?;

// Both devices in use simultaneously
// Both auto-returned when dropped
```

### With Discovery
```rust
// Discovery adds/removes devices dynamically
for event in event_rx {
    match event {
        discovery::Event::Added(info) => {
            let device = backend.open_device(&info.id)?;
            // Extract backend name from DeviceId
            let backend_name = match &info.id {
                sdr::DeviceId::Backend { backend, .. } => backend.clone(),
                sdr::DeviceId::Usb { .. } => "USB".to_string(),
            };
            pool.add_device(device, backend_name);
        }
        discovery::Event::Removed(id) => {
            pool.remove_device(&id);
        }
    }
}
```

## File Structure

```
src/
  pool/
    mod.rs              # Module exports
    pool.rs            # Pool implementation
    pooled_device.rs   # PooledDevice RAII wrapper
    types.rs          # TaskRequirements, PoolStatus, etc.
```

## Estimated Time

**Total**: 7-8 hours

- Step 1: Module structure (30 min)
- Step 2: Capability matching (1 hr)
- Step 3: Pool core (2 hrs)
- Step 4: RAII PooledDevice (1.5 hrs)
- Step 5: Allocation strategy (1 hr)
- Step 6: Integration testing (1 hr)

## Success Criteria

✅ Pool manages dynamic device inventory
✅ RAII auto-return verified (devices always returned)
✅ Capability matching works (right device for task)
✅ Hot-add/remove handled gracefully
✅ Thread-safe concurrent access
✅ Integration tests with real devices pass

## Next Steps

After completing this plan:
1. **Plan 008**: Subprocess IPC (pool spawns subprocesses for devices)
2. **Plan 009**: Task Abstraction (tasks acquire from pool)
3. **Plan 010**: Multi-SDR Orchestration (ties everything together)
