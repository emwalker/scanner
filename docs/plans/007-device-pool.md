# Plan 007: Tuner Pool with RAII

**Date**: October 2025
**Status**: Not Started
**Dependencies**: ✅ `005-backend-abstraction.md`, ✅ `006-device-discovery.md`
**Related Plans**: `004-multi-sdr.md` (parent plan)
**Enables**: Plans 009, 010

## Prerequisites Complete

All dependencies are now complete:
- ✅ Plan 005: Backend abstraction (`Backend`, `DeviceTrait`) with multi-tuner support
- ✅ Plan 006: Discovery service provides `Event::Added` / `Event::Removed`

This plan is ready to implement.

## Executive Summary

Implement dynamic tuner inventory using Rust's RAII (Resource Acquisition Is Initialization) pattern to guarantee proper resource management.

**Key innovation**: Tuners automatically return to pool when dropped - impossible to leak.

**Multi-tuner support**: Devices with multiple independent receivers (e.g., SDRplay RSPduo with 2 tuners) have all tuners automatically exposed and managed by the pool.

**Controlled rollout**: Pool includes optional filtering to constrain which tuners can be allocated, enabling safe transition from single-tuner to multi-tuner operation.

## Design Validation

This architecture is validated by real-world SDR applications:

**SDRTrunk** (mature Java SDR decoder) uses an identical pool-based approach:
- Manages SDRs as "pooled set of resources"
- Iterates through pool to find capable tuner for each channel
- Supports "Preferred Tuner" selection with automatic fallback
- Handles bandwidth constraints and dynamic allocation

Our design maps directly to SDRTrunk's proven architecture:
- SDRTrunk's "tuner pool iteration" = our capability-based allocation
- SDRTrunk's "preferred tuner" = our PoolFilter
- SDRTrunk's "automatic fallback" = our best-match algorithm

Additionally, our RAII pattern follows Rust community best practices:
- The "return to pool on Drop" pattern is canonical for Rust object pools
- Widely used by crates like `object-pool` and `lockfree-object-pool`
- Arc<Mutex<>> is the standard pattern for thread-safe resource pools

## Relationship to Existing Code: ActiveTuners

The current codebase has an `ActiveTuners` struct (`src/main_thread.rs:23-40`) that serves as a placeholder for this pool:

```rust
pub struct ActiveTuners {
    pub available: Vec<TunerId>,
    pub scanning: Vec<TunerId>,
    pub listening: Vec<TunerId>,
}
```

**Current limitations:**
- Hardcoded to single tuner (initialized with one `selected_tuner_id`)
- Manual state management (explicitly clears/copies vectors when switching modes)
- No RAII guarantees (must remember to update state)
- No capability matching
- No multi-tuner support
- Sends `TuiEvent::ActiveTunersUpdated` to notify UI of state changes

**The Tuner Pool replaces this entirely:**

| ActiveTuners | Tuner Pool |
|--------------|------------|
| `available: Vec<TunerId>` | `Pool.available_tuners` (HashMap with capabilities) |
| `scanning: Vec<TunerId>` | `Pool.allocated_tuners` (with task tracking) |
| `listening: Vec<TunerId>` | `Pool.allocated_tuners` (with task tracking) |
| Manual vector manipulation | Automatic RAII return on drop |
| Single tuner only | Full multi-tuner support |
| `send_active_tuners_update()` | `Pool.status()` for TUI display |

**Migration path:**
1. Implement Tuner Pool with `PoolFilter` (this plan)
2. Replace `MainThread.active_tuners: ActiveTuners` with `pool: Pool`
3. Use `Pool.status()` instead of `send_active_tuners_update()`
4. Remove `ActiveTuners` struct entirely
5. Initially use `PoolFilter::allow_only(selected_tuner_id)` for safe transition
6. Later PRs: relax filter to enable multi-tuner operation

## Problem Statement

Current approach uses raw device handles and ignores multi-tuner devices:
```rust
let device = devices[0].clone();  // ← Easy to leak
// ... use device ...
// Oops, forgot to return it!

// Also: SDRplay RSPduo has 2 tuners, but we only use one!
```

**Issues**:
- Manual device/tuner management (easy to forget to release)
- No dynamic inventory (devices fixed at startup)
- No capability-based allocation
- **Multi-tuner devices underutilized** (RSPduo has 2 tuners, we only use 1)
- **No tuner-level tracking** (can't show which tuner is doing what in UI)
- Resource leaks possible

## Goal

Self-managing tuner pool with guaranteed cleanup and full multi-tuner support:
```rust
{
    let tuner = pool.acquire(requirements)?;  // Returns pool::PooledTuner
    // ... use tuner ...
} // ← Automatically returned to pool when out of scope!

// With an RSPduo, both tuners are available:
let tuner1 = pool.acquire(&scan_requirements)?;   // Uses RSPduo tuner #1
let tuner2 = pool.acquire(&audio_requirements)?;  // Uses RSPduo tuner #2
// Both operations can run simultaneously!
```

## Design

### Core Types

```rust
/// Dynamic inventory of available tuners
pub struct Pool {
    /// Internal state (Arc<Mutex<>> for thread-safe sharing with PooledTuner)
    pool_ref: Arc<Mutex<PoolInner>>,

    /// Filter controlling which tuners can be allocated
    /// Enables safe transition from single-tuner to multi-tuner operation
    filter: Arc<PoolFilter>,
}

/// Controls which tuners are available for allocation
///
/// Used for gradual rollout of multi-tuner support:
/// - Phase 1: allow_only(selected_tuner_id) - single tuner only
/// - Phase 2+: Gradually relax constraints
/// - Final: allow_all() - full multi-tuner support
pub struct PoolFilter {
    allowed_tuners: Option<HashSet<TunerId>>,  // None = allow all
}

impl PoolFilter {
    /// Allow only specific tuners (transition/testing mode)
    pub fn allow_only(tuners: Vec<TunerId>) -> Self {
        Self {
            allowed_tuners: Some(tuners.into_iter().collect()),
        }
    }

    /// Allow all tuners (full multi-tuner mode)
    pub fn allow_all() -> Self {
        Self {
            allowed_tuners: None,
        }
    }

    /// Check if a tuner is allowed for allocation
    fn is_allowed(&self, tuner_id: &TunerId) -> bool {
        match &self.allowed_tuners {
            None => true,  // Allow all
            Some(set) => set.contains(tuner_id),
        }
    }
}

/// Internal state (needed for Arc<Mutex<>> pattern)
struct PoolInner {
    /// Devices (physical hardware)
    devices: HashMap<DeviceId, DeviceEntry>,

    /// Available tuners (ready for allocation)
    available_tuners: HashMap<TunerId, TunerEntry>,

    /// Allocated tuners (in use by tasks)
    allocated_tuners: HashMap<TunerId, AllocationInfo>,
}

/// Device entry (physical SDR hardware)
struct DeviceEntry {
    /// Shared reference to the device
    /// Multiple tuners from the same device share this
    device: Arc<Mutex<Box<dyn Device>>>,

    /// Device-level capabilities
    capabilities: Capabilities,

    /// Backend that provides this device
    backend_name: String,

    /// Number of tuners/channels this device has
    num_tuners: usize,

    /// When device was added to pool
    added_at: Instant,
}

/// Tuner entry (individual RX channel within a device)
struct TunerEntry {
    /// Which device this tuner belongs to
    device_id: DeviceId,

    /// Channel index (0 for first tuner, 1 for second, etc.)
    channel_index: usize,

    /// Tuner-specific capabilities (may differ from device-level)
    capabilities: Capabilities,
}

/// Tuner identifier: composite of device ID + channel index
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TunerId {
    pub device_id: DeviceId,
    pub channel_index: usize,
}

impl TunerId {
    pub fn new(device_id: DeviceId, channel_index: usize) -> Self {
        Self { device_id, channel_index }
    }
}

/// Allocation tracking
struct AllocationInfo {
    allocated_at: Instant,
    task_id: Option<TaskId>,  // Future use
    backend_name: String,
    model: String,
}

/// Smart pointer that auto-returns tuner on drop (RAII)
///
/// # Design: Explicit Methods vs. Deref
///
/// Unlike `object-pool` crate's `Reusable<T>` which implements `Deref`/`DerefMut`,
/// we use explicit methods (`add_source_to_graph()`, `tune()`, `set_gain()`).
///
/// **Why not Deref?**
/// - Our methods automatically pass the correct `channel_index`
/// - Deref would expose raw `Device` trait, bypassing channel logic
/// - Explicit methods make multi-channel handling clear
/// - Prevents accidental misuse (e.g., calling device.tune() with wrong channel)
///
/// **Trade-off:**
/// - Less "transparent" than Deref (can't treat PooledTuner exactly like Device)
/// - More type-safe and prevents channel-index bugs
///
/// # Lock Ordering
///
/// To prevent deadlocks, always acquire locks in this order:
/// 1. Device lock (`self.device`)
/// 2. Pool lock (`self.pool`)
///
/// All methods follow this ordering. The Drop implementation only locks the pool,
/// ensuring safe cleanup even if device lock is held elsewhere.
pub struct PooledTuner {
    /// Tuner identifier
    tuner_id: TunerId,

    /// Shared reference to the underlying device
    device: Arc<Mutex<Box<dyn Device>>>,

    /// Pool reference for auto-return
    pool: Arc<Mutex<PoolInner>>,
}

impl PooledTuner {
    /// Get the tuner ID
    pub fn id(&self) -> &TunerId {
        &self.tuner_id
    }

    /// Get the channel index for this tuner
    pub fn channel_index(&self) -> usize {
        self.tuner_id.channel_index
    }

    /// Add source to rustradio graph for this tuner
    ///
    /// This is a convenience method that automatically uses the correct channel index.
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        let device = self.device.lock().unwrap();
        device.add_source_to_graph(
            graph,
            self.tuner_id.channel_index,
            freq,
            samp_rate,
            gain_db,
        )
    }

    /// Tune this tuner to a new frequency
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn tune(&mut self, freq: f64) -> Result<()> {
        let mut device = self.device.lock().unwrap();
        device.tune(self.tuner_id.channel_index, freq)
    }

    /// Set gain for this tuner
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn set_gain(&mut self, gain: f64) -> Result<()> {
        let mut device = self.device.lock().unwrap();
        device.set_gain(self.tuner_id.channel_index, gain)
    }
}

impl Drop for PooledTuner {
    /// Return tuner to pool automatically when dropped
    ///
    /// # Lock ordering
    /// Acquires pool lock only (safe - device lock already released by caller)
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        pool.return_tuner(self.tuner_id.clone());
    }
}
```

### Pool Operations

```rust
impl Pool {
    /// Create new pool with filter
    pub fn new(filter: PoolFilter) -> Self {
        let inner = PoolInner {
            devices: HashMap::new(),
            available_tuners: HashMap::new(),
            allocated_tuners: HashMap::new(),
        };

        Self {
            pool_ref: Arc::new(Mutex::new(inner)),
            filter: Arc::new(filter),
        }
    }

    /// Create new pool allowing all tuners (convenience method)
    pub fn new_unfiltered() -> Self {
        Self::new(PoolFilter::allow_all())
    }

    /// Add newly discovered device and expose all its tuners
    pub fn add_device(
        &mut self,
        device: Box<dyn Device>,
        backend_name: String,
    ) -> Result<()> {
        let device_id = device.id().clone();
        let capabilities = device.capabilities().clone();
        let num_tuners = capabilities.channels;

        debug!(
            device_id = ?device_id,
            model = capabilities.model,
            backend = backend_name,
            num_tuners = num_tuners,
            "Adding device to pool"
        );

        let device_arc = Arc::new(Mutex::new(device));

        // Store device
        let device_entry = DeviceEntry {
            device: Arc::clone(&device_arc),
            capabilities: capabilities.clone(),
            backend_name: backend_name.clone(),
            num_tuners,
            added_at: Instant::now(),
        };

        let mut inner = self.pool_ref.lock().unwrap();
        inner.devices.insert(device_id.clone(), device_entry);

        // Expose all tuners as available
        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);

            debug!(
                tuner_id = ?tuner_id,
                "Exposing tuner {}/{}", channel_index + 1, num_tuners
            );

            let tuner_entry = TunerEntry {
                device_id: device_id.clone(),
                channel_index,
                capabilities: capabilities.clone(),  // May differ per channel in future
            };

            inner.available_tuners.insert(tuner_id, tuner_entry);
        }

        Ok(())
    }

    /// Remove device (hot-unplug)
    ///
    /// Removes device and all its tuners. Returns None if any tuner is allocated.
    pub fn remove_device(&mut self, device_id: &DeviceId) -> Result<()> {
        let mut inner = self.pool_ref.lock().unwrap();

        // Check if any tuner from this device is allocated
        let has_allocated_tuners = inner.allocated_tuners.keys()
            .any(|tuner_id| &tuner_id.device_id == device_id);

        if has_allocated_tuners {
            debug!(device_id = ?device_id, "Cannot remove device - tuners in use");
            return Err(ScannerError::DeviceInUse(device_id.clone()));
        }

        // Remove all tuners for this device
        let device_entry = inner.devices.get(device_id)
            .ok_or_else(|| ScannerError::DeviceNotFound(device_id.clone()))?;

        let num_tuners = device_entry.num_tuners;

        for channel_index in 0..num_tuners {
            let tuner_id = TunerId::new(device_id.clone(), channel_index);
            inner.available_tuners.remove(&tuner_id);
        }

        // Remove device
        inner.devices.remove(device_id);

        debug!(device_id = ?device_id, "Device and all tuners removed");
        Ok(())
    }

    /// Acquire tuner matching requirements
    ///
    /// Tuners are filtered based on the pool's PoolFilter before capability matching.
    /// This allows gradual rollout of multi-tuner support.
    pub fn acquire(&self, requirements: &TaskRequirements) -> Result<PooledTuner> {
        let mut inner = self.pool_ref.lock().unwrap();

        // Find best matching tuner
        let best_match = inner.available_tuners
            .iter()
            .filter(|(tuner_id, entry)| {
                // First check: is this tuner allowed by the filter?
                self.filter.is_allowed(tuner_id) &&
                // Second check: can it handle the task?
                entry.capabilities.can_handle_task(requirements)
            })
            .min_by_key(|(_, entry)| {
                // Prefer tuner with smallest freq range that still fits
                // (save wide-range tuners for tasks that need them)
                let range_size = entry.capabilities.freq_range_hz.1
                              - entry.capabilities.freq_range_hz.0;

                // Secondary sort: prefer lower channel indices (FIFO within device)
                (range_size as u64, entry.channel_index)
            });

        match best_match {
            Some((tuner_id, _)) => {
                let tuner_id = tuner_id.clone();
                let entry = inner.available_tuners.remove(&tuner_id).unwrap();

                // Get device info for allocation tracking
                let device_entry = inner.devices.get(&entry.device_id).unwrap();

                // Mark as allocated
                inner.allocated_tuners.insert(tuner_id.clone(), AllocationInfo {
                    allocated_at: Instant::now(),
                    task_id: None,
                    backend_name: device_entry.backend_name.clone(),
                    model: device_entry.capabilities.model.clone(),
                });

                debug!(
                    tuner_id = ?tuner_id,
                    model = device_entry.capabilities.model,
                    "Tuner acquired from pool"
                );

                Ok(PooledTuner {
                    tuner_id,
                    device: Arc::clone(&device_entry.device),
                    pool: Arc::clone(&self.pool_ref),
                })
            }
            None => Err(ScannerError::NoAvailableTuner(requirements.clone())),
        }
    }

    /// Try to acquire tuner matching requirements (non-blocking)
    ///
    /// Similar to `acquire()` but returns `None` instead of an error if no tuner is available.
    /// Useful for optional operations or when you want to skip rather than fail.
    ///
    /// # Use cases
    /// - "Scan on second tuner if available, otherwise just listen"
    /// - "Run multiple tasks in parallel if tuners available"
    /// - "Degrade gracefully when tuners are busy"
    ///
    /// Pattern borrowed from `object-pool` crate's `try_pull()` method.
    pub fn try_acquire(&self, requirements: &TaskRequirements) -> Option<PooledTuner> {
        let mut inner = self.pool_ref.lock().unwrap();

        // Find best matching tuner (same logic as acquire)
        let best_match = inner.available_tuners
            .iter()
            .filter(|(tuner_id, entry)| {
                self.filter.is_allowed(tuner_id) &&
                entry.capabilities.can_handle_task(requirements)
            })
            .min_by_key(|(_, entry)| {
                let range_size = entry.capabilities.freq_range_hz.1
                              - entry.capabilities.freq_range_hz.0;
                (range_size as u64, entry.channel_index)
            });

        match best_match {
            Some((tuner_id, _)) => {
                let tuner_id = tuner_id.clone();
                let entry = inner.available_tuners.remove(&tuner_id).unwrap();
                let device_entry = inner.devices.get(&entry.device_id).unwrap();

                inner.allocated_tuners.insert(tuner_id.clone(), AllocationInfo {
                    allocated_at: Instant::now(),
                    task_id: None,
                    backend_name: device_entry.backend_name.clone(),
                    model: device_entry.capabilities.model.clone(),
                });

                debug!(
                    tuner_id = ?tuner_id,
                    model = device_entry.capabilities.model,
                    "Tuner acquired from pool (try_acquire)"
                );

                Some(PooledTuner {
                    tuner_id,
                    device: Arc::clone(&device_entry.device),
                    pool: Arc::clone(&self.pool_ref),
                })
            }
            None => {
                debug!(
                    requirements = ?requirements,
                    "No tuner available (try_acquire returned None)"
                );
                None
            }
        }
    }

    /// Get pool status (for TUI display)
    pub fn status(&self) -> PoolStatus {
        let inner = self.pool_ref.lock().unwrap();

        PoolStatus {
            available_tuner_count: inner.available_tuners.len(),
            allocated_tuner_count: inner.allocated_tuners.len(),
            device_count: inner.devices.len(),
            tuners: inner.available_tuners.iter()
                .map(|(id, entry)| {
                    let device = inner.devices.get(&entry.device_id).unwrap();
                    TunerStatus {
                        id: id.clone(),
                        model: device.capabilities.model.clone(),
                        backend: device.backend_name.clone(),
                        channel_index: entry.channel_index,
                        state: TunerState::Available,
                    }
                })
                .chain(
                    inner.allocated_tuners.iter().map(|(id, info)| TunerStatus {
                        id: id.clone(),
                        model: info.model.clone(),
                        backend: info.backend_name.clone(),
                        channel_index: id.channel_index,
                        state: TunerState::Allocated,
                    })
                )
                .collect(),
        }
    }
}

impl PoolInner {
    /// Internal: return tuner to pool (called by PooledTuner::drop)
    ///
    /// # Tuner State
    ///
    /// Important: This implementation assumes tuners are returned in a usable state.
    /// Tuners are NOT automatically reset when returned to the pool.
    ///
    /// Current behavior: Tuner frequency/gain settings from previous use are preserved.
    /// This is acceptable because:
    /// 1. Each acquire() will reconfigure the tuner for the new task
    /// 2. Device::add_source_to_graph() sets frequency/sample rate/gain
    ///
    /// Future consideration: If devices maintain state that interferes with reuse,
    /// we may need to add a Device::reset() method to clear state on return.
    fn return_tuner(&mut self, tuner_id: TunerId) {
        debug!(tuner_id = ?tuner_id, "Tuner returned to pool");

        // Remove from allocated
        self.allocated_tuners.remove(&tuner_id);

        // Get device info
        if let Some(device_entry) = self.devices.get(&tuner_id.device_id) {
            // Add back to available
            let tuner_entry = TunerEntry {
                device_id: tuner_id.device_id.clone(),
                channel_index: tuner_id.channel_index,
                capabilities: device_entry.capabilities.clone(),
            };

            self.available_tuners.insert(tuner_id, tuner_entry);
        }
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
    pub priority: TaskPriority,  // Reserved for future priority-based scheduling
}

#[derive(Clone, Debug)]
pub enum TaskPriority {
    Low,       // Background scanning
    Normal,    // Regular audio
    High,      // P25 control channel
}

// Note: Priority is not currently used in allocation logic but is included
// for future enhancements like preemption or queue management.
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

                    // Add device to pool - all tuners automatically exposed
                    // E.g., RSPduo will expose 2 tuners, RTL-SDR will expose 1
                    pool.add_device(device, backend_name);
                }
            }
            Ok(discovery::Event::Removed(id)) => {
                // Remove device and all its tuners
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

1. Create `src/pool/mod.rs`
2. Create `src/pool/pool.rs` - Pool implementation
3. Create `src/pool/pooled_device.rs` - PooledDevice RAII wrapper
4. Create `src/pool/types.rs` - Supporting types (TaskRequirements, PoolStatus, etc.)

### Step 2: Implement Capability Matching

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

1. Implement `Pool::new()`
2. Implement `add_device()`
3. Implement `remove_device()`
4. Implement `acquire()` with capability matching
5. Add pool status query for TUI

### Step 4: Implement RAII PooledDevice

1. Define `PooledDevice` struct
2. Implement accessor methods (`as_device()`, `as_device_mut()`)
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

Implement smart device selection:
1. Filter by capability match
2. Sort by frequency range (prefer narrow-band for specific tasks)
3. Secondary sort by age (FIFO - oldest first)

**Note**: The frequency range heuristic (preferring narrow-band devices) may need adjustment based on real-world testing. A wideband SDR might provide better performance for FM reception than a narrow RTL-SDR, even if both can technically handle the frequency.

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

## Migration Strategy: Single-Tuner to Multi-Tuner

The pool is designed for immediate production use while enabling gradual rollout of multi-tuner support.

### Phase 1: Single-Tuner Mode (Initial PR)

**Goal**: Replace `ActiveTuners` with pool, but maintain single-tuner behavior

```rust
// In MainThread::new()
let filter = PoolFilter::allow_only(vec![selected_tuner_id.clone()]);
let pool = Pool::new(filter);

// Pool tracks all devices/tuners internally, but only one is allocatable
// Behavior identical to current ActiveTuners implementation
```

**Benefits of Phase 1:**
- ✅ RAII cleanup (impossible to leak tuner)
- ✅ Capability matching
- ✅ Hot-plug support
- ✅ Clean architecture for future expansion
- ✅ All multi-tuner code is tested (just filtered)

**Limitations:**
- Only one tuner can be allocated at a time (by design)
- Multi-tuner devices underutilized (RSPduo only uses 1 tuner)

### Phase 2: Selective Multi-Tuner (Future PRs)

**Goal**: Gradually enable multi-tuner support with specific constraints

```rust
// Example: Allow all tuners from SDRplay devices
let filter = PoolFilter::allow_only(
    discovered_tuners.iter()
        .filter(|id| id.backend() == "sdrplay")
        .cloned()
        .collect()
);
```

Or device-specific:
```rust
// Allow both tuners from a specific RSPduo
let rspduo_tuner_1 = TunerId::new(rspduo_device_id.clone(), 0);
let rspduo_tuner_2 = TunerId::new(rspduo_device_id.clone(), 1);
let filter = PoolFilter::allow_only(vec![rspduo_tuner_1, rspduo_tuner_2]);
```

### Phase 3: Full Multi-Tuner (Final State)

**Goal**: Remove all constraints, use all available tuners

```rust
// Simple: remove filter parameter
let pool = Pool::new_unfiltered();

// Or explicit:
let pool = Pool::new(PoolFilter::allow_all());
```

### Transition Timeline

| Phase | Filter | Behavior |
|-------|--------|----------|
| 1 (initial) | `allow_only(selected_tuner_id)` | Single tuner only |
| 2a | `allow_only(sdrplay_tuners)` | SDRplay multi-tuner |
| 2b | `allow_only(tested_tuners)` | Specific tested devices |
| 3 (final) | `allow_all()` | All tuners available |

**One-line change** to progress between phases - just update the filter construction.

## Benefits

### RAII Guarantees
✅ **Impossible to leak** - Tuners always returned on drop
✅ **Compiler enforced** - Rust ownership prevents forgetting cleanup
✅ **Exception safe** - Returns tuner even if panic occurs
✅ **Scoped lifetime** - Tuner lifetime matches usage scope
✅ **Canonical Rust pattern** - Follows community best practices for object pools

### Dynamic Inventory
✅ **Hot-plug support** - Add devices at runtime, tuners automatically exposed
✅ **Hot-remove handling** - Graceful device removal (if no tuners in use)
✅ **Capability-aware** - Automatic tuner matching
✅ **Multi-tuner support** - All device tuners available (RSPduo exposes 2, RTL-SDR exposes 1)
✅ **Controlled rollout** - PoolFilter enables safe transition
✅ **Production validated** - SDRTrunk uses identical architecture

### Pool Management
✅ **Smart allocation** - Best-fit tuner selection
✅ **Status visibility** - Query pool state for TUI (individual tuner states)
✅ **Thread-safe** - Arc<Mutex<>> standard pattern for concurrent access
✅ **Tuner-level tracking** - Know which tuner is doing what
✅ **Deadlock prevention** - Documented lock ordering prevents common pitfalls

## Usage Patterns

### Phase 1: Single-Tuner Mode (Initial Implementation)
```rust
// Create pool with filter limiting to selected tuner
let filter = PoolFilter::allow_only(vec![selected_tuner_id]);
let pool = Pool::new(filter);

// ... add devices via discovery ...

let requirements = pool::TaskRequirements {
    frequency_hz: 88.9e6,
    bandwidth_hz: 200e3,
    required_sample_rate: 2e6,
    priority: pool::TaskPriority::Normal,
};

// Only the filtered tuner can be acquired
let tuner = pool.acquire(&requirements)?;

// Use tuner to create rustradio graph
let mut graph = Graph::new();
let stream = tuner.add_source_to_graph(&mut graph, 88.9e6, 2.4e6, 20.0)?;
// ... build rest of graph ...

// Automatically returned when out of scope
```

### Phase 3: Full Multi-Tuner Mode (Future)
```rust
// Create pool with no filter
let pool = Pool::new_unfiltered();

// Now all discovered tuners can be allocated
let tuner = pool.acquire(&requirements)?;
```

### Multiple Tuners (Including from Same Device!)
```rust
let scan_tuner = pool.acquire(&scan_requirements)?;
let audio_tuner = pool.acquire(&audio_requirements)?;

// If you have an RSPduo, these could be tuner #1 and #2 from same device!
// Both tuners in use simultaneously
// Both auto-returned when dropped
```

### Optional Parallel Operation with try_acquire()
```rust
// Always scan
let scan_tuner = pool.acquire(&scan_requirements)?;

// Listen on second tuner IF available (graceful degradation)
if let Some(audio_tuner) = pool.try_acquire(&audio_requirements) {
    // Both operations running in parallel
    spawn_audio_task(audio_tuner);
} else {
    // Only one tuner available - that's okay, just scan
    debug!("No tuner available for audio, scan-only mode");
}

// Pattern borrowed from object-pool crate's try_pull()
```

### With Discovery
```rust
// Discovery adds/removes devices dynamically
// Pool automatically exposes all tuners from each device
for event in event_rx {
    match event {
        discovery::Event::Added(info) => {
            let device = backend.open_device(&info.id)?;
            let backend_name = match &info.id {
                sdr::DeviceId::Backend { backend, .. } => backend.clone(),
                sdr::DeviceId::Usb { .. } => "USB".to_string(),
            };

            // Adding device automatically exposes all its tuners
            // RSPduo adds 2 tuners, RTL-SDR adds 1
            pool.add_device(device, backend_name);
        }
        discovery::Event::Removed(id) => {
            // Removes device and all its tuners (if not in use)
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
    pool.rs             # Pool implementation (devices + tuners)
    pooled_tuner.rs     # PooledTuner RAII wrapper
    types.rs            # TunerId, TaskRequirements, PoolStatus, TunerStatus, etc.
```

## Implementation Notes

### Relationship to Existing Object Pool Crates

The Rust ecosystem has several object pooling crates:
- **object-pool**: Reusable buffers with automatic return on drop
- **lockfree-object-pool**: Lock-free variant for high-performance scenarios
- **derivable-object-pool**: Macro-based pool generation

**Why custom implementation?**

While these crates provide solid object pooling primitives, our use case requires SDR-specific functionality:
1. **Two-level hierarchy**: Devices → Tuners (not just generic objects)
2. **Capability matching**: Allocate based on frequency range, sample rate, etc.
3. **Multi-channel devices**: RSPduo has 2 tuners sharing one device
4. **Device constraints**: Future support for frequency separation limits
5. **Hot-plug integration**: Dynamic add/remove tied to discovery service

**Borrowing best practices:**

We adopt proven patterns from these crates:
- ✅ RAII wrapper that returns to pool on Drop
- ✅ Arc<Mutex<>> for thread-safe sharing
- ✅ Separate wrapper type for transparent usage
- ⚠️ Object state not automatically reset (documented)

**API Comparison:**

| Pattern | object-pool | lockfree-object-pool | Our Pool | Decision |
|---------|-------------|---------------------|----------|----------|
| RAII return on Drop | ✅ `Reusable<T>` | ✅ `Reusable<T>` | ✅ `PooledTuner` | **Adopted** |
| Deref/DerefMut | ✅ Transparent access | ✅ Transparent access | ❌ Explicit methods | **Not used** - channel logic |
| Non-blocking acquire | ✅ `try_pull()` | ✅ `try_pull()` | ✅ `try_acquire()` | **Adopted** |
| Manual detach/attach | ✅ Yes | ❌ No | ❌ No | **Not needed** - RAII sufficient |
| Fixed capacity | ✅ Yes | ✅ Yes | ❌ Dynamic | **Not applicable** - hardware-driven |
| Reset closure | ❌ No | ✅ Yes | ❌ No | **Future consideration** |
| Thread safety | ✅ Arc wrapping | ✅ Built-in | ✅ Arc<Mutex<>> | **Adopted** |
| Capability matching | ❌ No | ❌ No | ✅ Yes | **SDR-specific** |

**Key Decisions:**

1. **No Deref**: We use explicit methods to encapsulate channel index logic, preventing bugs
2. **Added try_acquire()**: Borrowed from `object-pool` for graceful degradation
3. **No detach/attach**: RAII handles all use cases; manual control would complicate lifecycle
4. **Dynamic capacity**: Pool size determined by discovered hardware, not configuration

## Success Criteria

✅ Pool manages dynamic tuner inventory
✅ RAII auto-return verified (tuners always returned)
✅ Multi-tuner devices fully supported (RSPduo exposes 2 tuners)
✅ Capability matching works (right tuner for task)
✅ Hot-add/remove handled gracefully
✅ Thread-safe concurrent access
✅ Integration tests with single-tuner and multi-tuner devices pass
✅ TUI can display individual tuner states

## Next Steps

After completing this plan:
1. **Plan 008**: Subprocess IPC (pool spawns subprocesses for devices)
2. **Plan 009**: Task Abstraction (tasks acquire from pool)
3. **Plan 010**: Multi-SDR Orchestration (ties everything together)
