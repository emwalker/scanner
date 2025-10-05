# Plan 006: Dynamic Device Discovery

**Date**: October 2025
**Status**: Not Started
**Dependencies**: `005-backend-abstraction.md`
**Related Plans**: `004-multi-sdr.md` (parent plan)
**Enables**: Plans 007, 010

## Executive Summary

Enable hot-plug support: automatically detect SDR devices appearing/disappearing at runtime.

**Platform-optimized approach**:
- **Linux**: Event-driven via `libudev` (zero overhead, instant detection)
- **Other OS**: Polling via backend enumeration (2-5 second interval)

This enables the "plug in RTL-SDR → automatically discovered and used" experience.

## Problem Statement

Current behavior:
```rust
// In bin/scanner.rs
let devices = enumerate_sdr_devices()?;  // ← Runs ONCE at startup

// In MainThread
let device = devices[0].clone();  // ← Fixed device for entire session
```

**Issues**:
- Devices enumerated once at startup
- Can't detect newly plugged devices
- Can't handle device removal gracefully
- Need to restart scanner to use new devices

## Goal

Continuous device monitoring that sends events when devices appear/disappear.

```rust
// Discovery service runs in background
let (event_tx, event_rx) = mpsc::channel();
coordinator.spawn_sdr_thread(|cancel| {
    discovery_service.run(event_tx, cancel)
});

// Elsewhere: receive events
while let Ok(event) = event_rx.recv() {
    match event {
        DeviceEvent::Added(info) => pool.add_device(info),
        DeviceEvent::Removed(id) => pool.remove_device(id),
    }
}
```

## Design

### Core Types

```rust
/// Device discovery abstraction (platform-specific implementations)
pub trait Service: Send {
    /// Run discovery loop until shutdown
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken);
}

/// Events emitted by discovery service
#[derive(Debug, Clone)]
pub enum Event {
    /// New device detected
    Added(sdr::DeviceInfo),

    /// Existing device removed
    Removed(sdr::DeviceId),
}
```

### Linux Implementation (Event-Driven)

```rust
use libudev::{Context, MonitorBuilder, EventType};

/// Linux-specific discovery via udev events
pub struct Udev {
    backends: Vec<Box<dyn sdr::Backend>>,
    known_devices: HashMap<sdr::DeviceId, sdr::DeviceInfo>,
}

impl Udev {
    pub fn new(backends: Vec<Box<dyn sdr::Backend>>) -> Result<Self> {
        Ok(Self {
            backends,
            known_devices: HashMap::new(),
        })
    }
}

impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        let context = Context::new().expect("Failed to create udev context");

        let mut monitor = MonitorBuilder::new(&context)
            .expect("Failed to create udev monitor")
            .match_subsystem("usb")
            .expect("Failed to match USB subsystem")
            .listen()
            .expect("Failed to start udev monitoring");

        // Initial enumeration to establish baseline
        self.rescan_devices(&event_tx);

        // Event loop
        loop {
            if cancel.is_cancelled() {
                break;
            }

            // Check for udev events (non-blocking poll with timeout)
            if let Some(event) = monitor.iter().next() {
                match event.event_type() {
                    EventType::Add | EventType::Remove => {
                        // USB event detected, rescan devices
                        debug!("USB event: {:?}", event.event_type());
                        self.rescan_devices(&event_tx);
                    }
                    _ => {}
                }
            }

            // Short sleep to avoid busy-wait
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Udev {
    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) {
        // Query all backends for current devices
        let mut current_devices = HashMap::new();

        for backend in &self.backends {
            if let Ok(devices) = backend.enumerate_devices() {
                for device in devices {
                    current_devices.insert(device.id.clone(), device);
                }
            }
        }

        // Detect new devices (in current but not in known)
        for (id, device) in &current_devices {
            if !self.known_devices.contains_key(id) {
                debug!("New device detected: {:?}", device);
                let _ = event_tx.send(Event::Added(device.clone()));
            }
        }

        // Detect removed devices (in known but not in current)
        for id in self.known_devices.keys() {
            if !current_devices.contains_key(id) {
                debug!("Device removed: {:?}", id);
                let _ = event_tx.send(Event::Removed(id.clone()));
            }
        }

        // Update known devices
        self.known_devices = current_devices;
    }
}
```

### Polling Implementation (Fallback)

```rust
/// Polling-based discovery (macOS, Windows, BSD)
pub struct Polling {
    backends: Vec<Box<dyn sdr::Backend>>,
    known_devices: HashMap<sdr::DeviceId, sdr::DeviceInfo>,
    poll_interval: Duration,
}

impl Polling {
    pub fn new(backends: Vec<Box<dyn sdr::Backend>>, poll_interval: Duration) -> Self {
        Self {
            backends,
            known_devices: HashMap::new(),
            poll_interval,
        }
    }
}

impl Service for Polling {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        // Initial scan
        self.rescan_devices(&event_tx);

        // Polling loop
        loop {
            if cancel.is_cancelled() {
                break;
            }

            std::thread::sleep(self.poll_interval);
            self.rescan_devices(&event_tx);
        }
    }
}

impl Polling {
    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) {
        // Same logic as Udev::rescan_devices()
        // (extract to shared helper)
        let mut current_devices = HashMap::new();

        for backend in &self.backends {
            if let Ok(devices) = backend.enumerate_devices() {
                for device in devices {
                    current_devices.insert(device.id.clone(), device);
                }
            }
        }

        // Detect changes
        for (id, device) in &current_devices {
            if !self.known_devices.contains_key(id) {
                let _ = event_tx.send(Event::Added(device.clone()));
            }
        }

        for id in self.known_devices.keys() {
            if !current_devices.contains_key(id) {
                let _ = event_tx.send(Event::Removed(id.clone()));
            }
        }

        self.known_devices = current_devices;
    }
}
```

### Platform Selection

```rust
/// Create appropriate discovery service for current platform
pub fn create(backends: Vec<Box<dyn sdr::Backend>>) -> Box<dyn Service> {
    #[cfg(target_os = "linux")]
    {
        Box::new(Udev::new(backends).unwrap())
    }

    #[cfg(not(target_os = "linux"))]
    {
        let interval = Duration::from_secs(3);  // Configurable
        Box::new(Polling::new(backends, interval))
    }
}
```

## Implementation Steps

### Step 1: Add Dependencies
**Time**: 15 minutes

```toml
# Cargo.toml
[target.'cfg(target_os = "linux")'.dependencies]
libudev = "0.3"
```

### Step 2: Create Module Structure
**Time**: 30 minutes

1. Create `src/discovery/mod.rs`
2. Create `src/discovery/service.rs` - `Service` trait and `Event` enum
3. Create `src/discovery/udev.rs` - Linux implementation
4. Create `src/discovery/polling.rs` - Fallback implementation

### Step 3: Implement Polling Discovery (Simplest First)
**Time**: 1 hour

1. Implement `Polling` service
2. Add configurable poll interval (default 3 seconds)
3. Test on current platform
4. Verify device detection works

### Step 4: Implement udev Discovery (Linux Only)
**Time**: 2 hours

1. Implement `Udev` service
2. Handle udev context creation errors gracefully
3. Test with real device plug/unplug
4. Verify instant detection (no polling delay)

### Step 5: Extract Common Logic
**Time**: 30 minutes

Both implementations share device diffing logic:
```rust
fn detect_changes(
    known: &HashMap<sdr::DeviceId, sdr::DeviceInfo>,
    current: &HashMap<sdr::DeviceId, sdr::DeviceInfo>,
) -> (Vec<sdr::DeviceInfo>, Vec<sdr::DeviceId>) {
    let added: Vec<_> = current.values()
        .filter(|d| !known.contains_key(&d.id))
        .cloned()
        .collect();

    let removed: Vec<_> = known.keys()
        .filter(|id| !current.contains_key(id))
        .cloned()
        .collect();

    (added, removed)
}
```

### Step 6: Integration with ShutdownCoordinator
**Time**: 30 minutes

```rust
// In bin/scanner.rs
use crate::discovery;
use crate::sdr;

let backends: Vec<Box<dyn sdr::Backend>> = vec![
    Box::new(sdr::Soapy),
];

let mut service = discovery::create(backends);
let (event_tx, event_rx) = mpsc::channel();

coordinator.spawn_sdr_thread(move |cancel| {
    service.run(event_tx, cancel);
});

// Main thread receives events
std::thread::spawn(move || {
    while let Ok(event) = event_rx.recv() {
        match event {
            discovery::Event::Added(info) => {
                println!("Device added: {} ({})", info.model, info.serial);
                // Will connect to pool in Plan 007
            }
            discovery::Event::Removed(id) => {
                println!("Device removed: {:?}", id);
            }
        }
    }
});
```

### Step 7: Testing
**Time**: 1 hour

```rust
#[test]
fn test_polling_discovery() {
    use crate::discovery::{Polling, Service};
    use crate::sdr;

    let backends: Vec<Box<dyn sdr::Backend>> = vec![Box::new(sdr::Soapy)];
    let mut service = Polling::new(backends, Duration::from_millis(100));
    let (tx, rx) = mpsc::channel();
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        service.run(tx, cancel_clone);
    });

    // Wait for initial scan
    std::thread::sleep(Duration::from_millis(200));

    // Should receive Added events for existing devices
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    assert!(!events.is_empty(), "Should detect existing devices");

    cancel.cancel();
}

#[cfg(target_os = "linux")]
#[test]
fn test_udev_discovery() {
    use crate::discovery::{Udev, Service};
    use crate::sdr;

    let backends: Vec<Box<dyn sdr::Backend>> = vec![Box::new(sdr::Soapy)];
    let mut service = Udev::new(backends).unwrap();
    let (tx, rx) = mpsc::channel();
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        service.run(tx, cancel_clone);
    });

    // Wait for initial scan
    std::thread::sleep(Duration::from_millis(200));

    // Collect initial events
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    assert!(!events.is_empty(), "Should detect existing devices via udev");

    cancel.cancel();
}

// Manual test: plug/unplug device
#[test]
#[ignore]  // Run manually: cargo test test_manual_hotplug -- --ignored
fn test_manual_hotplug() {
    use crate::discovery;
    use crate::sdr;

    let backends: Vec<Box<dyn sdr::Backend>> = vec![Box::new(sdr::Soapy)];
    let mut service = discovery::create(backends);
    let (tx, rx) = mpsc::channel();
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        service.run(tx, cancel_clone);
    });

    println!("Watching for device changes...");
    println!("Plug/unplug a device to test detection.");
    println!("Press Ctrl+C to stop.");

    while let Ok(event) = rx.recv() {
        match event {
            discovery::Event::Added(info) => {
                println!("✅ Device ADDED: {} ({})", info.model, info.serial);
            }
            discovery::Event::Removed(id) => {
                println!("❌ Device REMOVED: {:?}", id);
            }
        }
    }
}
```

## Usage Pattern

### Basic Usage
```rust
use crate::discovery;
use crate::sdr;

let backends = vec![Box::new(sdr::Soapy)];
let mut service = discovery::create(backends);
let (event_tx, event_rx) = mpsc::channel();

coordinator.spawn_sdr_thread(|cancel| {
    service.run(event_tx, cancel);
});

// Handle events
for event in event_rx {
    match event {
        discovery::Event::Added(info) => { /* ... */ }
        discovery::Event::Removed(id) => { /* ... */ }
    }
}
```

### With Multiple Backends (Future)
```rust
let backends: Vec<Box<dyn sdr::Backend>> = vec![
    Box::new(sdr::Soapy),
    Box::new(sdr::Seify),  // Future
];

let service = discovery::create(backends);
// Will detect devices from ALL backends
```

## Benefits

### Linux
✅ **Instant detection** - udev events trigger immediately on plug/unplug
✅ **Zero CPU overhead** - Event-driven, no polling
✅ **Standard practice** - Uses native Linux hotplug mechanism

### Other Platforms
✅ **Works everywhere** - Polling fallback for macOS, Windows, BSD
✅ **Configurable interval** - Balance between responsiveness and overhead
✅ **Simple implementation** - Just periodic enumeration

### Cross-Platform
✅ **Backend-agnostic** - Works with any `sdr::Backend` implementation
✅ **Automatic platform selection** - Compile-time detection
✅ **Consistent API** - Same `discovery::Event` type on all platforms

## Configuration

```rust
// Configurable poll interval for non-Linux platforms
pub struct Config {
    #[cfg(not(target_os = "linux"))]
    pub poll_interval: Duration,  // Default: 3 seconds
}

impl Default for Config {
    fn default() -> Self {
        Self {
            #[cfg(not(target_os = "linux"))]
            poll_interval: Duration::from_secs(3),
        }
    }
}
```

## Performance Characteristics

### Linux (udev)
- Detection latency: <100ms (typically instant)
- CPU overhead: ~0% (event-driven)
- Memory overhead: ~50KB (udev monitor)

### Polling (other OS)
- Detection latency: 0 - `poll_interval` (default 3 seconds)
- CPU overhead: ~0.1% (quick enumeration every 3s)
- Memory overhead: ~10KB (device list storage)

## Error Handling

```rust
impl Udev {
    pub fn new(backends: Vec<Box<dyn sdr::Backend>>) -> Result<Self> {
        // Verify udev is available
        let context = Context::new()
            .map_err(|e| ScannerError::Custom(
                format!("udev not available: {}", e)
            ))?;

        // Fall back to polling if udev fails
        Ok(Self { backends, known_devices: HashMap::new() })
    }
}

impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        // If udev initialization fails, fall back to polling
        let context = match Context::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                debug!("udev unavailable ({}), falling back to polling", e);
                let mut polling = Polling::new(
                    std::mem::take(&mut self.backends),
                    Duration::from_secs(3)
                );
                return polling.run(event_tx, cancel);
            }
        };

        // ... rest of udev implementation
    }
}
```

## File Structure

```
src/
  discovery/
    mod.rs              # Module exports, platform selection
    service.rs          # Service trait and Event enum
    udev.rs            # Linux udev implementation
    polling.rs         # Polling fallback
    common.rs          # Shared diffing logic
```

## Estimated Time

**Total**: 5-6 hours

- Step 1: Dependencies (15 min)
- Step 2: Module structure (30 min)
- Step 3: Polling implementation (1 hr)
- Step 4: udev implementation (2 hrs)
- Step 5: Extract common logic (30 min)
- Step 6: ShutdownCoordinator integration (30 min)
- Step 7: Testing (1 hr)

## Success Criteria

✅ `Polling` service works on all platforms
✅ `Udev` service works on Linux (instant detection)
✅ Graceful fallback if udev unavailable
✅ Integrates with ShutdownCoordinator
✅ Manual hotplug test passes (plug/unplug detected)
✅ Works with multiple backends

## Next Steps

After completing this plan:
1. **Plan 007**: Device Pool (receives `discovery::Event`s, manages inventory)
2. **Plan 010**: Multi-SDR Orchestration (uses discovery + pool together)
