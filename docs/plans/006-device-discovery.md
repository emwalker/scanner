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
    pub fn new(backends: Vec<Box<dyn sdr::Backend>>) -> Self {
        Self {
            backends,
            known_devices: HashMap::new(),
        }
    }
}

impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        let context = match Context::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                debug!(error = %e, "udev unavailable, falling back to polling");
                let mut polling = Polling::new(
                    std::mem::take(&mut self.backends),
                    Duration::from_secs(3)
                );
                return polling.run(event_tx, cancel);
            }
        };

        let mut monitor = match MonitorBuilder::new(&context)
            .and_then(|m| m.match_subsystem("usb"))
            .and_then(|m| m.listen())
        {
            Ok(m) => m,
            Err(e) => {
                debug!(error = %e, "failed to create udev monitor, falling back to polling");
                let mut polling = Polling::new(
                    std::mem::take(&mut self.backends),
                    Duration::from_secs(3)
                );
                return polling.run(event_tx, cancel);
            }
        };

        // Initial enumeration to establish baseline
        if self.rescan_devices(&event_tx).is_err() {
            return;
        }

        // Event loop
        loop {
            if cancel.is_cancelled() {
                break;
            }

            // Check for udev events (blocking with timeout via socket)
            if let Some(event) = monitor.iter().next() {
                if cancel.is_cancelled() {
                    break;
                }

                match event.event_type() {
                    EventType::Add | EventType::Remove => {
                        debug!(event_type = ?event.event_type(), "USB event detected");
                        if self.rescan_devices(&event_tx).is_err() {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Udev {
    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
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
                debug!(device_id = ?id, "new device detected");
                event_tx.send(Event::Added(device.clone())).map_err(|_| ())?;
            }
        }

        // Detect removed devices (in known but not in current)
        for id in self.known_devices.keys() {
            if !current_devices.contains_key(id) {
                debug!(device_id = ?id, "device removed");
                event_tx.send(Event::Removed(id.clone())).map_err(|_| ())?;
            }
        }

        // Update known devices
        self.known_devices = current_devices;
        Ok(())
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
        if self.rescan_devices(&event_tx).is_err() {
            return;
        }

        // Polling loop
        loop {
            if cancel.is_cancelled() {
                break;
            }

            std::thread::sleep(self.poll_interval);

            if cancel.is_cancelled() {
                break;
            }

            if self.rescan_devices(&event_tx).is_err() {
                break;
            }
        }
    }
}

impl Polling {
    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
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
                event_tx.send(Event::Added(device.clone())).map_err(|_| ())?;
            }
        }

        for id in self.known_devices.keys() {
            if !current_devices.contains_key(id) {
                event_tx.send(Event::Removed(id.clone())).map_err(|_| ())?;
            }
        }

        self.known_devices = current_devices;
        Ok(())
    }
}
```

### Platform Selection

```rust
pub enum DiscoveryMode {
    Auto,
    ForcePolling(Duration),
    #[cfg(target_os = "linux")]
    ForceUdev,
}

/// Create appropriate discovery service for current platform
pub fn create(backends: Vec<Box<dyn sdr::Backend>>, mode: DiscoveryMode) -> Box<dyn Service> {
    match mode {
        DiscoveryMode::ForcePolling(interval) => {
            Box::new(Polling::new(backends, interval))
        }
        #[cfg(target_os = "linux")]
        DiscoveryMode::ForceUdev => {
            Box::new(Udev::new(backends))
        }
        DiscoveryMode::Auto => {
            #[cfg(target_os = "linux")]
            {
                Box::new(Udev::new(backends))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Box::new(Polling::new(backends, Duration::from_secs(3)))
            }
        }
    }
}
```

## Implementation Steps

### Step 1: Add Dependencies

```toml
# Cargo.toml
[target.'cfg(target_os = "linux")'.dependencies]
libudev = "0.3"
```

### Step 2: Create Module Structure

1. Create `src/discovery/mod.rs`
2. Create `src/discovery/service.rs` - `Service` trait and `Event` enum
3. Create `src/discovery/udev.rs` - Linux implementation
4. Create `src/discovery/polling.rs` - Fallback implementation

### Step 3: Implement Polling Discovery (Simplest First)

1. Implement `Polling` service
2. Add configurable poll interval (default 3 seconds)
3. Test on current platform
4. Verify device detection works

### Step 4: Implement udev Discovery (Linux Only)

1. Implement `Udev` service with fallback to polling on error
2. Handle udev context creation errors gracefully
3. Test with real device plug/unplug
4. Verify instant detection (no polling delay)

### Step 5: Extract Common Logic

Both implementations share device diffing logic. Extract to common helper to avoid duplication:
```rust
pub(crate) fn detect_changes<'a>(
    known: &'a HashMap<sdr::DeviceId, sdr::DeviceInfo>,
    current: &'a HashMap<sdr::DeviceId, sdr::DeviceInfo>,
) -> (
    impl Iterator<Item = &'a sdr::DeviceInfo>,
    impl Iterator<Item = &'a sdr::DeviceId>,
) {
    let added = current.iter()
        .filter(move |(id, _)| !known.contains_key(id))
        .map(|(_, device)| device);

    let removed = known.keys()
        .filter(move |id| !current.contains_key(id));

    (added, removed)
}
```

### Step 6: Integration with ShutdownCoordinator

```rust
// In bin/scanner.rs
use crate::discovery;
use crate::sdr;

let backends: Vec<Box<dyn sdr::Backend>> = vec![
    Box::new(sdr::Soapy),
];

let mut service = discovery::create(backends, discovery::DiscoveryMode::Auto);
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

    // Collect events with timeout
    let mut events = Vec::new();
    let timeout = Duration::from_millis(300);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = rx.recv_timeout(Duration::from_millis(50)) {
            events.push(event);
        }
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
    let mut service = Udev::new(backends);
    let (tx, rx) = mpsc::channel();
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    std::thread::spawn(move || {
        service.run(tx, cancel_clone);
    });

    // Collect events with timeout
    let mut events = Vec::new();
    let timeout = Duration::from_millis(300);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = rx.recv_timeout(Duration::from_millis(50)) {
            events.push(event);
        }
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
    let mut service = discovery::create(backends, discovery::DiscoveryMode::Auto);
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
                println!("Device ADDED: {} ({})", info.model, info.serial);
            }
            discovery::Event::Removed(id) => {
                println!("Device REMOVED: {:?}", id);
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
let mut service = discovery::create(backends, discovery::DiscoveryMode::Auto);
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

let service = discovery::create(backends, discovery::DiscoveryMode::Auto);
// Will detect devices from ALL backends
```

### Force Polling (Testing)
```rust
// Force polling mode even on Linux (useful for testing)
let service = discovery::create(
    backends,
    discovery::DiscoveryMode::ForcePolling(Duration::from_secs(2))
);
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

Discovery services handle errors gracefully:

1. **Event send failures**: When the receiver is dropped (shutdown), send fails and the loop exits
2. **udev initialization failures**: Falls back to polling automatically
3. **Backend enumeration errors**: Logged but don't stop discovery (allows partial device lists)
4. **Cancellation**: Checked after blocking operations to ensure timely shutdown

```rust
// Example: rescan_devices returns Result to signal shutdown
fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
    // ...
    event_tx.send(Event::Added(device.clone())).map_err(|_| ())?;
    // Send failure means receiver dropped = time to shut down
    Ok(())
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
