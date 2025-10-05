# Plan 006: Dynamic Device Discovery

**Date**: October 2025
**Status**: ✅ Completed
**Dependencies**: `005-backend-abstraction.md`
**Related Plans**: `004-multi-sdr.md` (parent plan)
**Enables**: Plans 007, 010

## Implementation Notes

**Completed Features**:
- ✅ Multi-source enumeration (Backend + USB VID/PID database)
- ✅ Platform-optimized detection (udev on Linux, polling elsewhere)
- ✅ Event-driven architecture with `Event::Added` / `Event::Removed`
- ✅ Deduplication across sources with priority handling (Backend > UsbInspection)
- ✅ Shutdown-responsive with cancellation tokens
- ✅ Test isolation with `create_for_testing()` to avoid real hardware detection
- ✅ TUI integration with real-time device display
- ✅ RtAudio stderr suppression to prevent TUI corruption
- ✅ RSPduo multi-mode support (ST, DT, MA, MA8)

**Implementation Details**:

### 1. Test Isolation (`src/discovery/mod.rs:65`)
Created `create_for_testing()` function that bypasses USB enumeration during integration tests:
```rust
pub fn create_for_testing(backends, mode) -> Box<dyn Service>
```
This ensures tests only enumerate from provided Mock backend, preventing detection of real hardware on developer machines. Critical for CI/CD environments.

### 2. TUI Integration (`src/terminal/mod.rs`, `bin/scanner.rs`)
- Created `TuiEvent` enum combining `ProgressEvent` and discovery events
- Added event forwarding threads to bridge discovery → TUI
- Updated `Model` to track discovered devices in `Vec<DeviceInfo>`
- Modified tuner renderer to display real devices instead of hardcoded mock data
- Shows "No SDR devices detected" when list is empty

### 3. RtAudio Stderr Suppression (`src/sdr/soapy.rs:11-38`)
Fixed "RtApi::getDeviceInfo: deviceId argument not found" errors corrupting TUI:
```rust
fn suppress_stderr<F, R>(f: F) -> R {
    // Temporarily redirects stderr to /dev/null using POSIX fd manipulation
    // Wraps soapysdr::enumerate() to suppress RtAudio library spam
}
```
Also filters out audio devices: `if driver == "audio" { return None; }`

### 4. RSPduo Multi-Mode Support (`src/sdr/soapy.rs:66-70`)
Fixed issue where RSPduo only showed 1 device instead of 4 modes:
```rust
let unique_serial = if mode.is_empty() {
    serial.clone()
} else {
    format!("{}:{}", serial, mode)  // ST, DT, MA, MA8
};
```
Creates unique DeviceIds for each operational mode, preventing TUI deduplication.

### 5. Deduplication Logic (`src/discovery/common.rs`)
Multi-source enumeration with priority ordering:
- Backend enumeration (SoapySDR) takes priority
- USB VID/PID inspection provides fallback
- Deterministic ordering ensures consistent device list
- Changes detected via set difference (added/removed)

**Key Decisions**:
- Used POSIX file descriptor manipulation (`libc::dup2`) for stderr suppression instead of SoapySDR log levels (which don't affect RtAudio)
- Implemented mode-in-serial for RSPduo: `sdrplay:2301034E34:ST` instead of just `sdrplay:2301034E34`
- Added regression test `test_rspduo_multi_mode_enumeration()` to prevent future issues
- Event forwarding threads bridge discovery service to TUI without tight coupling

**Files Modified**:
- `src/discovery/mod.rs` - Added `create_for_testing()`
- `src/sdr/soapy.rs` - Stderr suppression, audio filtering, RSPduo mode handling
- `src/terminal/mod.rs` - `TuiEvent` enum
- `src/terminal/tui/model.rs` - Device tracking
- `src/terminal/tui/renderers/tuners_caladan.rs` - Real device rendering
- `bin/scanner.rs` - Discovery service integration with TUI
- `Cargo.toml` - Added `libc = "0.2"` dependency
- `tests/discovery_test.rs` - Updated to use `create_for_testing()`

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

/// Strategy for discovering SDR devices beyond SoapySDR enumeration
pub trait DeviceEnumerator: Send {
    /// Enumerate all currently connected devices
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>>;

    /// Get display name for this enumerator
    fn name(&self) -> &str;
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

### Device Enumeration Strategy

The discovery service uses multiple sources to build a complete device list:

1. **Backend Enumeration** - Query SDR backends (SoapySDR, native crates)
2. **USB Device Inspection** - Parse USB vendor/product IDs for known SDR devices
3. **Serial Device Scanning** - Check /dev/ttyUSB*, /dev/ttyACM* for serial-based SDRs

This multi-source approach ensures:
- Works even if SoapySDR modules are missing
- Detects devices before drivers are loaded
- Allows migration away from SoapySDR
- Identifies devices by hardware characteristics

**Conflict Resolution**: When the same device is found by multiple sources:
- Backend enumeration (SourcePriority::Backend = 2) takes precedence over USB inspection
- Backend-provided info is more complete and authoritative (actual driver communication)
- USB inspection provides fallback when backend doesn't recognize device

```rust
/// Priority for device information sources (higher = more authoritative)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourcePriority {
    UsbInspection = 1,  // Lowest - hardware detection only
    Backend = 2,        // Higher - actual driver communication
}

/// Combines multiple enumeration strategies with conflict resolution
pub struct MultiEnumerator {
    enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)>,
}

impl MultiEnumerator {
    fn enumerate(&self) -> Vec<sdr::DeviceInfo> {
        let mut devices_by_id = HashMap::new();

        for (enumerator, priority) in &self.enumerators {
            match enumerator.enumerate() {
                Ok(devs) => {
                    debug!(enumerator = enumerator.name(), count = devs.len(),
                           "enumerated devices");
                    for device in devs {
                        let id = device.id.clone();
                        devices_by_id
                            .entry(id)
                            .and_modify(|(existing_dev, existing_priority)| {
                                // Keep device from higher priority source
                                if priority > existing_priority {
                                    *existing_dev = device.clone();
                                    *existing_priority = *priority;
                                }
                            })
                            .or_insert((device, *priority));
                    }
                }
                Err(e) => {
                    debug!(enumerator = enumerator.name(), error = %e,
                           "enumeration failed");
                }
            }
        }

        // Extract devices, sorted by ID for consistency
        let mut devices: Vec<_> = devices_by_id.into_iter()
            .map(|(_, (device, _))| device)
            .collect();
        devices.sort_by(|a, b| a.id.cmp(&b.id));
        devices
    }
}
```

### Linux Implementation (Event-Driven)

```rust
use udev::{MonitorBuilder, EventType};
use std::os::unix::io::AsRawFd;
use nix::poll::{poll, PollFd, PollFlags};

/// Debounce duration to avoid spurious USB enumeration events
const DEBOUNCE_DURATION: Duration = Duration::from_millis(150);

/// Linux-specific discovery via udev events
pub struct Udev {
    enumerator: MultiEnumerator,
    known_devices: HashMap<sdr::DeviceId, sdr::DeviceInfo>,
    pending_rescan: bool,
}

impl Udev {
    pub fn new(enumerator: MultiEnumerator) -> Self {
        Self {
            enumerator,
            known_devices: HashMap::new(),
            pending_rescan: false,
        }
    }
}

impl Service for Udev {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken) {
        let mut socket = match MonitorBuilder::new()
            .and_then(|m| m.match_subsystem("usb"))
            .and_then(|m| m.listen())
        {
            Ok(s) => s,
            Err(e) => {
                debug!(error = %e, "failed to create udev monitor, falling back to polling");
                let mut polling = Polling::new(
                    std::mem::take(&mut self.enumerator),
                    Duration::from_secs(3)
                );
                return polling.run(event_tx, cancel);
            }
        };

        // Initial enumeration to establish baseline
        if self.rescan_devices(&event_tx).is_err() {
            return;
        }

        let fd = socket.as_raw_fd();
        let mut fds = [PollFd::new(fd, PollFlags::POLLIN)];
        let mut last_event_time = Instant::now();

        // Event loop with FD polling
        loop {
            if cancel.is_cancelled() {
                break;
            }

            // Poll with timeout for cancellation responsiveness
            match poll(&mut fds, 100) {
                Ok(n) if n > 0 => {
                    // Drain all pending udev events
                    while let Some(event) = socket.iter().next() {
                        match event.event_type() {
                            EventType::Add | EventType::Remove => {
                                debug!(event_type = ?event.event_type(), "USB event detected");
                                self.pending_rescan = true;
                                last_event_time = Instant::now();
                            }
                            _ => {}
                        }
                    }
                }
                Ok(_) => {
                    // Timeout - check if we should debounce-rescan
                    if self.pending_rescan && last_event_time.elapsed() >= DEBOUNCE_DURATION {
                        self.pending_rescan = false;
                        if self.rescan_devices(&event_tx).is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    debug!(error = ?e, "poll error");
                    break;
                }
            }
        }
    }
}

impl Udev {
    fn rescan_devices(&mut self, event_tx: &mpsc::Sender<Event>) -> Result<(), ()> {
        // Query all enumerators for current devices
        let devices = self.enumerator.enumerate();
        let mut current_devices = HashMap::new();

        for device in devices {
            current_devices.insert(device.id.clone(), device);
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
    enumerator: MultiEnumerator,
    known_devices: HashMap<sdr::DeviceId, sdr::DeviceInfo>,
    poll_interval: Duration,
}

impl Polling {
    pub fn new(enumerator: MultiEnumerator, poll_interval: Duration) -> Self {
        Self {
            enumerator,
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
        let devices = self.enumerator.enumerate();
        let mut current_devices = HashMap::new();

        for device in devices {
            current_devices.insert(device.id.clone(), device);
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

### Enumerator Implementations

```rust
/// Enumerates devices via SDR backends (SoapySDR, native crates)
pub struct BackendEnumerator {
    backends: Vec<Box<dyn sdr::Backend>>,
}

impl DeviceEnumerator for BackendEnumerator {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>> {
        let mut devices = Vec::new();
        for backend in &self.backends {
            if let Ok(devs) = backend.enumerate_devices() {
                devices.extend(devs);
            }
        }
        Ok(devices)
    }

    fn name(&self) -> &str {
        "backend"
    }
}

/// Enumerates SDR devices by inspecting USB vendor/product IDs
#[cfg(target_os = "linux")]
pub struct UsbEnumerator {
    known_devices: HashMap<(u16, u16), &'static str>, // (vid, pid) -> model
}

#[cfg(target_os = "linux")]
impl UsbEnumerator {
    pub fn new() -> Self {
        Self::with_database(Self::default_database())
    }

    pub fn with_database(known_devices: HashMap<(u16, u16), &'static str>) -> Self {
        Self { known_devices }
    }

    fn default_database() -> HashMap<(u16, u16), &'static str> {
        let mut db = HashMap::new();
        // RTL-SDR devices
        db.insert((0x0bda, 0x2838), "RTL-SDR");
        db.insert((0x0bda, 0x2832), "RTL-SDR");
        // HackRF
        db.insert((0x1d50, 0x6089), "HackRF One");
        // AirSpy
        db.insert((0x1d50, 0x60a1), "AirSpy");
        db.insert((0x1d50, 0x60a6), "AirSpy HF+");
        db.insert((0x03eb, 0x800c), "AirSpy Mini");
        // LimeSDR
        db.insert((0x1d50, 0x6108), "LimeSDR-USB");
        db.insert((0x0403, 0x601f), "LimeSDR-Mini");
        // PlutoSDR
        db.insert((0x0456, 0xb673), "PlutoSDR");
        // BladeRF
        db.insert((0x2cf0, 0x5246), "BladeRF");
        db.insert((0x1d50, 0x6066), "BladeRF 2.0");
        // Add more as needed
        db
    }
}

#[cfg(target_os = "linux")]
impl DeviceEnumerator for UsbEnumerator {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>> {
        let mut devices = Vec::new();
        let mut enumerator = udev::Enumerator::new()?;
        enumerator.match_subsystem("usb")?;

        for device in enumerator.scan_devices()? {
            if let (Some(vid), Some(pid)) = (
                device.attribute_value("idVendor"),
                device.attribute_value("idProduct"),
            ) {
                let vid = u16::from_str_radix(vid.to_str().unwrap_or(""), 16).ok();
                let pid = u16::from_str_radix(pid.to_str().unwrap_or(""), 16).ok();

                if let (Some(vid), Some(pid)) = (vid, pid) {
                    if let Some(model) = self.known_devices.get(&(vid, pid)) {
                        // Create DeviceInfo from USB device
                        let serial = device.property_value("ID_SERIAL_SHORT")
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");

                        // USB serial numbers are often unreliable (missing, duplicated, or bogus)
                        // Combine with USB bus/port location for better uniqueness
                        let bus = device.property_value("BUSNUM")
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");
                        let port = device.property_value("DEVNUM")
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");

                        devices.push(sdr::DeviceInfo {
                            id: sdr::DeviceId::Usb {
                                vid,
                                pid,
                                serial: serial.to_string(),
                                bus_port: format!("{}-{}", bus, port), // Physical location
                            },
                            model: model.to_string(),
                            // Other fields...
                        });
                    }
                }
            }
        }

        Ok(devices)
    }

    fn name(&self) -> &str {
        "usb"
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
    // Build enumerator with multiple sources, prioritized by reliability
    let mut enumerators: Vec<(Box<dyn DeviceEnumerator>, SourcePriority)> = vec![
        (Box::new(BackendEnumerator { backends }), SourcePriority::Backend),
    ];

    #[cfg(target_os = "linux")]
    enumerators.push((
        Box::new(UsbEnumerator::new()),
        SourcePriority::UsbInspection,
    ));

    let enumerator = MultiEnumerator { enumerators };

    match mode {
        DiscoveryMode::ForcePolling(interval) => {
            Box::new(Polling::new(enumerator, interval))
        }
        #[cfg(target_os = "linux")]
        DiscoveryMode::ForceUdev => {
            Box::new(Udev::new(enumerator))
        }
        DiscoveryMode::Auto => {
            #[cfg(target_os = "linux")]
            {
                Box::new(Udev::new(enumerator))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Box::new(Polling::new(enumerator, Duration::from_secs(3)))
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
udev = "0.9"  # Actively maintained (libudev is deprecated)
nix = { version = "0.29", features = ["poll"] }  # For FD polling
```

### Step 2: Create Module Structure

1. Create `src/discovery/mod.rs`
2. Create `src/discovery/service.rs` - `Service` trait and `Event` enum
3. Create `src/discovery/enumerator.rs` - `DeviceEnumerator` trait
4. Create `src/discovery/backend_enum.rs` - Backend-based enumeration
5. Create `src/discovery/usb_enum.rs` - USB VID/PID enumeration (Linux)
6. Create `src/discovery/udev.rs` - Linux udev implementation
7. Create `src/discovery/polling.rs` - Fallback polling implementation

### Step 3: Implement Enumerator Trait and Implementations

1. Define `DeviceEnumerator` trait
2. Implement `BackendEnumerator` (wraps existing backend enumeration)
3. Implement `UsbEnumerator` for Linux (USB VID/PID lookup)
4. Implement `MultiEnumerator` (combines multiple sources)
5. Add known device database (RTL-SDR, HackRF, AirSpy, etc.)

### Step 4: Implement Polling Discovery (Simplest First)

1. Implement `Polling` service using `MultiEnumerator`
2. Add configurable poll interval (default 3 seconds)
3. Test on current platform
4. Verify device detection works with multiple sources

### Step 5: Implement udev Discovery (Linux Only)

1. Implement `Udev` service using `MultiEnumerator`
2. Use `poll()` on udev FD for proper event-driven monitoring
3. Add debouncing logic (150ms) to handle USB enumeration event bursts
4. Test with real device plug/unplug
5. Verify instant detection with debouncing working correctly

### Step 6: Extract Common Logic

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

### Step 7: Integration with ShutdownCoordinator

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

### Step 8: Testing

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

    // Collect events with timeout (account for debouncing)
    let mut events = Vec::new();
    let timeout = Duration::from_millis(500);
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
    Box::new(sdr::Seify),  // Future native backend
];

let service = discovery::create(backends, discovery::DiscoveryMode::Auto);
// Will detect devices from:
// - All backends (SoapySDR, native crates)
// - USB VID/PID inspection (Linux)
// - Future: serial ports, network devices
```

### Force Polling (Testing)
```rust
// Force polling mode even on Linux (useful for testing)
let service = discovery::create(
    backends,
    discovery::DiscoveryMode::ForcePolling(Duration::from_secs(2))
);
```

### Custom Enumerator
```rust
// Add custom enumeration strategy
struct MyCustomEnumerator;

impl DeviceEnumerator for MyCustomEnumerator {
    fn enumerate(&self) -> Result<Vec<sdr::DeviceInfo>> {
        // Custom device discovery logic
        Ok(vec![])
    }

    fn name(&self) -> &str {
        "custom"
    }
}

// Build custom discovery service
let mut enumerators: Vec<Box<dyn DeviceEnumerator>> = vec![
    Box::new(BackendEnumerator { backends }),
    Box::new(MyCustomEnumerator),
];
let enumerator = MultiEnumerator { enumerators };
let service = Polling::new(enumerator, Duration::from_secs(3));
```

## Benefits

### Linux
✅ **Near-instant detection** - udev events + 150ms debounce (prevents spurious events)
✅ **Zero CPU overhead** - Event-driven with poll(), no busy waiting
✅ **Standard practice** - Uses native Linux hotplug mechanism
✅ **Robust** - Debouncing handles USB enumeration event bursts correctly

### Other Platforms
✅ **Works everywhere** - Polling fallback for macOS, Windows, BSD
✅ **Configurable interval** - Balance between responsiveness and overhead
✅ **Simple implementation** - Just periodic enumeration

### Cross-Platform
✅ **Backend-agnostic** - Works with any `sdr::Backend` implementation
✅ **Multiple enumeration sources** - Backends, USB VID/PID, serial devices
✅ **Automatic platform selection** - Compile-time detection
✅ **Consistent API** - Same `discovery::Event` type on all platforms
✅ **SoapySDR-optional** - Can detect devices without SoapySDR modules
✅ **Future-proof** - Easy to add new enumeration strategies

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
- Detection latency: ~150ms (instant event + 150ms debounce)
- CPU overhead: ~0% (event-driven with poll)
- Memory overhead: ~50KB (udev monitor)
- Debouncing: 150ms to avoid spurious USB enumeration events

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
5. **USB debouncing**: 150ms delay prevents spurious events from USB enumeration process

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
    mod.rs              # Module exports, platform selection, MultiEnumerator
    service.rs          # Service trait and Event enum
    enumerator.rs       # DeviceEnumerator trait
    backend_enum.rs     # BackendEnumerator implementation
    usb_enum.rs         # UsbEnumerator implementation (Linux)
    udev.rs             # Linux udev implementation
    polling.rs          # Polling fallback
    common.rs           # Shared diffing logic
    device_db.rs        # Known device database (VID/PID mappings)
```

## Success Criteria

✅ `Polling` service works on all platforms
✅ `Udev` service works on Linux (instant detection with debouncing)
✅ Graceful fallback if udev unavailable
✅ Integrates with ShutdownCoordinator
✅ Manual hotplug test passes (plug/unplug detected)
✅ `DeviceEnumerator` trait supports multiple enumeration strategies
✅ `BackendEnumerator` works with SoapySDR and future backends
✅ `UsbEnumerator` detects devices by USB VID/PID (Linux)
✅ `MultiEnumerator` correctly deduplicates devices from multiple sources
✅ Device detection works even without SoapySDR modules loaded

## Future Considerations

### Async Alternative
When switching to a hybrid async/sync model in the future, consider `tokio-udev` crate:
- Provides `Stream`-based API for device events
- Integrates naturally with tokio runtime
- Same udev backend, async interface
- Would replace blocking thread with async task

Example future API:
```rust
#[cfg(feature = "async")]
pub async fn create_async(backends: Vec<Box<dyn sdr::Backend>>) -> impl Stream<Item = Event> {
    // Use tokio-udev::MonitorSocket
}
```

### Device ID Design Considerations

**USB Serial Number Reliability**: Research shows USB serial numbers are often unreliable:
- Many devices have no serial number at all
- Some manufacturers use hard-coded serial numbers (all devices report same value)
- Serial numbers may be bogus (e.g., GoPro cameras report "123456789ABC")

**Recommended Approach**: Combine multiple identifiers for uniqueness:
- VID/PID (identifies device type)
- Serial number (if present and non-zero)
- USB bus/port location (physical topology)

This composite approach provides better uniqueness than serial number alone, especially when multiple identical devices are connected.

### Expanding Device Database
The USB VID/PID database in `device_db.rs` should be expanded over time:
- Database now includes RTL-SDR, HackRF, AirSpy, LimeSDR, PlutoSDR, BladeRF
- Add more variants as discovered
- Consider loading from external config file for user-added devices
- Consider checking against linux-usb.org database or similar

### Additional Enumerators
Future enumerator implementations could include:
- **SerialEnumerator**: Scan `/dev/ttyUSB*`, `/dev/ttyACM*` for serial-based SDRs
- **NetworkEnumerator**: Discover network-attached SDRs (e.g., KiwiSDR, remote RTL-TCP)
- **WindowsEnumerator**: Use Windows Device Manager API on Windows platforms
- **MacOSEnumerator**: Use IOKit for device enumeration on macOS

## Next Steps

After completing this plan:
1. **Plan 007**: Device Pool (receives `discovery::Event`s, manages inventory)
2. **Plan 010**: Multi-SDR Orchestration (uses discovery + pool together)
