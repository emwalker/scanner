# Plan 005: Backend Abstraction Layer

**Date**: October 2025
**Status**: ✅ Completed
**Dependencies**: None (foundational)
**Related Plans**: `004-multi-sdr.md` (parent plan)
**Enables**: Plans 006, 007, 008

## Executive Summary

Introduce a thin abstraction layer over SDR backend implementations to:
1. **Isolate SoapySDR dependency** - Keep migration path to native Rust drivers open
2. **Enable multiple backends** - Can use SoapySDR, Seify, rtl-sdr-rs simultaneously
3. **Simplify testing** - Easy to mock backends for tests
4. **Future-proof architecture** - Switch backends without changing pool/task logic

This is the foundational layer that all subsequent multi-SDR plans depend on.

## Problem Statement

Current code directly uses `soapysdr` crate throughout:
```rust
// Direct coupling to SoapySDR
let device = soapysdr::Device::new(args)?;
let devices = soapysdr::enumerate()?;
```

**Issues**:
- Increasing SoapySDR dependency as we add multi-SDR features
- No migration path to native Rust drivers (Seify, rtl-sdr-rs)
- Harder to test (requires real hardware)
- Can't mix backends (e.g., native RTL-SDR + SoapySDR for others)

## Goal

Create backend abstraction that makes SoapySDR just one implementation choice among many, while maintaining type safety and proper integration with rustradio's graph-based architecture.

## Design

### Core Traits

```rust
// src/sdr/backend.rs
/// Abstraction over different SDR backend implementations
pub trait Backend: Send + Sync {
    /// Enumerate all devices this backend can access
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Open a specific device by ID
    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn Device>>;

    /// Backend identifier (e.g., "SoapySDR", "Seify", "rtl-sdr-rs")
    fn name(&self) -> &str;
}

// src/sdr/device.rs
/// Individual device abstraction (backend-agnostic)
pub trait Device: Send {
    /// Stable device identifier
    fn id(&self) -> &DeviceId;

    /// Device capabilities (frequency range, sample rates, etc.)
    fn capabilities(&self) -> &Capabilities;

    /// Add source block to rustradio graph and return stream handle
    /// This is called each time a new graph is created
    ///
    /// Unlike returning Box<dyn Any>, this maintains type safety by always
    /// returning a concrete ReadStream<Complex> that works with rustradio
    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<rustradio::Complex>>;

    /// Tune to frequency (for devices that support runtime retuning)
    /// Not all devices support this - some require rebuilding the graph
    fn tune(&mut self, freq: f64) -> Result<()>;

    /// Set gain (for devices that support runtime gain adjustment)
    fn set_gain(&mut self, gain: f64) -> Result<()>;

    /// Consume device and return backend-specific representation
    /// Provides escape hatch for advanced users who need direct backend access
    /// Following embedded-hal best practices
    fn into_inner(self: Box<Self>) -> Box<dyn Any>;
}
```

**Why `add_source_to_graph()` instead of `create_source()`?**

The original design used `fn create_source() -> Result<Box<dyn Any>>` which is type-unsafe and requires downcasting. The new design:
- ✅ Maintains type safety (always returns `ReadStream<Complex>`)
- ✅ Integrates naturally with rustradio's graph system
- ✅ Backend-agnostic (caller doesn't care about concrete block types)
- ✅ No downcasting required

**Design validated by existing Rust SDR ecosystem:**
- **rustradio** uses this exact pattern: blocks added to graph, streams returned
- **FutureSDR** uses similar approach with async/sync block variants
- **Seify** (native Rust HAL) uses `DeviceTrait` pattern similar to ours
- **embedded-hal** recommends trait-based abstraction with fallible methods

**Performance considerations:**
- Dynamic dispatch costs ~25 CPU cycles per call
- Factor 1.2x overhead when dispatch is outside tight loops
- **Our case:** `add_source_to_graph()` called once per graph creation (not in sample processing loop)
- **Impact:** Negligible (milliseconds during setup, zero during sample processing)

### Device Types

```rust
// src/sdr/types.rs
/// Device information returned by enumeration
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub serial: String,
    pub model: String,
    pub backend: String,  // Which backend provides this device
}

/// Stable device identifier
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct DeviceId(String);

impl DeviceId {
    pub fn from_serial(backend: &str, serial: &str) -> Self {
        Self(format!("{backend}:{serial}"))
    }

    pub fn backend(&self) -> &str {
        self.0.split(':').next().unwrap_or("unknown")
    }

    pub fn serial(&self) -> &str {
        self.0.split(':').nth(1).unwrap_or("unknown")
    }
}

/// Comprehensive device capabilities
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub device_id: DeviceId,

    // Frequency ranges (min, max) in Hz
    pub rx_frequency_ranges: Vec<(f64, f64)>,

    // Sample rate ranges (min, max) in Hz
    pub rx_sample_rate_ranges: Vec<(f64, f64)>,

    // Gain range (min, max) in dB
    pub gain_range: (f64, f64),

    // Optional features
    pub has_agc: bool,
    pub antenna_options: Vec<String>,
    pub channels: usize,

    // Performance characteristics (for device pool allocation)
    pub max_bandwidth: f64,
    pub typical_latency_us: u64,
}

impl Capabilities {
    pub fn from_soapy_device(device: &soapysdr::Device) -> Result<Self> {
        let driver = device.driver_key()?;
        let hardware_info = device.hardware_info()?;

        let rx_freq_ranges = device
            .frequency_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        let rx_sample_rate_ranges = device
            .sample_rate_range(soapysdr::Direction::Rx, 0)?
            .into_iter()
            .map(|r| (r.minimum, r.maximum))
            .collect();

        let gain_range = {
            let r = device.gain_range(soapysdr::Direction::Rx, 0)?;
            (r.minimum, r.maximum)
        };

        let max_bandwidth = rx_sample_rate_ranges
            .last()
            .map(|r| r.1)
            .unwrap_or(0.0);

        // Estimate latency based on driver type
        let typical_latency_us = match driver.as_str() {
            "rtlsdr" => 50,      // Fast USB2.0 device
            "sdrplay" => 100,    // Moderate latency
            "hackrf" => 75,      // Fast USB2.0
            "lime" => 150,       // Higher latency
            _ => 100,            // Default estimate
        };

        Ok(Self {
            device_id: DeviceId::from_serial(&driver, &hardware_info),
            rx_frequency_ranges: rx_freq_ranges,
            rx_sample_rate_ranges,
            gain_range,
            has_agc: device.has_gain_mode(soapysdr::Direction::Rx, 0).unwrap_or(false),
            antenna_options: device
                .list_antennas(soapysdr::Direction::Rx, 0)
                .unwrap_or_default(),
            channels: device.num_channels(soapysdr::Direction::Rx).unwrap_or(1),
            max_bandwidth,
            typical_latency_us,
        })
    }

    /// Check if device supports a given frequency
    pub fn supports_frequency(&self, freq: f64) -> bool {
        self.rx_frequency_ranges
            .iter()
            .any(|(min, max)| freq >= *min && freq <= *max)
    }

    /// Check if device supports a given sample rate
    pub fn supports_sample_rate(&self, rate: f64) -> bool {
        self.rx_sample_rate_ranges
            .iter()
            .any(|(min, max)| rate >= *min && rate <= *max)
    }
}
```

### Error Handling

```rust
// src/sdr/types.rs
#[derive(Debug)]
pub struct DeviceError {
    pub kind: DeviceErrorKind,
    pub backend: String,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceErrorKind {
    NotFound,
    NotAvailable,
    InvalidParameter,
    HardwareError,
    Timeout,
    Unsupported,
}

impl DeviceError {
    pub fn new(kind: DeviceErrorKind, backend: &str, details: impl Into<String>) -> Self {
        Self {
            kind,
            backend: backend.to_string(),
            details: details.into(),
        }
    }
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} error from {}: {}",
            match self.kind {
                DeviceErrorKind::NotFound => "Device not found",
                DeviceErrorKind::NotAvailable => "Device not available",
                DeviceErrorKind::InvalidParameter => "Invalid parameter",
                DeviceErrorKind::HardwareError => "Hardware error",
                DeviceErrorKind::Timeout => "Timeout",
                DeviceErrorKind::Unsupported => "Unsupported operation",
            },
            self.backend,
            self.details
        )
    }
}

impl std::error::Error for DeviceError {}

impl From<soapysdr::Error> for DeviceError {
    fn from(e: soapysdr::Error) -> Self {
        // Map SoapySDR errors to generic kinds
        let kind = match e.to_string().as_str() {
            s if s.contains("not found") => DeviceErrorKind::NotFound,
            s if s.contains("timeout") => DeviceErrorKind::Timeout,
            s if s.contains("not supported") => DeviceErrorKind::Unsupported,
            _ => DeviceErrorKind::HardwareError,
        };

        Self {
            kind,
            backend: "SoapySDR".to_string(),
            details: e.to_string(),
        }
    }
}
```

### SoapySDR Backend Implementation

```rust
// src/sdr/soapy.rs
use super::{Backend, Device, DeviceInfo, DeviceId, Capabilities, DeviceError};
use crate::types::Result;

/// SoapySDR backend (current implementation)
pub struct Soapy;

impl Backend for Soapy {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        let devices = soapysdr::enumerate("")?;

        Ok(devices.into_iter().map(|d| {
            let serial = d.get("serial").unwrap_or("unknown").to_string();
            let model = d.get("label").unwrap_or("Unknown").to_string();
            let driver = d.get("driver").unwrap_or("soapy").to_string();

            DeviceInfo {
                id: DeviceId::from_serial(&driver, &serial),
                serial,
                model,
                backend: "SoapySDR".to_string(),
            }
        }).collect())
    }

    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn Device>> {
        // Build device args from ID
        let backend = id.backend();
        let serial = id.serial();
        let args = format!("driver={},serial={}", backend, serial);

        Ok(Box::new(SoapyDevice::new(args)?))
    }

    fn name(&self) -> &str {
        "SoapySDR"
    }
}

/// SoapySDR device wrapper
///
/// Important: We store device_args (String) not soapysdr::Device because:
/// 1. rustradio's SoapySdrSource::builder() consumes the device
/// 2. We need to create multiple graphs from the same device
/// 3. Creating a fresh device each time is safe with SoapySDR
pub struct SoapyDevice {
    device_id: DeviceId,
    device_args: String,
    capabilities: Capabilities,
}

impl SoapyDevice {
    pub fn new(device_args: String) -> Result<Self> {
        // Create temporary device to query capabilities
        let temp_device = soapysdr::Device::new(&device_args)?;

        let capabilities = Capabilities::from_soapy_device(&temp_device)?;
        let device_id = capabilities.device_id.clone();

        Ok(Self {
            device_id,
            device_args,
            capabilities,
        })
    }
}

impl Device for SoapyDevice {
    fn id(&self) -> &DeviceId {
        &self.device_id
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<rustradio::Complex>> {
        // Create fresh device for this graph
        let device = soapysdr::Device::new(&self.device_args)?;

        // Configure device
        if device.has_gain_mode(soapysdr::Direction::Rx, 0)? {
            device.set_gain_mode(soapysdr::Direction::Rx, 0, false)?;
        }

        // Normalize gain to 0.0-1.0 range (SDRplay uses 0-48 dB)
        let normalized_gain = (gain_db.clamp(0.0, 48.0)) / 48.0;

        // Build source and add to graph
        let (source_block, output_stream) =
            rustradio::blocks::SoapySdrSource::builder(&device, freq, samp_rate)
                .igain(normalized_gain)
                .build()?;

        graph.add(Box::new(source_block));
        Ok(output_stream)
    }

    fn tune(&mut self, freq: f64) -> Result<()> {
        // Note: This requires recreating the device
        // Most efficient to rebuild the graph instead
        let device = soapysdr::Device::new(&self.device_args)?;
        device.set_frequency(soapysdr::Direction::Rx, 0, freq, "")?;
        Ok(())
    }

    fn set_gain(&mut self, gain: f64) -> Result<()> {
        let device = soapysdr::Device::new(&self.device_args)?;
        device.set_gain(soapysdr::Direction::Rx, 0, gain)?;
        Ok(())
    }

    fn into_inner(self: Box<Self>) -> Box<dyn Any> {
        // Return device args as the "raw" representation
        Box::new(self.device_args)
    }
}
```

### Mock Backend for Testing

```rust
// src/sdr/mock.rs
use super::{Backend, Device, DeviceInfo, DeviceId, Capabilities, DeviceError, DeviceErrorKind};
use crate::types::Result;
use rustradio::Complex;

/// Mock backend for testing without hardware
pub struct Mock;

impl Backend for Mock {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![
            DeviceInfo {
                id: DeviceId::from_serial("mock", "001"),
                serial: "001".to_string(),
                model: "Mock RTL-SDR".to_string(),
                backend: "Mock".to_string(),
            },
            DeviceInfo {
                id: DeviceId::from_serial("mock", "002"),
                serial: "002".to_string(),
                model: "Mock SDRplay".to_string(),
                backend: "Mock".to_string(),
            },
        ])
    }

    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn Device>> {
        Ok(Box::new(MockDevice::new(id.clone(), false)))
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

/// Mock device implementation
pub struct MockDevice {
    device_id: DeviceId,
    capabilities: Capabilities,
    fail_on_tune: bool,
}

impl MockDevice {
    pub fn new(device_id: DeviceId, fail_on_tune: bool) -> Self {
        let capabilities = Capabilities {
            device_id: device_id.clone(),
            rx_frequency_ranges: vec![(24e6, 1766e6)],  // Typical RTL-SDR range
            rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
            gain_range: (0.0, 48.0),
            has_agc: true,
            antenna_options: vec!["RX".to_string()],
            channels: 1,
            max_bandwidth: 2_400_000.0,
            typical_latency_us: 50,
        };

        Self {
            device_id,
            capabilities,
            fail_on_tune,
        }
    }
}

impl Device for MockDevice {
    fn id(&self) -> &DeviceId {
        &self.device_id
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        _gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        // Generate test signal: 100 kHz tone at center frequency
        let tone_freq = 100_000.0;
        let samples_per_period = (samp_rate / tone_freq) as usize;
        let total_samples = samples_per_period * 10; // 10 periods

        let samples: Vec<Complex> = (0..total_samples)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * tone_freq * i as f32 / samp_rate as f32;
                Complex::new(phase.cos() * 0.5, phase.sin() * 0.5)
            })
            .collect();

        let (source, stream) = rustradio::blocks::VectorSource::new(samples, true);
        graph.add(Box::new(source));
        Ok(stream)
    }

    fn tune(&mut self, _freq: f64) -> Result<()> {
        if self.fail_on_tune {
            Err(DeviceError::new(
                DeviceErrorKind::HardwareError,
                "Mock",
                "simulated tuning failure"
            ).into())
        } else {
            Ok(())
        }
    }

    fn set_gain(&mut self, _gain: f64) -> Result<()> {
        Ok(())
    }

    fn into_inner(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.device_id)
    }
}
```

### Seify Backend (Native Rust)

**Status Update (2025):** Seify is actively developed and maturing faster than expected:
- Version 0.17.0 published ~1 month ago (actively maintained)
- Native RTL-SDR driver already implemented (`seify-rtlsdr`)
- Uses `DeviceTrait` + `RxStreamer`/`TxStreamer` pattern
- Designed for zero-installation (no system libraries via rusb)

**Migration Strategy:**

When Seify becomes production-ready (likely 2025-2026), we'll add a Seify backend that bridges its streaming API to rustradio's graph blocks:

```rust
// src/sdr/seify_backend.rs
pub struct SeifyDevice {
    inner: seify::Device,
    capabilities: Capabilities,
}

impl Device for SeifyDevice {
    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<rustradio::Complex>> {
        // Bridge Seify's RxStreamer to rustradio block
        let rx_streamer = self.inner.rx_streamer(&[0])?;
        let bridge_block = SeifyBridgeBlock::new(rx_streamer, freq, samp_rate, gain_db);
        let stream = bridge_block.output_stream();
        graph.add(Box::new(bridge_block));
        Ok(stream)
    }
}

/// Bridges Seify's streaming API to rustradio's block-based system
struct SeifyBridgeBlock {
    streamer: seify::RxStreamer,
    // ... configuration
}

impl rustradio::Block for SeifyBridgeBlock {
    fn work(&mut self) -> rustradio::Result<()> {
        // Read from Seify streamer, push to rustradio stream
        let mut buffer = vec![Complex::default(); 1024];
        let n = self.streamer.read(&[&mut buffer], 100_000)?;
        // ... push to output stream
        Ok(())
    }
}
```

### Future Backend Stubs

```rust
// src/sdr/seify.rs
/// Seify backend (native Rust)
/// NOTE: Seify is actively developed - consider implementing this in Phase 2
pub struct Seify;

impl Backend for Seify {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        // When Seify is ready:
        // let devices = seify::enumerate()?;
        Ok(vec![])  // Stub for now
    }

    fn open_device(&self, _id: &DeviceId) -> Result<Box<dyn Device>> {
        // TODO: Implement when Seify reaches v1.0
        // Will require SeifyBridgeBlock to connect to rustradio
        Err(DeviceError::new(
            DeviceErrorKind::Unsupported,
            "Seify",
            "Seify backend planned for Phase 2 (when v1.0 is released)"
        ).into())
    }

    fn name(&self) -> &str {
        "Seify (native Rust)"
    }

    // When implemented, will look like:
    // fn open_device(&self, id: &DeviceId) -> Result<Box<dyn Device>> {
    //     let device = seify::Device::new()?;
    //     Ok(Box::new(SeifyDevice::new(device)?))
    // }
}

// src/sdr/rtlsdr.rs
/// RTL-SDR native backend (future optimization)
pub struct RtlSdr;

impl Backend for RtlSdr {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        // When rtl-sdr-rs is integrated:
        // let devices = rtlsdr::get_device_count()?;
        Ok(vec![])  // Stub for now
    }

    fn open_device(&self, _id: &DeviceId) -> Result<Box<dyn Device>> {
        Err(DeviceError::new(
            DeviceErrorKind::Unsupported,
            "RtlSdr",
            "rtl-sdr-rs backend not yet implemented"
        ).into())
    }

    fn name(&self) -> &str {
        "rtl-sdr-rs (native)"
    }
}
```

## Rustradio Integration

The key insight is that the abstraction must work *with* rustradio, not around it.

### Current Pattern (soapy.rs:136-164)

```rust
// Current SoapySdrManager creates graphs like this:
let mut graph = Graph::new();
let (sdr_source_block, sdr_output_stream) = self
    .sdr_source
    .lock()
    .unwrap()
    .create_raw_source_block(freq, self.samp_rate, self.sdr_gain, self.agc_settling_time)?;

graph.add(Box::new(sdr_source_block));
// ... add more blocks
graph.run()?;
```

### With Abstraction

```rust
// SoapySdrManager can now work with any backend:
let mut graph = Graph::new();

// Device abstraction hides backend-specific details
let output_stream = self.device.add_source_to_graph(
    &mut graph,
    freq,
    self.samp_rate,
    self.sdr_gain,
)?;

// ... add more blocks
graph.run()?;
```

**No changes needed** to existing graph construction logic!

## Implementation Steps

### Step 1: Create Module Structure

1. Create `src/sdr/mod.rs`
2. Create `src/sdr/backend.rs` - Backend trait
3. Create `src/sdr/device.rs` - Device trait
4. Create `src/sdr/types.rs` - DeviceInfo, DeviceId, Capabilities, DeviceError
5. Create `src/sdr/soapy.rs` - Soapy backend implementation
6. Create `src/sdr/mock.rs` - Mock backend for testing
7. Create `src/sdr/seify.rs` - Seify stub
8. Create `src/sdr/rtlsdr.rs` - RtlSdr stub

### Step 2: Implement Core Traits

1. Define `Backend` trait in `backend.rs`
2. Define `Device` trait in `device.rs`
3. Define types in `types.rs`:
   - `DeviceInfo`
   - `DeviceId` with `from_serial()`, `backend()`, `serial()` methods
   - `Capabilities` with comprehensive fields
   - `DeviceError` with `DeviceErrorKind` enum
4. Add documentation with examples

### Step 3: Implement Soapy Backend

1. Implement `Soapy` backend in `soapy.rs`
2. Implement `SoapyDevice` (storing device args, not device instance)
3. Implement `Capabilities::from_soapy_device()` helper
4. Test with real hardware

### Step 4: Implement Mock Backend

1. Implement `Mock` backend in `mock.rs`
2. Implement `MockDevice` with configurable behavior
3. Support failure injection for robustness testing
4. Generate realistic test signals (sine waves)

### Step 5: Add Future Backend Stubs

1. Create `src/sdr/seify.rs` with stub returning `Unsupported` error
2. Create `src/sdr/rtlsdr.rs` with stub returning `Unsupported` error
3. Document migration path in each stub
4. Add feature flags:
   ```toml
   [features]
   seify = ["dep:seify"]  # Future
   rtlsdr = ["dep:rtlsdr"]  # Future
   ```

### Step 6: Update SoapySdrManager (Minimal Changes)

Current `SoapySdrManager` in `src/soapy.rs`:
```rust
pub struct SoapySdrManager {
    sdr_source: Arc<Mutex<SoapySdrSource>>,
    // ...
}
```

Change to:
```rust
pub struct SoapySdrManager {
    device: Box<dyn Device>,  // Backend-agnostic!
    // ...
}
```

Update `start_sdr_graph()`:
```rust
// Before
let (sdr_source_block, sdr_output_stream) = self
    .sdr_source
    .lock()
    .unwrap()
    .create_raw_source_block(freq, self.samp_rate, self.sdr_gain, self.agc_settling_time)?;

// After
let sdr_output_stream = self.device.add_source_to_graph(
    &mut graph,
    freq,
    self.samp_rate,
    self.sdr_gain,
)?;
```

### Step 7: Testing

```rust
// src/sdr/tests.rs

#[test]
fn test_device_id_creation() {
    let id1 = DeviceId::from_serial("soapy", "12345");
    let id2 = DeviceId::from_serial("soapy", "12345");
    assert_eq!(id1, id2);

    assert_eq!(id1.backend(), "soapy");
    assert_eq!(id1.serial(), "12345");
}

#[test]
fn test_soapy_backend_enumeration() {
    let backend = Soapy;
    let devices = backend.enumerate_devices().unwrap();

    assert!(!devices.is_empty(), "Should find connected devices");

    for device in devices {
        assert!(!device.serial.is_empty());
        assert_eq!(device.backend, "SoapySDR");
    }
}

#[test]
fn test_mock_backend_enumeration() {
    let backend = Mock;
    let devices = backend.enumerate_devices().unwrap();

    assert_eq!(devices.len(), 2, "Mock backend should return 2 devices");
    assert_eq!(devices[0].serial, "001");
    assert_eq!(devices[1].serial, "002");
}

#[test]
fn test_mock_device_graph_integration() {
    use rustradio::graph::Graph;

    let backend = Mock;
    let devices = backend.enumerate_devices().unwrap();
    let device = backend.open_device(&devices[0].id).unwrap();

    let mut graph = Graph::new();
    let stream = device
        .add_source_to_graph(&mut graph, 88.9e6, 2.4e6, 20.0)
        .unwrap();

    // Verify stream is valid
    assert!(stream.len() > 0);
}

#[test]
fn test_capabilities_frequency_check() {
    let caps = Capabilities {
        device_id: DeviceId::from_serial("test", "001"),
        rx_frequency_ranges: vec![(24e6, 1766e6)],
        rx_sample_rate_ranges: vec![(225_000.0, 2_400_000.0)],
        gain_range: (0.0, 48.0),
        has_agc: true,
        antenna_options: vec![],
        channels: 1,
        max_bandwidth: 2_400_000.0,
        typical_latency_us: 50,
    };

    assert!(caps.supports_frequency(88.9e6));
    assert!(!caps.supports_frequency(10e6));  // Below range
    assert!(!caps.supports_frequency(2000e6));  // Above range
}

#[test]
fn test_device_error_from_soapy() {
    // Simulate SoapySDR error
    let soapy_err = soapysdr::Error::new(-1, "Device not found");
    let device_err: DeviceError = soapy_err.into();

    assert_eq!(device_err.kind, DeviceErrorKind::NotFound);
    assert_eq!(device_err.backend, "SoapySDR");
}
```

## Usage Patterns

### Current (Direct SoapySDR)
```rust
// In bin/scanner.rs or wherever devices are created
let device_args = format!("driver={},serial={}", driver, serial);
let device = soapy::Device::new(device_args)?;
let manager = SoapySdrManager::new(config, center_freq, device)?;
```

### With Abstraction (Phase 1)
```rust
use crate::sdr::{Backend, Soapy};

// Enumerate devices
let backend = Soapy;
let devices = backend.enumerate_devices()?;

// Open device
let device = backend.open_device(&devices[0].id)?;

// Create manager (works with any backend)
let manager = SoapySdrManager::new(config, center_freq, device)?;
```

### Future (Multiple Backends)
```rust
use crate::sdr::{Backend, Soapy, Seify, RtlSdr};

let backends: Vec<Box<dyn Backend>> = vec![
    Box::new(Soapy),
    Box::new(Seify),
    Box::new(RtlSdr),
];

// Enumerate from all backends
let mut all_devices = Vec::new();
for backend in &backends {
    all_devices.extend(backend.enumerate_devices()?);
}

// Find suitable device for FM scanning
let device_info = all_devices
    .into_iter()
    .find(|d| {
        // Could check capabilities here
        d.backend == "SoapySDR"
    })
    .ok_or("No suitable device found")?;

// Open device from appropriate backend
let backend = backends
    .iter()
    .find(|b| b.name().contains(&device_info.backend))
    .unwrap();

let device = backend.open_device(&device_info.id)?;
```

### Testing Pattern
```rust
#[test]
fn test_scanning_without_hardware() {
    use crate::sdr::{Backend, Mock};

    let backend = Mock;
    let devices = backend.enumerate_devices().unwrap();
    let device = backend.open_device(&devices[0].id).unwrap();

    // Test device pool, task scheduler, etc. without real hardware
    let pool = DevicePool::new();
    pool.add_device(device);
    // ... rest of test
}
```

## Benefits

### Immediate
✅ **Type safety** - No `Box<dyn Any>` downcasting required
✅ **Rustradio integration** - Natural fit with graph-based architecture
✅ **Testability** - Mock backend enables hardware-free testing
✅ **Comprehensive capabilities** - Device pool can make smart allocation decisions
✅ **Unified error handling** - Backend-agnostic error types
✅ **Escape hatch** - `into_inner()` provides direct backend access for advanced users
✅ **Validated design** - Aligns with embedded-hal, FutureSDR, and Seify patterns

### Future
✅ **Multiple backends** - Use SoapySDR + native drivers simultaneously
✅ **Performance** - Native drivers can be faster (no C FFI overhead)
✅ **Reduced dependencies** - Eventually could remove SoapySDR for some devices
✅ **Flexibility** - Easy to add new backends as they mature
✅ **Seify ready** - Clear migration path to native Rust SDR abstraction

## Compatibility

### With Current Code (src/soapy.rs)
✅ Minimal changes to `SoapySdrManager`
✅ Keep existing graph construction patterns
✅ Backward compatible - can still use SoapySDR directly if needed

### With Plan 006 (Device Discovery)
✅ Discovery service uses `Backend::enumerate_devices()`
✅ Backend-agnostic discovery logic

### With Plan 007 (Device Pool)
✅ Pool stores `Box<dyn Device>` (backend-agnostic)
✅ Capabilities enable smart device allocation
✅ Pool can manage devices from multiple backends

### With Plan 008 (Subprocess IPC)
✅ Subprocess layer wraps `Box<dyn Device>`
✅ Works with any backend (SoapySDR, Seify, etc.)

## Migration Timeline

### Phase 1 (Now - 2025 Q4)
- Implement core traits with `add_source_to_graph()` pattern
- Add `Soapy` backend
- Add `Mock` backend for testing
- Add `into_inner()` escape hatch method
- Update `SoapySdrManager` to use `Box<dyn Device>`
- All existing tests pass

### Phase 2 (2025-2026 - When Seify v1.0 releases)
- **Priority**: Implement `Seify` backend with bridge pattern
- Create `SeifyBridgeBlock` to connect Seify's RxStreamer to rustradio
- Add feature flag: `--features seify`
- Test with RTL-SDR via native Seify driver
- Benchmark performance vs SoapySDR
- Support both backends simultaneously

**Why Seify in Phase 2:**
- Seify is actively developed (v0.17.0, updated monthly)
- Native RTL-SDR driver already exists
- Zero-installation design (no system dependencies)
- Clear path to WASM and async support

### Phase 3 (Optional optimization)
- Implement `RtlSdr` backend for direct RTL-SDR access (if not using Seify)
- Measure performance gain vs SoapySDR
- Use native if significantly faster

### Phase 4 (Optional - Async support)
- When FutureSDR-style async becomes relevant
- Add optional `AsyncDevice` trait variant
- Async `add_source_to_graph_async()` method
- Note: Current sync design is sufficient for rustradio

## File Structure

```
src/
  sdr/
    mod.rs              # Module exports
    backend.rs          # Backend trait
    device.rs           # Device trait
    types.rs            # DeviceInfo, DeviceId, Capabilities, DeviceError
    soapy.rs            # Soapy backend implementation
    mock.rs             # Mock backend for testing
    seify.rs            # Seify stub (future)
    rtlsdr.rs           # RtlSdr stub (future)
    tests.rs            # Integration tests
```

## Design Rationale & Research

This design was validated against existing Rust SDR ecosystem patterns:

### rustradio Graph Pattern
- Blocks connected by unidirectional streams
- Flow from sources (no inputs) to sinks (no outputs)
- Our `add_source_to_graph()` fits this model perfectly
- Returning `ReadStream<Complex>` is the established abstraction

### embedded-hal Best Practices
Following the embedded Rust community's hardware abstraction patterns:
- ✅ All trait methods are fallible (return `Result`)
- ✅ Device-specific details are erased (no magic numbers in API)
- ✅ Provide `into_inner()` to consume wrapper and return raw peripheral
- ✅ Reduces complexity from M×N to M+N (M backends, N features)

### FutureSDR Async Architecture
- Supports both sync and async blocks
- Performance penalty exists for async but deemed acceptable
- Clear separation between sync/async implementations
- Our sync design aligns; async can be added later if needed

### Seify Native Rust HAL
- Uses `DeviceTrait` pattern similar to our `Device` trait
- `RxStreamer`/`TxStreamer` for streaming abstraction
- Our design provides clear bridge pattern for integration
- Active development (v0.17.0, monthly updates)

### Performance Analysis
**Dynamic Dispatch:**
- Virtual method calls: ~25 CPU cycles
- Factor 1.2x overhead when decision is outside tight loops
- Factor 3.4x overhead when decision is inside tight loops

**Our Case:**
- `add_source_to_graph()` called once during graph setup
- Sample processing happens in rustradio blocks (no virtual dispatch)
- Performance impact: negligible (milliseconds at setup time)

**Alternative Considered: enum_dispatch**
- Could eliminate virtual dispatch overhead
- Provides up to 10x improvement in tight loops
- **Rejected:** Overkill for our use case (dispatch at setup, not per-sample)

## Success Criteria

✅ `Soapy` backend can enumerate devices
✅ `Soapy` backend can open devices
✅ Devices integrate seamlessly with rustradio graphs (type-safe)
✅ `Mock` backend enables testing without hardware
✅ `Capabilities` provides comprehensive device information
✅ Error handling is unified and informative
✅ All existing tests pass
✅ No `Box<dyn Any>` downcasting anywhere
✅ `into_inner()` provides escape hatch for advanced users
✅ Design aligns with embedded-hal and Seify patterns

## Research & References

This plan was informed by research into existing Rust SDR and hardware abstraction patterns:

### Rust SDR Ecosystem
- **rustradio** (ThomasHabets/rustradio): Graph-based SDR framework, similar to GNURadio
- **FutureSDR** (FutureSDR/FutureSDR): Async SDR runtime with sync/async block support
- **Seify** (FutureSDR/seify): Native Rust SDR HAL, actively developed
- **rust-soapysdr** (kevinmehall/rust-soapysdr): Rust bindings for SoapySDR

### Design Patterns
- **embedded-hal**: Rust embedded systems HAL patterns and best practices
- **wgpu-hal**: Graphics API abstraction (Vulkan, Metal, D3D, GL backends)
- **gfx-hal**: Hardware abstraction for graphics adapters

### Key Insights
1. Trait-based abstraction is the Rust standard for hardware backends
2. Dynamic dispatch overhead is negligible at setup time
3. Seify is maturing faster than expected (ready for Phase 2)
4. `into_inner()` escape hatch is an embedded-hal best practice
5. rustradio graph integration requires stream-based abstraction

## Next Steps

After completing this plan:
1. **Plan 006**: Device Discovery Service (uses `enumerate_devices()`)
2. **Plan 007**: Device Pool (stores `Box<dyn Device>`, uses `Capabilities`)
3. **Plan 008**: Subprocess IPC (wraps `Box<dyn Device>`)

**Future Consideration:**
- Monitor Seify development for v1.0 release
- Implement Seify backend in Phase 2 (2025-2026)
- Consider async support if FutureSDR integration becomes relevant
