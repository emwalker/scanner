# Plan 008: Universal Subprocess IPC

**Date**: October 2025
**Status**: Not Started
**Dependencies**: `005-backend-abstraction.md`, `007-device-pool.md`
**Related Plans**: `004-multi-sdr.md` (parent plan), `003-structured-concurrency-shutdown.md`
**Enables**: Plan 010

## Executive Summary

Run **every** SDR device in isolated subprocess with custom Unix socket IPC.

**Why all devices**:
- **Required**: SDRplay (global handle), BladeRF (API locking)
- **Beneficial**: LimeSDR (stability), all devices (crash/memory isolation)
- **Simplicity**: One code path, no device-type branching

**Architecture**: Custom IPC protocol via Unix domain sockets (not SoapySDRServer).

## Problem Statement

### Multi-Device Challenges

1. **SDRplay**: Global device handle - only one per process
2. **BladeRF**: API-level locking - process isolation required
3. **LimeSDR**: Stability issues documented - isolation helps
4. **All devices**: Driver bugs can crash main process

### Why NOT per-device branching
```rust
// Complex: Different code paths per device type ❌
match device_type {
    DeviceType::SDRplay => spawn_subprocess(...)?,
    DeviceType::BladeRF => spawn_subprocess(...)?,
    DeviceType::RTL => use_inprocess(...)?,
    DeviceType::HackRF => use_inprocess(...)?,
    // Easy to forget updates as we add device types
}
```

### Why universal subprocess
```rust
// Simple: One code path for all ✅
let device = ipc::Device::new(device_id, device_args)?;
// Works for RTL-SDR, SDRplay, HackRF, USRP, LimeSDR, BladeRF, etc.
```

## Goal

Universal subprocess architecture with minimal overhead:
- **~25μs latency** per I/Q packet (Unix sockets)
- **100% crash isolation** (driver bug doesn't kill main process)
- **Automatic cleanup** (kill subprocess = guaranteed resource release)
- **No new dependencies** (we own the IPC code)

## Design

### Architecture

```
Main Process                    Worker Subprocess (per device)
    │                                 │
    ├─ control socket ────────────────┤  (bidirectional commands)
    │  /tmp/scanner-{id}-ctl.sock     │
    │                                 │
    ├─ data socket ───────────────────┤  (unidirectional I/Q stream)
    │  /tmp/scanner-{id}-dat.sock     │
    │                                 │
    └─ manages Child process          └─ soapysdr::Device (any type)
                                        RTL-SDR, SDRplay, HackRF,
                                        USRP, LimeSDR, BladeRF, etc.
```

### IPC Protocol

```rust
/// Control messages (bidirectional)
#[derive(Serialize, Deserialize, Debug)]
pub enum ControlMessage {
    // Main → Worker commands
    Tune { freq_hz: f64 },
    SetGain { gain_db: f64 },
    SetSampleRate { rate: f64 },
    Start,
    Stop,
    Shutdown,

    // Worker → Main responses
    Ready,
    Tuned { actual_freq: f64 },
    GainSet { actual_gain: f64 },
    Error { msg: String },
}

/// I/Q data packet (unidirectional: worker → main)
#[derive(Serialize, Deserialize)]
pub struct IQPacket {
    pub samples: Vec<Complex<f32>>,
    pub timestamp: u64,
    pub sequence: u64,  // For detecting drops
}
```

### Worker Subprocess

```rust
// In bin/scanner.rs
#[derive(Parser)]
struct Args {
    // Normal scanner args...

    /// Worker mode: run as device subprocess
    #[arg(long)]
    device_worker: Option<WorkerArgs>,
}

#[derive(Parser)]
struct WorkerArgs {
    /// Unique device ID
    device_id: String,

    /// SoapySDR device args
    device_args: String,

    /// Sample rate
    #[arg(long)]
    sample_rate: f64,

    /// Initial frequency
    #[arg(long)]
    frequency: f64,

    /// Initial gain
    #[arg(long)]
    gain: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Worker mode for device subprocess
    if let Some(worker_args) = args.device_worker {
        return device_worker_main(worker_args);
    }

    // ... normal main process
}

/// Worker subprocess main loop (works for ANY SoapySDR device)
fn device_worker_main(args: WorkerArgs) -> Result<()> {
    let ctl_path = format!("/tmp/scanner-{}-ctl.sock", args.device_id);
    let dat_path = format!("/tmp/scanner-{}-dat.sock", args.device_id);

    // Cleanup stale sockets
    let _ = std::fs::remove_file(&ctl_path);
    let _ = std::fs::remove_file(&dat_path);

    // Create Unix listeners
    let ctl_listener = UnixListener::bind(&ctl_path)?;
    let dat_listener = UnixListener::bind(&dat_path)?;

    // Open SoapySDR device (ANY type: RTL-SDR, SDRplay, HackRF, etc.)
    let device = soapysdr::Device::new(&args.device_args)?;

    // Configure device
    device.set_sample_rate(soapysdr::Direction::Rx, 0, args.sample_rate)?;
    device.set_frequency(soapysdr::Direction::Rx, 0, args.frequency, "")?;
    device.set_gain(soapysdr::Direction::Rx, 0, args.gain)?;

    // Accept control connection
    let (mut ctl_stream, _) = ctl_listener.accept()?;

    // Send Ready message
    send_control_msg(&mut ctl_stream, ControlMessage::Ready)?;

    // Accept data connection
    let (mut dat_stream, _) = dat_listener.accept()?;

    // Activate stream
    let mut stream = device.rx_stream::<Complex<f32>>(&[0])?;
    stream.activate(None)?;

    let mut sequence = 0u64;
    let mut running = true;

    // Main worker loop
    while running {
        // Check for control messages (non-blocking)
        if let Ok(msg) = try_recv_control_msg(&mut ctl_stream) {
            match msg {
                ControlMessage::Tune { freq_hz } => {
                    device.set_frequency(soapysdr::Direction::Rx, 0, freq_hz, "")?;
                    let actual = device.get_frequency(soapysdr::Direction::Rx, 0)?;
                    send_control_msg(&mut ctl_stream,
                        ControlMessage::Tuned { actual_freq: actual })?;
                }
                ControlMessage::SetGain { gain_db } => {
                    device.set_gain(soapysdr::Direction::Rx, 0, gain_db)?;
                    let actual = device.get_gain(soapysdr::Direction::Rx, 0)?;
                    send_control_msg(&mut ctl_stream,
                        ControlMessage::GainSet { actual_gain: actual })?;
                }
                ControlMessage::Shutdown => {
                    running = false;
                }
                _ => {}
            }
        }

        // Read I/Q samples from device
        let mut samples = vec![Complex::new(0.0, 0.0); 1024];
        match stream.read(&mut samples, 1_000_000) {
            Ok(n) => {
                samples.truncate(n);

                // Send to main process via data socket
                let packet = IQPacket {
                    samples,
                    timestamp: timestamp_now(),
                    sequence,
                };

                if let Err(e) = bincode::serialize_into(&mut dat_stream, &packet) {
                    eprintln!("Failed to send I/Q packet: {}", e);
                    break;
                }

                sequence += 1;
            }
            Err(e) => {
                eprintln!("Device read error: {}", e);
                break;
            }
        }
    }

    // Cleanup
    stream.deactivate(None)?;
    std::fs::remove_file(&ctl_path)?;
    std::fs::remove_file(&dat_path)?;

    Ok(())
}

fn send_control_msg(stream: &mut UnixStream, msg: ControlMessage) -> Result<()> {
    bincode::serialize_into(stream, &msg)?;
    Ok(())
}

fn try_recv_control_msg(stream: &mut UnixStream) -> Result<ControlMessage> {
    stream.set_nonblocking(true)?;
    match bincode::deserialize_from(stream) {
        Ok(msg) => Ok(msg),
        Err(_) => Err(ScannerError::Custom("No message".into())),
    }
}

fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}
```

### Main Process Device Wrapper

```rust
/// Generic subprocess wrapper for ANY SoapySDR device
pub struct Device {
    device_id: sdr::DeviceId,
    worker_process: Child,
    control_stream: UnixStream,
    data_stream: UnixStream,
    socket_paths: (PathBuf, PathBuf),
    expected_sequence: u64,
}

impl Device {
    pub fn new(
        device_id: sdr::DeviceId,
        device_args: String,
        sample_rate: f64,
        frequency: f64,
        gain: f64,
    ) -> Result<Self> {
        let ctl_path = PathBuf::from(format!("/tmp/scanner-{}-ctl.sock", device_id.0));
        let dat_path = PathBuf::from(format!("/tmp/scanner-{}-dat.sock", device_id.0));

        // Spawn worker subprocess (works for ANY device type)
        let worker = Command::new(env::current_exe()?)
            .arg("--device-worker")
            .arg(&device_id.0)
            .arg(&device_args)
            .arg("--sample-rate")
            .arg(sample_rate.to_string())
            .arg("--frequency")
            .arg(frequency.to_string())
            .arg("--gain")
            .arg(gain.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Wait for sockets to appear (worker creates them)
        let timeout = Duration::from_secs(5);
        let start = Instant::now();

        while !ctl_path.exists() || !dat_path.exists() {
            if start.elapsed() > timeout {
                return Err(ScannerError::Custom(
                    "Worker subprocess failed to create sockets".into()
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // Connect to Unix sockets
        let control_stream = UnixStream::connect(&ctl_path)?;
        let data_stream = UnixStream::connect(&dat_path)?;

        // Set data socket to non-blocking for polling
        data_stream.set_nonblocking(true)?;

        // Wait for Ready message
        let mut device = Self {
            device_id,
            worker_process: worker,
            control_stream,
            data_stream,
            socket_paths: (ctl_path, dat_path),
            expected_sequence: 0,
        };

        match device.recv_control()? {
            ControlMessage::Ready => Ok(device),
            msg => Err(ScannerError::Custom(
                format!("Unexpected response: {:?}", msg)
            )),
        }
    }

    pub fn tune(&mut self, freq_hz: f64) -> Result<f64> {
        self.send_control(ControlMessage::Tune { freq_hz })?;

        match self.recv_control()? {
            ControlMessage::Tuned { actual_freq } => Ok(actual_freq),
            ControlMessage::Error { msg } => Err(ScannerError::DeviceError(msg)),
            msg => Err(ScannerError::Custom(format!("Unexpected response: {:?}", msg))),
        }
    }

    pub fn set_gain(&mut self, gain_db: f64) -> Result<f64> {
        self.send_control(ControlMessage::SetGain { gain_db })?;

        match self.recv_control()? {
            ControlMessage::GainSet { actual_gain } => Ok(actual_gain),
            ControlMessage::Error { msg } => Err(ScannerError::DeviceError(msg)),
            msg => Err(ScannerError::Custom(format!("Unexpected response: {:?}", msg))),
        }
    }

    pub fn read_samples(&mut self) -> Result<Vec<Complex<f32>>> {
        // Read from data socket
        match bincode::deserialize_from(&mut self.data_stream) {
            Ok(packet) => {
                let IQPacket { samples, sequence, .. } = packet;

                // Check for dropped packets
                if sequence != self.expected_sequence {
                    debug!(
                        expected = self.expected_sequence,
                        received = sequence,
                        "Dropped I/Q packets detected"
                    );
                }

                self.expected_sequence = sequence + 1;
                Ok(samples)
            }
            Err(e) => {
                // Non-blocking socket might not have data yet
                Err(ScannerError::from(e))
            }
        }
    }

    fn send_control(&mut self, msg: ControlMessage) -> Result<()> {
        bincode::serialize_into(&mut self.control_stream, &msg)?;
        Ok(())
    }

    fn recv_control(&mut self) -> Result<ControlMessage> {
        Ok(bincode::deserialize_from(&mut self.control_stream)?)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        debug!(device_id = ?self.device_id, "Shutting down worker subprocess");

        // Send shutdown command
        let _ = self.send_control(ControlMessage::Shutdown);

        // Give worker time to cleanup
        std::thread::sleep(Duration::from_millis(100));

        // Kill if still running
        let _ = self.worker_process.kill();
        let _ = self.worker_process.wait();

        // Cleanup socket files
        let _ = std::fs::remove_file(&self.socket_paths.0);
        let _ = std::fs::remove_file(&self.socket_paths.1);

        debug!(device_id = ?self.device_id, "Worker subprocess shut down");
    }
}
```

### Integration with Pool

```rust
// In pool::Pool::acquire()
impl pool::Pool {
    pub fn acquire(&self, requirements: &pool::TaskRequirements) -> Result<pool::PooledDevice> {
        let mut inner = self.pool_ref.lock().unwrap();

        // Find matching device
        let (device_id, entry) = /* ... find best match ... */;

        // Spawn subprocess for device (universal - works for all types)
        let subprocess = ipc::Device::new(
            device_id.clone(),
            entry.device_args.clone(),
            requirements.required_sample_rate,
            requirements.frequency_hz,
            20.0,  // Default gain
        )?;

        // Wrap in pool::PooledDevice (RAII auto-return)
        Ok(pool::PooledDevice {
            device: Some(Box::new(subprocess)),
            pool: Arc::clone(&self.pool_ref),
            device_id,
        })
    }
}
```

## Implementation Steps

### Step 1: Add Dependencies
**Time**: 15 minutes

```toml
# Cargo.toml
[dependencies]
bincode = "1.3"           # Binary serialization for IPC
serde = { version = "1", features = ["derive"] }
```

### Step 2: Create IPC Protocol Module
**Time**: 1 hour

1. Create `src/ipc/mod.rs`
2. Create `src/ipc/protocol.rs` - Message types
3. Define `ControlMessage` enum
4. Define `IQPacket` struct
5. Add serialization helpers

### Step 3: Implement Worker Subprocess
**Time**: 3 hours

1. Add `--device-worker` arg parsing to `bin/scanner.rs`
2. Implement `device_worker_main()`
3. Socket creation and cleanup
4. Control message handling
5. I/Q streaming loop
6. Error handling and logging

### Step 4: Implement Main Process Wrapper
**Time**: 2 hours

1. Create `ipc::Device` struct
2. Implement `new()` - spawn and connect
3. Implement `tune()`, `set_gain()` control methods
4. Implement `read_samples()` data method
5. Implement `Drop` for cleanup

### Step 5: Integration with Pool
**Time**: 1 hour

1. Update `pool::Pool::acquire()` to spawn subprocess
2. Remove any device-type branching (use subprocess for all)
3. Verify RAII cleanup works with subprocess

### Step 6: Testing
**Time**: 2 hours

```rust
#[test]
fn test_subprocess_device_single() {
    let device = ipc::Device::new(
        sdr::DeviceId::from_serial("test", "001"),
        "driver=rtlsdr".into(),
        2e6,
        88.9e6,
        20.0,
    ).unwrap();

    // Should be able to read samples
    let samples = device.read_samples().unwrap();
    assert!(!samples.is_empty());
}

#[test]
fn test_subprocess_device_multiple_same_type() {
    // Test SDRplay isolation (two devices, same type)
    let device1 = ipc::Device::new(
        sdr::DeviceId::from_serial("sdrplay", "001"),
        "driver=sdrplay,serial=001".into(),
        2e6,
        88.9e6,
        20.0,
    ).unwrap();

    let device2 = ipc::Device::new(
        sdr::DeviceId::from_serial("sdrplay", "002"),
        "driver=sdrplay,serial=002".into(),
        2e6,
        162.5e6,
        20.0,
    ).unwrap();

    // Both should work independently (validates SDRplay isolation)
    let samples1 = device1.read_samples().unwrap();
    let samples2 = device2.read_samples().unwrap();

    assert!(!samples1.is_empty());
    assert!(!samples2.is_empty());
}

#[test]
fn test_subprocess_cleanup() {
    let device = ipc::Device::new(
        sdr::DeviceId::from_serial("test", "001"),
        "driver=rtlsdr".into(),
        2e6,
        88.9e6,
        20.0,
    ).unwrap();

    let pid = device.worker_process.id();
    let sockets = device.socket_paths.clone();

    drop(device);

    // Wait for cleanup
    std::thread::sleep(Duration::from_millis(200));

    // Verify process killed
    assert!(is_process_dead(pid));

    // Verify sockets removed
    assert!(!sockets.0.exists());
    assert!(!sockets.1.exists());
}

#[test]
fn test_subprocess_crash_isolation() {
    let device = ipc::Device::new(
        sdr::DeviceId::from_serial("test", "001"),
        "driver=nonexistent".into(),  // Will fail
        2e6,
        88.9e6,
        20.0,
    );

    // Should fail to create, but not crash main process
    assert!(device.is_err());

    // Main process still running
    assert!(std::process::id() > 0);
}

fn is_process_dead(pid: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|s| !s.success())
        .unwrap_or(true)
}
```

## Performance Characteristics

### Latency
- **Socket overhead**: ~25μs per packet
- **Serialization**: ~5μs per packet (bincode)
- **Total overhead**: ~30μs (negligible vs typical SDR sample rates)

### Throughput
- **Unix sockets**: ~100 Gbps on localhost
- **Typical SDR**: 2 MSPS × 8 bytes/sample = 16 MB/s = 0.128 Gbps
- **Headroom**: 780x more bandwidth than needed

### Memory
- **Per subprocess**: 10-20 MB overhead
- **3 devices**: ~60 MB total (acceptable on modern systems)

### CPU
- **Serialization**: <1% CPU per device
- **Context switching**: Minimal (Unix sockets are efficient)

## Benefits

### Architectural
✅ **Simplicity**: One code path for all device types
✅ **No branching**: No `if device_type == X` logic
✅ **Easy to reason**: All devices work the same way

### Reliability
✅ **Crash isolation**: Driver bug doesn't kill main process
✅ **Memory isolation**: Leaks contained to subprocess
✅ **Easy debugging**: Attach to specific device's process

### Performance
✅ **Low latency**: ~30μs overhead (negligible)
✅ **High throughput**: Unix sockets handle 100+ Gbps
✅ **Efficient**: bincode serialization is fast

### No Dependencies
✅ **No SoapySDRServer**: We own the IPC code
✅ **No TCP overhead**: Unix sockets are faster
✅ **No port conflicts**: Socket files in /tmp

## Error Handling

### Worker Subprocess Failures
```rust
impl Device {
    pub fn new(...) -> Result<Self> {
        let worker = Command::new(...)
            .stderr(Stdio::piped())
            .spawn()?;

        // Monitor stderr for errors
        let stderr = worker.stderr.take().unwrap();
        let error_reader = std::io::BufReader::new(stderr);

        // If Ready not received within timeout, capture stderr
        match wait_for_ready(&mut control_stream, Duration::from_secs(5)) {
            Ok(_) => Ok(device),
            Err(e) => {
                // Read stderr to get actual error
                let mut errors = String::new();
                for line in error_reader.lines() {
                    errors.push_str(&line.unwrap());
                }

                worker.kill()?;
                Err(ScannerError::Custom(format!(
                    "Worker failed: {} (stderr: {})", e, errors
                )))
            }
        }
    }
}
```

### Device Disconnection
```rust
impl Device {
    pub fn read_samples(&mut self) -> Result<Vec<Complex<f32>>> {
        match bincode::deserialize_from(&mut self.data_stream) {
            Ok(packet) => Ok(packet.samples),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data yet (non-blocking socket)
                Ok(vec![])
            }
            Err(e) => {
                // Connection lost or device error
                debug!("Device connection lost: {}", e);
                Err(ScannerError::DeviceDisconnected(self.device_id.clone()))
            }
        }
    }
}
```

## File Structure

```
src/
  ipc/
    mod.rs              # Module exports
    protocol.rs         # ControlMessage, IQPacket
    worker.rs          # Worker subprocess implementation
    device.rs          # Device wrapper

bin/
  scanner.rs           # Main entry point + worker mode
```

## Estimated Time

**Total**: 9-10 hours

- Step 1: Dependencies (15 min)
- Step 2: IPC protocol module (1 hr)
- Step 3: Worker subprocess (3 hrs)
- Step 4: Main process wrapper (2 hrs)
- Step 5: Pool integration (1 hr)
- Step 6: Testing (2 hrs)

## Success Criteria

✅ Worker subprocess spawns successfully
✅ Control commands work (tune, gain)
✅ I/Q data streaming works
✅ Multiple devices work simultaneously
✅ SDRplay isolation verified (two SDRplay devices)
✅ Crash isolation verified
✅ Cleanup verified (process killed, sockets removed)
✅ Performance acceptable (<50μs latency)

## Next Steps

After completing this plan:
1. **Plan 009**: Task Abstraction (tasks use subprocess devices)
2. **Plan 010**: Multi-SDR Orchestration (ties everything together)
