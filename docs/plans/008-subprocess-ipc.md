# Plan 008: Universal Subprocess IPC

## Guidance for Updates

When updating this plan as work progresses, avoid adding:
- Lists of accomplishments or completion summaries
- Self-aggrandizement or subjective quality assessments
- Rationales and benefits sections (unless specifically requested)
- Speculation about future improvements or possibilities
- Time estimates or risk assessments

Keep updates matter-of-fact and focused on concrete technical details. Check off completed tasks and add technical notes as needed. For each section, add a brief summary of anything that you learned from internet searches.

## Context

Run every SDR device in isolated subprocess with custom Unix socket IPC. This plan addresses multiple device isolation requirements and provides a universal architecture that works for all SDR types.

### Problem Statement

**SDRplay Global Handle**: The mirsdrapi-rsp.h driver uses an internal global handle rather than returning a proper handle for each device. Cannot support multiple open devices in single process. Industry workaround: spawn SoapySDRServer per device (process isolation).

**Terminal Corruption**: RTL-SDR prints "Reattached kernel driver" to stderr during enumeration, corrupting TUI display. Modern Linux (Ubuntu 16.04+) has udev rules but messages still appear during hotplug.

**Device Enumeration Output**: Opening devices to query capabilities prints messages to terminal. During discovery scan, these messages corrupt the TUI interface.

**Driver Crashes**: Driver bugs in any SDR can crash the main process. LimeSDR stability issues documented, but any device can have driver problems.

### Research Findings

**Unix Domain Socket Performance** (verified via benchmark studies):
- Latency: 2-6μs per message (vs 6μs for TCP loopback)
- Throughput: 7x faster than TCP, 1.73M msg/s vs 0.25M msg/s
- CPU efficiency: Near-zero CPU with DMA controllers

**Serialization Performance** (rust_serialization_benchmark):
- postcard: 60ns serialize, 180ns deserialize, ~70% message size of bincode
- Built for embedded systems, excellent for small I/Q packets
- bincode: 547μs serialize (alternative if postcard has issues)
- rkyv: 384μs read, zero-copy (alternative for larger payloads)

**Process Cleanup** (Rust std::process documentation):
- Critical: Always call wait() after kill() to prevent zombie processes
- Zombies exhaust system resources (PIDs) in long-running applications
- Tokio's kill() includes automatic wait(), std's does not

**BladeRF Thread Safety** (libbladeRF documentation):
- API is explicitly thread-safe with per-stream internal locks
- Designed for multi-threaded concurrent access
- No documented requirement for process isolation

**SoapySDR Multi-Channel** (RSPduo documentation):
- Supports independent frequency control per channel
- API includes channel parameter: `setFrequency(Direction::Rx, channel, freq)`
- Single device handle can manage multiple channels
- RSPduo provides 2 independent RX channels with separate tuning

**SoapySDRServer** (existing remote solution):
- Uses TCP + UDP, spawns threads per device
- User reports: 40% CPU, 6.5Mbps at 3.2MHz, negligible latency
- Battle-tested by CubicSDR and other applications
- We're choosing custom IPC for: simpler deployment, Unix socket performance, full control

### Architecture Overview

**Two Subprocess Types**:

1. **Enumeration Subprocess (Discovery)**: Short-lived (seconds), spawned during device scan, isolates terminal output from device enumeration, returns device list via stdout, then exits

2. **Device Worker Subprocess (Pool)**: Long-lived (minutes/hours), spawned on first tuner allocation for a device, one subprocess per device (not per tuner), manages all channels/tuners for that device, streams I/Q data via Unix sockets

**Key Architectural Points**:
- Per-Device, Not Per-Tuner: Multi-tuner devices (RSPduo with 2 tuners) share one subprocess. Subprocess manages multiple channels via SoapySDR API. IPC protocol includes channel/tuner routing.
- Lazy Spawning: Discovery returns metadata only (no subprocess). Pool stores inventory (no subprocess). First `acquire()` spawns subprocess. Subsequent allocations reuse subprocess.
- Subprocess Lifetime: Lives until device removed from pool or shutdown. Independent of individual tuner allocations. Dropping tuner does NOT terminate subprocess.

### System Architecture

```
Main Process
├─ DiscoveryService
│   └─ enumerate_devices_subprocess()  ← SHORT-LIVED
│       ├─ Spawns: scanner worker enumerate
│       ├─ Discards stderr (kernel messages)
│       ├─ Returns: Vec<DeviceInfo> via stdout JSON
│       └─ Exits immediately
│
├─ Pool
│   ├─ devices: HashMap<DeviceId, DeviceEntry>  (metadata only)
│   ├─ available_tuners: HashMap<TunerId, TunerEntry>
│   ├─ allocated_tuners: HashMap<TunerId, AllocationInfo>
│   └─ subprocesses: HashMap<DeviceId, Arc<SubprocessHandle>>  ← LONG-LIVED
│
└─ Pool::acquire(requirements)
    └─ get_or_spawn_subprocess(device_id)
        ├─ If exists: reuse subprocess
        ├─ If not: spawn "scanner worker device" subprocess
        └─ start_stream(channel)
```

### Subprocess Lifecycle

```
Phase 0: Discovery
─────────────────
DiscoveryService::scan_devices()
  → spawn enumeration subprocess
  → Returns Vec<DeviceInfo>
  → Subprocess exits
  → NO long-lived subprocess

Phase 1: Pool Registration
───────────────────────────
Pool::add_device(DeviceInfo)
  → Stores device metadata
  → Creates TunerEntry for each channel
  → NO subprocess spawned

Phase 2: First Allocation
──────────────────────────
Pool::acquire(requirements)
  → Finds matching tuner
  → get_or_spawn_subprocess(device_id)
      ├─ Check: subprocess exists?
      ├─ NO: Spawn device-worker subprocess  ← CREATED HERE
      │   └─ Opens SoapySDR device
      │   └─ Creates Unix sockets
      │   └─ Sends Ready message
      └─ YES: Reuse existing subprocess
  → subprocess.start_stream(channel)
  → Return Tuner wrapper

Phase 3: Second Allocation (Same Device)
─────────────────────────────────────────
Pool::acquire(requirements)
  → Finds different tuner on same device
  → get_or_spawn_subprocess(device_id)
      └─ Reuses existing subprocess  ← NOT SPAWNED
  → subprocess.start_stream(different_channel)
  → Return Tuner wrapper

Phase 4: Tuner Release
───────────────────────
Tuner::drop()
  → subprocess.stop_stream(channel)
  → return_tuner_to_pool()
  → Subprocess continues running  ← NOT TERMINATED

Phase 5: Pool Shutdown
──────────────────────
Pool::shutdown()
  → For each subprocess:
      → Send Shutdown message
      → Graceful wait (500ms)
      → SIGTERM if needed (500ms)
      → SIGKILL if needed
      → wait() to reap zombie
  → Clear subprocesses HashMap
```

## Proposal 1: IPC Protocol Foundation

Create message types and serialization infrastructure for subprocess communication.

**Status**: ✅ Complete

### Implementation Notes

Implemented trait-based design to enable multiple implementations (Unix sockets, TCP/IP) and facilitate testing:
- **Traits**: `ControlChannel`, `DataReceiver`, `DataSender` define interfaces
- **Unix Implementation**: `UnixControlChannel`, `UnixDataReceiver`, `UnixDataSender` with RAII cleanup
- **Mock Implementation**: `MockControlChannel`, `MockDataReceiver`, `MockDataSender` for testing
- **Atomic Commands**: `ConfigureAndStart`, `StopStream`, `Shutdown` (no fine-grained operations)
- **Structured Responses**: `Ready`, `StreamStarted`, `StreamStopped`, `Error` with actual hardware values
- **Directional Data Channels**: Worker uses `DataSender`, main uses `DataReceiver` (unidirectional I/Q flow)

### Tasks

- [x] Add dependencies: `postcard` (with `alloc` feature), enable `serde` feature for `num` crate
- [x] Create `src/ipc/` module with `traits.rs`, `protocol.rs`, `mock.rs`
- [x] Define `ControlMessage` enum with atomic operations
- [x] Define `IQPacket` struct for sample data
- [x] Implement Unix socket channels with RAII cleanup
- [x] Implement mock channels for testing
- [x] Add IPC error types to `ScannerError`
- [x] Run `make lint` (passing)

### IPC Protocol

```rust
/// Control messages (bidirectional Unix socket)
#[derive(Serialize, Deserialize, Debug)]
pub enum ControlMessage {
    // Main → Worker commands
    Tune { channel: usize, freq_hz: f64 },
    SetGain { channel: usize, gain_db: f64 },
    SetSampleRate { channel: usize, rate: f64 },
    StartStream { channel: usize },
    StopStream { channel: usize },
    Shutdown,

    // Worker → Main responses
    Ready,
    Tuned { channel: usize, actual_freq: f64 },
    GainSet { channel: usize, actual_gain: f64 },
    Error { msg: String },
}

/// I/Q data packet (unidirectional Unix socket)
#[derive(Serialize, Deserialize)]
pub struct IQPacket {
    pub channel: usize,           // Which tuner/channel
    pub samples: Vec<Complex<f32>>,
    pub timestamp: u64,
    pub sequence: u64,            // For detecting drops
}
```

## Proposal 2: Worker Command Structure

Add hidden `worker` subcommands to scanner binary for internal subprocess operations.

**Status**: ✅ Complete

### Implementation Notes

Implemented backend-specific enumeration to maintain flexibility and allow parallel enumeration of multiple backends:
- **Backend parameter**: `Enumerate { backend: String }` allows per-backend subprocess isolation
- **Worker module**: Created `src/cli/worker.rs` with stub handlers
- **Hidden command**: Uses `#[command(hide = true, subcommand)]` to hide from help
- **Main routing**: Added `Commands::Worker(WorkerCommand)` with nested match on variants

Process safety research (see `docs/research/2025-10-process-safety.md`):
- Multiple processes can safely enumerate concurrently (process memory isolation)
- Safe to enumerate in one subprocess while streaming in another (no shared state)
- SDRplay driver limitations avoided by per-device subprocess architecture

### Tasks

- [x] Add `worker` command to args.rs with `#[command(hide = true, subcommand)]`
- [x] Create `WorkerCommand` enum with `Enumerate { backend }` and `Device` variants
- [x] Create `src/cli/worker.rs` with stub handlers
- [x] Add command routing in bin/scanner.rs to worker handlers
- [x] Export from cli module
- [x] Test worker commands (enumerate and device return "not yet implemented")
- [x] Verify hidden from help (`scanner --help` doesn't show worker)
- [x] Run `make lint` (passing)

### Binary Modes

The scanner binary supports multiple operational modes. Subprocess workers use dedicated commands separate from user-facing commands like `scan` and `train`.

```rust
// bin/scanner.rs
#[derive(Parser)]
enum Command {
    /// Normal scanning mode (user-facing)
    Scan { /* scan args */ },

    /// Training mode for ML models (user-facing)
    Train { /* train args */ },

    /// Internal: Enumeration worker subprocess
    /// Not intended for direct user invocation
    #[command(hide = true)]
    Worker(WorkerCommand),
}

#[derive(Subcommand)]
enum WorkerCommand {
    /// Enumerate devices for a specific backend (short-lived subprocess)
    Enumerate {
        backend: String,  // "soapy", "seify", "rtlsdr"
    },

    /// Stream I/Q from specific device (long-lived subprocess)
    Device {
        device_id: String,
        device_args: String,
        sample_rate: f64,
        // ... other args
    },
}

fn main() -> Result<()> {
    match args.command {
        Command::Scan { .. } => scanner_main(args),
        Command::Train { .. } => train_main(args),
        Command::Worker(WorkerCommand::Enumerate { backend }) => {
            enumeration_worker_main(&backend)
        }
        Command::Worker(WorkerCommand::Device { .. }) => device_worker_main(args),
    }
}
```

**Command Usage**:
- `scanner scan --band fm` - User-facing scan command
- `scanner train --model-type heuristic3` - User-facing training command
- `scanner worker enumerate --backend soapy` - Internal subprocess (spawned by DiscoveryService)
- `scanner worker device --device-id X ...` - Internal subprocess (spawned by Pool)

## Proposal 3: Enumeration Worker Implementation

Implement short-lived subprocess that isolates terminal output during device enumeration.

**Status**: ✅ Complete

### Implementation Notes

Implemented Unix socket-based enumeration worker using the IPC protocol from Proposal 1:
- **Unix Socket Communication**: Uses `UnixControlChannel` and `ControlMessage` protocol (not JSON to stdout)
- **Backend Selection**: `backend_from_name()` helper maps backend names to implementations
- **Command Arguments**: `socket_path` and `log_file` passed as CLI arguments (not env vars)
- **RAII Cleanup**: Socket cleaned up automatically via `UnixControlChannel::with_cleanup()`
- **Error Handling**: Backend and enumeration errors sent via `ControlMessage::Error`
- **Structured Logging**: Context fields include backend name, PID, socket path

Actual implementation uses same IPC protocol as device workers for consistency.

### Tasks

- [x] Add `socket_path` and `log_file` arguments to `Enumerate` command in args.rs
- [x] Implement `backend_from_name()` helper function in worker.rs
- [x] Implement `handle_enumerate_command()` function in worker.rs
- [x] Call `backend.enumerate_devices()` and send via Unix socket
- [x] Send device list as `ControlMessage::DeviceList`
- [x] Add optional file-based logging via `--log-file` argument
- [x] Test with mock backend (verified 2 devices returned)
- [x] Test error handling (invalid backend sends error message)
- [x] Verify terminal messages don't corrupt socket communication
- [x] Run `make lint` (passing)

### Enumeration Worker (Short-lived)

```rust
// src/cli/worker.rs
fn backend_from_name(name: &str) -> Result<Box<dyn Backend>> {
    match name {
        "soapy" => Ok(Box::new(Soapy)),
        "mock" => Ok(Box::new(Mock)),
        other => Err(ScannerError::Custom(format!("Unknown backend: {}", other))),
    }
}

pub fn handle_enumerate_command(
    backend_name: &str,
    socket_path: &str,
    log_file: Option<&str>,
) -> Result<()> {
    // Set up optional logging to file (append mode)
    if let Some(log_path) = log_file {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        tracing_subscriber::fmt()
            .with_writer(file.with_max_level(tracing::Level::DEBUG))
            .with_ansi(false)
            .init();
    }

    debug!(
        backend = backend_name,
        pid = %std::process::id(),
        socket_path = socket_path,
        "Enumeration worker starting"
    );

    // Bind socket and wait for parent
    let listener = UnixListener::bind(socket_path)?;
    let (stream, _) = listener.accept()?;
    let mut channel = UnixControlChannel::with_cleanup(stream, socket_path.into());

    // Enumerate devices (terminal output goes to stderr, discarded by parent)
    let result = backend_from_name(backend_name)
        .and_then(|backend| backend.enumerate_devices());

    // Send result via socket
    match result {
        Ok(devices) => {
            channel.send(&ControlMessage::DeviceList { devices })?;
        }
        Err(e) => {
            channel.send(&ControlMessage::Error {
                channel: None,
                message: e.to_string(),
            })?;
        }
    }

    // Exit - socket cleaned up via RAII
    Ok(())
}
```

## Proposal 4: Discovery Service Integration

Update DiscoveryService to spawn enumeration subprocess instead of direct enumeration.

**Status**: ✅ Complete

### Implementation Notes

Implemented subprocess-based device enumeration using the IPC protocol from Proposals 1 and 3:
- **SubprocessEnumerator**: New enumerator that spawns worker subprocesses
- **DirectEnumerator**: Renamed from BackendEnumerator, used only for testing
- **Backend names**: Production uses strings ("soapy", "mock") instead of trait objects
- **Unix socket communication**: Uses ControlMessage protocol (not JSON/stdout as originally planned)
- **Process isolation**: Terminal output from drivers isolated in subprocesses
- **Testing**: Tests use DirectEnumerator (no subprocess overhead), production uses SubprocessEnumerator

### Tasks

- [x] Rename BackendEnumerator to DirectEnumerator
- [x] Create SubprocessEnumerator with Unix socket IPC
- [x] Update discovery::create() to accept Vec<String> backend names
- [x] Map backend names to SubprocessEnumerator instances
- [x] Update CLI integration to pass backend names
- [x] Add debug logging for subprocess enumeration
- [x] Set stdin/stdout/stderr to Stdio::null() to discard output
- [x] Handle subprocess spawn failures and timeouts
- [x] Test with mock backend (manual verification)
- [x] Run make lint and make test (all pass)

### SubprocessEnumerator Implementation

```rust
// src/discovery/enumerator.rs
pub struct SubprocessEnumerator {
    backend_name: String,
}

impl SubprocessEnumerator {
    pub fn new(backend_name: String) -> Self {
        Self { backend_name }
    }

    fn spawn_and_enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let socket_path = format!(
            "/tmp/scanner-enum-{}-{}-{}.sock",
            self.backend_name,
            std::process::id(),
            timestamp
        );

        let mut cmd = Command::new(env::current_exe()?);
        cmd.arg("worker")
            .arg("enumerate")
            .arg("--backend")
            .arg(&self.backend_name)
            .arg("--socket-path")
            .arg(&socket_path);

        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = cmd.spawn()?;

        // Wait for socket with 5 second timeout
        let start = Instant::now();
        while !Path::new(&socket_path).exists() {
            if start.elapsed() > Duration::from_secs(5) {
                let _ = child.kill();
                return Err("Socket creation timeout".into());
            }
            thread::sleep(Duration::from_millis(10));
        }

        // Connect and receive device list
        let stream = UnixStream::connect(&socket_path)?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut channel = UnixControlChannel::new(stream);

        match channel.recv()? {
            ControlMessage::DeviceList { devices } => {
                let _ = child.wait();
                Ok(devices)
            }
            ControlMessage::Error { message, .. } => {
                let _ = child.wait();
                Err(message.into())
            }
            _ => Err("Unexpected message".into())
        }
    }
}

impl DeviceEnumerator for SubprocessEnumerator {
    fn enumerate(&self) -> Result<Vec<hardware::DeviceInfo>, Box<dyn std::error::Error>> {
        self.spawn_and_enumerate()
    }

    fn name(&self) -> &str {
        &self.backend_name
    }
}
```

### Discovery Module Updates

```rust
// src/discovery/mod.rs - Production
pub fn create(backend_names: Vec<String>, mode: DiscoveryMode) -> Box<dyn Service> {
    let mut enumerators = backend_names
        .into_iter()
        .map(|name| {
            (
                Box::new(SubprocessEnumerator::new(name)) as Box<dyn DeviceEnumerator>,
                SourcePriority::Backend,
            )
        })
        .collect();
    // ... rest unchanged
}

// Testing - uses DirectEnumerator
pub fn create_for_testing(backends: Vec<Box<dyn Backend>>, mode: DiscoveryMode) -> Box<dyn Service> {
    let enumerators = vec![(
        Box::new(DirectEnumerator { backends }),
        SourcePriority::Backend,
    )];
    // ... rest unchanged
}
```

### CLI Integration

```rust
// src/cli/discovery.rs
pub fn start_discovery_service(...) -> Result<DiscoverySetup> {
    let backend_names = vec!["soapy".to_string()];
    let mut discovery_service = discovery::create(backend_names, DiscoveryMode::Auto);
    // ... rest unchanged
}
```

## Proposal 5: Device Worker Implementation

Implement long-lived subprocess that manages one device with multiple channels and streams I/Q data.

**Status**: ✅ Complete

### Implementation Notes

Implemented StreamingDevice trait and device worker subprocess following the architecture from tmp/ipc.md:
- **StreamingDevice trait**: Separate interface for direct sample streaming (src/hardware/streaming.rs)
- **Backend support**: Both Soapy and Mock backends implement `open_streaming_device()`
- **Device worker**: Full implementation in `handle_device_command()` with main loop
- **Multi-channel support**: Single subprocess manages all channels for a device (e.g., RSPduo with 2 channels)
- **Backpressure handling**: Drops samples gracefully when parent can't keep up
- **Atomic commands**: ConfigureAndStart (combines tune + gain + start), StopStream, Shutdown
- **IQPacket structure**: Includes channel, samples, timestamp, sequence number

### Tasks

- [x] Create StreamingDevice trait with configure_rx, start_stream, read_samples, stop_stream
- [x] Add DeviceId::backend() helper method
- [x] Update Backend trait with open_streaming_device()
- [x] Implement SoapyStreamingDevice
- [x] Implement MockStreamingDevice
- [x] Update Device command args with device_id_str, control/data socket paths, log_file
- [x] Implement `handle_device_command()` function
- [x] Create Unix socket listeners for control and data
- [x] Open streaming device via backend and wait for parent connection
- [x] Send Ready message after connection
- [x] Implement main loop with control message handling
- [x] Implement multi-channel stream management (HashMap of active streams)
- [x] Handle ConfigureAndStart, StopStream, Shutdown messages
- [x] Send I/Q packets with channel tags via data socket
- [x] Add graceful cleanup on shutdown (stop all streams)
- [x] Add file-based logging via --log-file argument
- [x] Update bin/scanner.rs routing for Device command
- [x] Run `make lint` (passing)

### Device Worker (Long-lived)

```rust
/// Subprocess that manages one device with multiple channels
fn device_worker_main(args: DeviceWorkerArgs) -> Result<()> {
    // Set up logging with device ID in filename
    if let Ok(log_dir) = std::env::var("SCANNER_WORKER_LOG_DIR") {
        let log_path = format!("{}/worker-{}.log", log_dir, args.device_id);
        tracing_subscriber::fmt()
            .with_writer(std::fs::File::create(log_path)?)
            .with_ansi(false)
            .init();
    }

    debug!(device_id = %args.device_id, "Device worker starting");

    let ctl_path = format!("/tmp/scanner-{}-ctl.sock", args.device_id);
    let dat_path = format!("/tmp/scanner-{}-dat.sock", args.device_id);

    // Cleanup stale sockets
    let _ = std::fs::remove_file(&ctl_path);
    let _ = std::fs::remove_file(&dat_path);

    // Create Unix listeners
    let ctl_listener = UnixListener::bind(&ctl_path)?;
    let dat_listener = UnixListener::bind(&dat_path)?;

    // Open SoapySDR device (works for ANY type)
    let device = soapysdr::Device::new(&args.device_args)?;

    // Wait for parent to connect
    let (mut ctl_stream, _) = ctl_listener.accept()?;
    let (mut dat_stream, _) = dat_listener.accept()?;

    // Send Ready message
    send_control_msg(&mut ctl_stream, ControlMessage::Ready)?;

    let mut active_streams: HashMap<usize, Stream> = HashMap::new();
    let mut running = true;

    // Main loop: manage multiple channels
    while running {
        // Check for control messages (non-blocking)
        match try_recv_control_msg(&mut ctl_stream) {
            Ok(ControlMessage::StartStream { channel }) => {
                let stream = device.rx_stream::<Complex<f32>>(&[channel])?;
                stream.activate(None)?;
                active_streams.insert(channel, stream);
            }
            Ok(ControlMessage::StopStream { channel }) => {
                if let Some(mut stream) = active_streams.remove(&channel) {
                    stream.deactivate(None)?;
                }
            }
            Ok(ControlMessage::Tune { channel, freq_hz }) => {
                device.set_frequency(Direction::Rx, channel, freq_hz, "")?;
                let actual = device.get_frequency(Direction::Rx, channel)?;
                send_control_msg(&mut ctl_stream,
                    ControlMessage::Tuned { channel, actual_freq: actual })?;
            }
            Ok(ControlMessage::Shutdown) => {
                running = false;
            }
            Err(_) => {} // No message
        }

        // Read samples from all active streams
        for (channel, stream) in active_streams.iter_mut() {
            let mut samples = vec![Complex::new(0.0, 0.0); 1024];
            match stream.read(&mut samples, 100_000) {
                Ok(n) => {
                    samples.truncate(n);

                    // Send to main process with channel tag
                    let packet = IQPacket {
                        channel: *channel,
                        samples,
                        timestamp: timestamp_now(),
                        sequence: get_sequence(*channel),
                    };

                    postcard::to_io(&packet, &mut dat_stream)?;
                }
                Err(e) => {
                    debug!(channel, error = ?e, "Stream read error");
                }
            }
        }
    }

    // Graceful cleanup
    for (channel, mut stream) in active_streams.drain() {
        let _ = stream.deactivate(None);
    }

    debug!(device_id = %args.device_id, "Device worker shutting down");

    std::fs::remove_file(&ctl_path)?;
    std::fs::remove_file(&dat_path)?;

    Ok(())
}
```

## Proposal 6: SubprocessHandle

Create handle type that manages device worker subprocess lifecycle and IPC communication.

### Tasks

- [ ] Create `SubprocessHandle` struct with process, sockets, channels tracking
- [ ] Implement `spawn()` with socket setup and connection
- [ ] Implement `start_stream(channel)` to send StartStream command
- [ ] Implement `stop_stream(channel)` to send StopStream command
- [ ] Implement `shutdown()` with timeout escalation (Shutdown → SIGTERM → SIGKILL)
- [ ] Ensure `wait()` always called to prevent zombie processes
- [ ] Implement `cleanup_sockets()` for socket file removal
- [ ] Add shutdown-safe behavior with try_lock
- [ ] Run `make lint` and `make test`

### SubprocessHandle (per device)

```rust
pub struct SubprocessHandle {
    device_id: hardware::DeviceId,
    process: Child,
    control_stream: Mutex<UnixStream>,
    active_channels: Mutex<HashSet<usize>>,
    shutdown_token: CancellationToken,
    socket_paths: (PathBuf, PathBuf),
}

impl SubprocessHandle {
    /// Start streaming on a channel
    pub fn start_stream(&self, channel: usize) -> Result<()> {
        if self.shutdown_token.is_cancelled() {
            return Err(ScannerError::PoolShutdown);
        }

        let mut stream = self.control_stream.lock()?;
        send_message(&mut *stream, ControlMessage::StartStream { channel })?;

        self.active_channels.lock()?.insert(channel);
        Ok(())
    }

    /// Stop streaming on a channel (called by Tuner::drop)
    pub fn stop_stream(&self, channel: usize) {
        if self.shutdown_token.is_cancelled() {
            return; // Subprocess shutting down anyway
        }

        if let Ok(mut stream) = self.control_stream.try_lock() {
            let _ = send_message(&mut *stream, ControlMessage::StopStream { channel });
            if let Ok(mut channels) = self.active_channels.try_lock() {
                channels.remove(&channel);
            }
        }
    }

    /// Graceful shutdown with timeout escalation
    pub fn shutdown(&mut self) -> Result<()> {
        debug!(device_id = ?self.device_id, "Shutting down device subprocess");

        // Step 1: Send graceful shutdown command
        if let Ok(mut stream) = self.control_stream.try_lock() {
            let _ = send_message(&mut *stream, ControlMessage::Shutdown);
        }

        // Step 2: Wait with timeout (500ms)
        let timeout = Duration::from_millis(500);
        let start = Instant::now();

        loop {
            match self.process.try_wait()? {
                Some(status) => {
                    debug!(device_id = ?self.device_id, ?status, "Subprocess exited gracefully");
                    self.cleanup_sockets();
                    return Ok(());
                }
                None if start.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                None => break, // Timeout - escalate
            }
        }

        // Step 3: Send SIGTERM (500ms timeout)
        debug!(device_id = ?self.device_id, "Graceful timeout, sending SIGTERM");
        #[cfg(unix)]
        {
            use nix::sys::signal::{Signal, kill};
            use nix::unistd::Pid;
            let pid = Pid::from_raw(self.process.id() as i32);
            let _ = kill(pid, Signal::SIGTERM);
        }

        let timeout = Duration::from_millis(500);
        let start = Instant::now();

        loop {
            match self.process.try_wait()? {
                Some(status) => {
                    debug!(device_id = ?self.device_id, ?status, "Subprocess exited after SIGTERM");
                    self.cleanup_sockets();
                    return Ok(());
                }
                None if start.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                None => break, // Timeout - force kill
            }
        }

        // Step 4: Force kill with SIGKILL
        debug!(device_id = ?self.device_id, "SIGTERM timeout, force killing");
        self.process.kill()?;

        // Step 5: CRITICAL - Always wait to reap zombie
        let status = self.process.wait()?;
        debug!(device_id = ?self.device_id, ?status, "Subprocess killed and reaped");

        self.cleanup_sockets();
        Ok(())
    }

    fn cleanup_sockets(&self) {
        let _ = std::fs::remove_file(&self.socket_paths.0);
        let _ = std::fs::remove_file(&self.socket_paths.1);
    }
}
```

### Shutdown Strategy

**Layered Shutdown**:

```
User Signal (Ctrl-C)
  ↓
ShutdownCoordinator::shutdown()
  → Cancel global token
  ↓
Pool::shutdown()
  → For each device subprocess:
      → Send Shutdown message (500ms timeout)
      → Send SIGTERM (500ms timeout)
      → Send SIGKILL (guaranteed)
      → wait() to reap zombie
  ↓
ShutdownCoordinator::wait()
  → Join all threads
  → Verify clean state
```

**Timeout-Based Escalation**: Each subprocess gets graceful shutdown with fallback:
1. Send Shutdown control message (500ms)
2. Send SIGTERM signal (500ms)
3. Send SIGKILL (guaranteed kill)
4. wait() to reap zombie (prevents resource leak)

**Non-Blocking Drop**: Tuner::drop must never block during shutdown:
```rust
impl Drop for Tuner {
    fn drop(&mut self) {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return; // Fast exit during shutdown
        }

        // Use try_lock to avoid blocking
        self.subprocess.stop_stream(self.channel);
    }
}
```

## Proposal 7: Pool Integration

Integrate subprocess management into Pool with lazy spawning and subprocess reuse.

### Tasks

- [ ] Add `subprocesses: Mutex<HashMap<DeviceId, Arc<SubprocessHandle>>>` field to Pool
- [ ] Implement `get_or_spawn_subprocess(device_id)` with lazy spawning
- [ ] Update `Pool::acquire()` to use subprocess for streaming
- [ ] Update `Pool::shutdown()` to terminate all subprocesses
- [ ] Test subprocess spawned only on first allocation
- [ ] Test subprocess reused for second allocation on same device
- [ ] Run `make lint` and `make test`

### Pool Integration

```rust
impl Pool {
    /// Map of device_id → subprocess handle
    subprocesses: Mutex<HashMap<hardware::DeviceId, Arc<SubprocessHandle>>>,

    pub fn acquire(&self, requirements: &TaskRequirements) -> Result<Tuner> {
        // Check shutdown
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return Err(ScannerError::PoolShutdown);
        }

        // Find available tuner
        let tuner_id = self.find_matching_tuner(requirements)?;

        // Get or spawn subprocess for device
        let subprocess = self.get_or_spawn_subprocess(&tuner_id.device_id)?;

        // Start streaming on channel
        subprocess.start_stream(tuner_id.channel_index)?;

        // Create Tuner wrapper
        Ok(Tuner {
            tuner_id,
            subprocess: Arc::clone(&subprocess),
            pool_inner: Arc::clone(&self.pool_ref),
            on_return: self.create_on_return_closure(&tuner_id),
            shutdown_mode: Arc::clone(&self.shutdown_mode),
        })
    }

    fn get_or_spawn_subprocess(
        &self,
        device_id: &hardware::DeviceId,
    ) -> Result<Arc<SubprocessHandle>> {
        let mut subprocesses = self.subprocesses.lock()?;

        // Check if subprocess already exists
        if let Some(handle) = subprocesses.get(device_id) {
            debug!(device_id = ?device_id, "Reusing existing subprocess");
            return Ok(Arc::clone(handle));
        }

        // Get device entry
        let pool_inner = self.pool_ref.lock()?;
        let device_entry = pool_inner.devices.get(device_id)
            .ok_or_else(|| ScannerError::DeviceNotFound(device_id.clone()))?;

        debug!(
            device_id = ?device_id,
            num_tuners = device_entry.num_tuners,
            "Spawning new subprocess (first allocation)"
        );

        // Spawn subprocess
        let handle = Arc::new(SubprocessHandle::spawn(
            device_id.clone(),
            device_entry.capabilities.clone(),
            self.shutdown_coordinator.token(),
        )?);

        // Store for future allocations
        subprocesses.insert(device_id.clone(), Arc::clone(&handle));

        Ok(handle)
    }

    /// Shutdown all device subprocesses
    pub fn shutdown(&self) {
        debug!("Pool: Initiating shutdown");

        // Set shutdown flag
        self.shutdown_mode.store(true, Ordering::SeqCst);

        // Transition state
        if let Ok(mut state) = self.state.lock() {
            *state = PoolState::ShuttingDown(ShuttingDown);
        }

        // Shutdown all subprocesses
        if let Ok(mut subprocesses) = self.subprocesses.lock() {
            for (device_id, subprocess) in subprocesses.iter_mut() {
                debug!(device_id = ?device_id, "Shutting down device subprocess");

                // Get mutable access via Arc
                if let Some(handle) = Arc::get_mut(subprocess) {
                    if let Err(e) = handle.shutdown() {
                        debug!(device_id = ?device_id, error = ?e, "Subprocess shutdown error");
                    }
                }
            }

            subprocesses.clear();
        }

        debug!("Pool: Shutdown complete");
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shutdown();
    }
}
```

## Proposal 8: Tuner RAII Updates

Update Tuner type to use SubprocessHandle and ensure non-blocking drop behavior during shutdown.

### Tasks

- [ ] Update `Tuner` struct to hold `Arc<SubprocessHandle>` and channel index
- [ ] Update `Tuner::drop()` to call `subprocess.stop_stream(channel)`
- [ ] Add shutdown mode check for fast exit during shutdown
- [ ] Ensure non-blocking behavior with try_lock
- [ ] Run `make lint` and `make test`

### Tuner RAII

```rust
pub struct Tuner {
    tuner_id: TunerId,
    subprocess: Arc<SubprocessHandle>,
    pool_inner: Arc<Mutex<PoolInner>>,
    on_return: Box<dyn Fn() + Send + Sync>,
    shutdown_mode: Arc<AtomicBool>,
}

impl Drop for Tuner {
    fn drop(&mut self) {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return; // Fast exit during shutdown
        }

        // Stop streaming (non-blocking)
        self.subprocess.stop_stream(self.tuner_id.channel_index);

        // Return to pool (invokes state change callbacks)
        (self.on_return)();
    }
}
```

## Proposal 9: Testing and Validation

Comprehensive testing of subprocess lifecycle, IPC communication, and shutdown safety.

### Tasks

- [ ] Test enumeration subprocess exits immediately
- [ ] Test device worker subprocess persists across allocations
- [ ] Test multiple channels on same device
- [ ] Test subprocess reuse for second allocation
- [ ] Test subprocess spawned lazily (not during add_device)
- [ ] Test control commands work (tune, gain, start/stop stream)
- [ ] Test I/Q data streams with channel tags
- [ ] Test multiple devices work simultaneously
- [ ] Test graceful shutdown with Shutdown message
- [ ] Test SIGTERM escalation on timeout
- [ ] Test SIGKILL escalation on hung subprocess
- [ ] Test zombie process prevention (no defunct processes after shutdown)
- [ ] Test socket cleanup (no stale /tmp/scanner-*.sock files)
- [ ] Test crash isolation (driver crash doesn't kill main process)
- [ ] Verify terminal output isolation (no corruption in TUI)
- [ ] Test performance (<20μs latency with postcard)

## Proposal 10: Debugging Infrastructure

Add logging, environment variables, and utilities for troubleshooting subprocess workers.

### Tasks

- [ ] Add SCANNER_WORKER_LOG environment variable for enumeration worker logging
- [ ] Add SCANNER_WORKER_LOG_DIR environment variable for device worker logging
- [ ] Implement file-based logging in enumeration_worker_main
- [ ] Implement file-based logging with device ID in device_worker_main
- [ ] Add debug assertions for subprocess state validation
- [ ] Create troubleshooting checklist documentation
- [ ] Document manual testing commands
- [ ] Test logging with actual subprocess runs

### Debugging and Troubleshooting

**Worker Command Visibility**: The `worker` command is hidden from normal help output (`#[command(hide = true)]`) to avoid confusing users, but can be invoked directly for debugging.

**Manual Testing**:
```bash
# Test enumeration worker
scanner worker enumerate

# Test device worker (requires valid device)
scanner worker device \
  --device-id "sdrplay:123456" \
  --device-args "driver=sdrplay,serial=123456" \
  --sample-rate 2000000
```

**Logging Strategy**:

Main Process Logging:
- Normal tracing/debug logs to stderr or file
- Subprocess spawn/shutdown events logged at debug level
- IPC errors logged at warn level

Worker Subprocess Logging:
```rust
// Enumeration worker
fn enumeration_worker_main() -> Result<()> {
    // Set up minimal logging to file (not stderr - that's discarded)
    if let Ok(log_file) = std::env::var("SCANNER_WORKER_LOG") {
        // Initialize file-based logging
        tracing_subscriber::fmt()
            .with_writer(std::fs::File::create(log_file)?)
            .init();
    }

    debug!("Enumeration worker starting");
    // ... enumerate devices
    debug!("Enumeration worker complete");
    Ok(())
}

// Device worker
fn device_worker_main(args: DeviceWorkerArgs) -> Result<()> {
    // Set up logging with device ID in filename
    if let Ok(log_dir) = std::env::var("SCANNER_WORKER_LOG_DIR") {
        let log_path = format!("{}/worker-{}.log", log_dir, args.device_id);
        tracing_subscriber::fmt()
            .with_writer(std::fs::File::create(log_path)?)
            .with_ansi(false)
            .init();
    }

    debug!(device_id = %args.device_id, "Device worker starting");
    // ... main loop
    debug!(device_id = %args.device_id, "Device worker shutting down");
    Ok(())
}
```

**Environment Variables**:
```bash
# Enable worker logging
export SCANNER_WORKER_LOG="/tmp/scanner-enum-worker.log"
export SCANNER_WORKER_LOG_DIR="/tmp/scanner-workers"

# Run scanner
scanner scan --band fm

# Check logs
cat /tmp/scanner-enum-worker.log
ls -la /tmp/scanner-workers/
```

**Debugging Subprocess Issues**:

Issue: Enumeration worker fails silently
```bash
# Check exit status and stderr
scanner worker enumerate
echo $?  # Non-zero indicates failure

# Capture stderr (normally discarded)
scanner worker enumerate 2>&1

# Enable logging
SCANNER_WORKER_LOG=/tmp/enum.log scanner scan --band fm
cat /tmp/enum.log
```

Issue: Device worker not responding
```bash
# Check if subprocess is running
ps aux | grep "scanner worker device"

# Check socket files exist
ls -la /tmp/scanner-*.sock

# Check worker logs
tail -f /tmp/scanner-workers/worker-sdrplay-123456.log

# Manually test IPC
# (connect to socket and send test message)
```

Issue: Subprocess zombie processes
```bash
# Check for zombies
ps aux | grep defunct

# Should see no zombies after shutdown
scanner scan --band fm
# Ctrl-C
sleep 2
ps aux | grep "scanner worker" | grep defunct  # Should be empty
```

Issue: Socket files not cleaned up
```bash
# Check for stale sockets
ls -la /tmp/scanner-*.sock

# Should be cleaned up after shutdown
# If not, indicates shutdown bug
```

**Debug Build with Extra Validation**:

```rust
#[cfg(debug_assertions)]
fn validate_subprocess_state(handle: &SubprocessHandle) {
    // Extra checks in debug builds
    assert!(handle.socket_paths.0.exists(), "Control socket missing");
    assert!(handle.socket_paths.1.exists(), "Data socket missing");

    // Check process is still alive
    match handle.process.try_wait() {
        Ok(None) => {}, // Still running, good
        Ok(Some(status)) => panic!("Subprocess died: {:?}", status),
        Err(e) => panic!("Cannot check subprocess: {:?}", e),
    }
}
```

**Troubleshooting Checklist**:

Worker Won't Start:
- [ ] Check device permissions (SDR hardware access)
- [ ] Check SoapySDR driver installed
- [ ] Check socket directory writable (`/tmp`)
- [ ] Review worker logs for startup errors
- [ ] Test worker command manually

Worker Crashes:
- [ ] Check worker logs for panic/error
- [ ] Verify device args correct
- [ ] Test with different device if available
- [ ] Check system resources (memory, file descriptors)

IPC Communication Fails:
- [ ] Verify socket files exist
- [ ] Check socket permissions
- [ ] Review serialization errors in logs
- [ ] Test with smaller sample buffer

Performance Issues:
- [ ] Profile serialization overhead
- [ ] Check for socket buffer saturation
- [ ] Monitor subprocess CPU usage
- [ ] Verify no memory leaks in worker

**Development Utilities**:

```rust
// src/ipc/debug.rs (debug builds only)
#[cfg(debug_assertions)]
pub mod debug {
    use super::*;

    /// Dump IPC message for debugging
    pub fn dump_control_message(msg: &ControlMessage) {
        eprintln!("[IPC] Control: {:?}", msg);
    }

    /// Dump IQ packet metadata (not full samples)
    pub fn dump_iq_packet_metadata(packet: &IQPacket) {
        eprintln!(
            "[IPC] IQ: channel={} samples={} seq={} ts={}",
            packet.channel,
            packet.samples.len(),
            packet.sequence,
            packet.timestamp
        );
    }

    /// Check for dropped packets
    pub fn check_sequence_gaps(
        expected: &mut HashMap<usize, u64>,
        packet: &IQPacket,
    ) {
        let prev = expected.entry(packet.channel).or_insert(0);
        if packet.sequence != *prev {
            eprintln!(
                "[IPC] WARNING: Dropped packets on channel {}: expected {} got {}",
                packet.channel, *prev, packet.sequence
            );
        }
        *prev = packet.sequence + 1;
    }
}
```

## Performance Characteristics

### Latency Breakdown
- Unix socket: 2-6μs (verified)
- postcard serialize: ~0.06μs (60ns, negligible)
- postcard deserialize: ~0.18μs (180ns, negligible)
- Context switch: 5-10μs
- **Total**: 7-16μs per packet (negligible for SDR)

### Throughput
- Unix sockets: 100+ Gbps capability
- Typical SDR: 2 MSPS × 8 bytes = 16 MB/s = 0.128 Gbps
- **Headroom**: 780x more bandwidth than needed

### Memory
- Per subprocess: 10-20 MB overhead
- 3 devices: ~60 MB total (acceptable)

### CPU
- Serialization: <1% per device
- Context switching: Minimal with Unix sockets

## Notes from Research

### BladeRF Clarification
libbladeRF is thread-safe with internal per-stream locks. No documented requirement for process isolation. Subprocess approach is precautionary for consistency across all devices.

### RTL-SDR Terminal Output
"Reattached kernel driver" can be fixed with udev blacklisting on modern Linux. Subprocess isolation is one solution; proper udev configuration is simpler but requires system configuration.

### SoapySDRServer Comparison
Choosing custom IPC over SoapySDRServer for: simpler deployment, Unix socket performance advantage, no network stack overhead, full control over protocol. SoapySDRServer is battle-tested alternative with TCP+UDP.

### Serialization Choice: postcard

Using postcard for IPC serialization:
- 60ns serialize, 180ns deserialize (compared to bincode's 547μs)
- 70% message size of bincode (better for Unix socket bandwidth)
- Built for embedded systems, well-suited for small I/Q packets
- Fallback options: bincode (if compatibility issues), rkyv (if zero-copy needed)
