# Subprocess Management for SDR Worker Processes

This reference covers subprocess patterns for delegating SDR hardware interface to a separate process, focused on the scanner's use case of isolating hardware I/O from the main application.

## When to Use Subprocesses vs Threads

### Use Subprocess When:
- **Process isolation**: Hardware driver crashes shouldn't kill main app
- **Language boundaries**: SDR driver in C/C++, main app in Rust
- **Resource limits**: Want to kill and restart SDR process if it leaks memory
- **Security**: Untrusted SDR driver, want OS-level isolation
- **Privilege separation**: SDR access requires different permissions

### Use Threads When:
- **Low latency**: Subprocess IPC has overhead (serialization, context switches)
- **Shared memory**: Need zero-copy access to sample buffers
- **Simple coordination**: Thread synchronization simpler than process coordination
- **Same language**: No FFI boundary, all Rust

### Scanner Decision: Subprocess

**Rationale**:
- SoapySDR C++ driver could crash on hardware errors
- Clean restart possible if SDR hangs
- Main UI remains responsive during SDR operations
- Logs can be separated (subprocess writes to separate file)

## Subprocess Architecture

### Process Roles

**Main Process**:
- ECS event loop
- UI/TUI rendering
- Peak detection
- Audio playback
- Coordination and state management

**SDR Subprocess**:
- SoapySDR hardware interface
- Sample reading from SDR
- Optional: basic preprocessing (DC removal, decimation)
- Send samples to main process via IPC

### Communication Channels

**Main → Subprocess (Commands)**:
```rust
pub enum SdrCommand {
    SetFrequency(f64),        // Tune to frequency (Hz)
    SetSampleRate(f64),       // Change sample rate
    SetGain(f32),             // Adjust gain (dB)
    Start,                    // Start streaming
    Stop,                     // Stop streaming
    Shutdown,                 // Clean exit
}
```

**Subprocess → Main (Data + Events)**:
```rust
pub enum SdrMessage {
    Samples(Vec<Complex<f32>>),           // IQ sample chunk
    Started { actual_freq: f64, actual_rate: f64 },
    Stopped,
    Error(String),
    Overrun,                              // Buffer overrun on hardware
}
```

## IPC Mechanisms

### Option 1: Unix Domain Socket (Recommended)

**Advantages**:
- Bidirectional, full-duplex
- Reliable, ordered delivery (like TCP)
- Low latency (no network stack)
- Works on Linux, macOS, BSD

**Implementation**:
```rust
use std::os::unix::net::{UnixStream, UnixListener};
use serde::{Serialize, Deserialize};
use bincode;

// Main process: create socket, spawn subprocess
let socket_path = "/tmp/scanner-sdr.sock";
let listener = UnixListener::bind(socket_path)?;

let child = Command::new("./sdr-worker")
    .arg("--socket")
    .arg(socket_path)
    .spawn()?;

let (socket, _) = listener.accept()?;

// Send command
let cmd = SdrCommand::SetFrequency(88.9e6);
let encoded = bincode::serialize(&cmd)?;
socket.write_all(&encoded)?;

// Receive samples
let mut buffer = vec![0u8; 65536];
let n = socket.read(&mut buffer)?;
let msg: SdrMessage = bincode::deserialize(&buffer[..n])?;
```

**Subprocess side**:
```rust
// Connect to socket
let socket_path = args.socket;
let mut socket = UnixStream::connect(socket_path)?;

// Event loop: read commands, send samples
loop {
    // Check for command (non-blocking)
    socket.set_nonblocking(true)?;
    match socket.read(&mut cmd_buffer) {
        Ok(n) if n > 0 => {
            let cmd: SdrCommand = bincode::deserialize(&cmd_buffer[..n])?;
            handle_command(cmd);
        }
        _ => {}
    }

    // Read samples from SDR
    if streaming {
        let samples = sdr_device.read_stream(1024)?;

        // Send to main process
        let msg = SdrMessage::Samples(samples);
        let encoded = bincode::serialize(&msg)?;
        socket.write_all(&encoded)?;
    }
}
```

### Option 2: Pipe (Simpler, Unidirectional)

**Use for**: Commands only (main → subprocess), use separate pipe for data

```rust
use std::process::{Command, Stdio};

let mut child = Command::new("./sdr-worker")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()?;

let mut stdin = child.stdin.take().unwrap();
let mut stdout = child.stdout.take().unwrap();

// Send command
let cmd = SdrCommand::SetFrequency(88.9e6);
serde_json::to_writer(&mut stdin, &cmd)?;

// Read samples
let msg: SdrMessage = serde_json::from_reader(&mut stdout)?;
```

**Limitation**: Sequential, can't interleave commands and data easily

### Option 3: Shared Memory (Highest Performance)

**Use for**: Very high sample rates where serialization overhead matters

```rust
use shared_memory::{ShmemConf, Shmem};

// Main process: create shared memory
let shmem = ShmemConf::new()
    .size(1024 * 1024)  // 1 MB ring buffer
    .create()?;

// Spawn subprocess with shmem handle
let child = Command::new("./sdr-worker")
    .arg("--shmem-id")
    .arg(shmem.get_os_id())
    .spawn()?;

// Use ring buffer for zero-copy sample transfer
// Still need separate channel for commands and sync
```

**Complexity**: High (need ring buffer, synchronization, handle wraparound)

**Recommendation**: Start with Unix domain socket, optimize to shared memory only if profiling shows IPC bottleneck

## Subprocess Lifecycle

### Startup Sequence

```rust
pub struct SdrSubprocess {
    child: Child,
    socket: UnixStream,
    shutdown_flag: Arc<AtomicBool>,
}

impl SdrSubprocess {
    pub fn spawn(socket_path: &str) -> Result<Self> {
        // 1. Create socket
        let listener = UnixListener::bind(socket_path)?;

        // 2. Spawn subprocess
        let child = Command::new("./sdr-worker")
            .arg("--socket")
            .arg(socket_path)
            .spawn()?;

        // 3. Wait for subprocess to connect (with timeout)
        listener.set_nonblocking(false)?;
        let (socket, _) = listener.accept_timeout(Duration::from_secs(5))?;

        // 4. Send initial configuration
        let init_cmd = SdrCommand::SetSampleRate(2.048e6);
        Self::send_command(&socket, &init_cmd)?;

        // 5. Wait for ready signal
        let msg: SdrMessage = Self::recv_message(&socket)?;
        match msg {
            SdrMessage::Started { .. } => {}
            _ => return Err("Unexpected message during startup"),
        }

        Ok(Self {
            child,
            socket,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        })
    }
}
```

### Graceful Shutdown

```rust
impl SdrSubprocess {
    pub fn shutdown(&mut self) -> Result<()> {
        // 1. Send shutdown command
        self.send_command(&SdrCommand::Shutdown)?;

        // 2. Wait for subprocess to exit (with timeout)
        match self.child.wait_timeout(Duration::from_secs(2))? {
            Some(status) => {
                if status.success() {
                    Ok(())
                } else {
                    Err("Subprocess exited with error")
                }
            }
            None => {
                // 3. Timeout, force kill
                eprintln!("Subprocess didn't exit, killing");
                self.child.kill()?;
                self.child.wait()?;
                Err("Had to force kill subprocess")
            }
        }
    }
}

impl Drop for SdrSubprocess {
    fn drop(&mut self) {
        // Always try graceful shutdown
        let _ = self.shutdown();
    }
}
```

### Restart on Failure

```rust
pub struct SdrSubprocessManager {
    subprocess: Option<SdrSubprocess>,
    restart_count: usize,
    max_restarts: usize,
}

impl SdrSubprocessManager {
    pub fn ensure_running(&mut self) -> Result<&mut SdrSubprocess> {
        if let Some(ref mut subprocess) = self.subprocess {
            // Check if still alive
            match subprocess.child.try_wait()? {
                Some(_status) => {
                    // Process exited, restart
                    eprintln!("SDR subprocess exited, restarting");
                    self.subprocess = None;
                }
                None => {
                    // Still running
                    return Ok(subprocess);
                }
            }
        }

        // Spawn new subprocess
        if self.restart_count >= self.max_restarts {
            return Err("Max restarts exceeded");
        }

        self.restart_count += 1;
        let subprocess = SdrSubprocess::spawn("/tmp/scanner-sdr.sock")?;
        self.subprocess = Some(subprocess);
        Ok(self.subprocess.as_mut().unwrap())
    }
}
```

## Error Recovery

### Handling Subprocess Crashes

**Symptoms**:
- Socket read/write returns error (broken pipe, connection reset)
- `child.try_wait()` returns `Some(status)` unexpectedly

**Recovery**:
```rust
pub fn read_samples(&mut self) -> Result<Vec<Complex<f32>>> {
    match self.recv_message_timeout(Duration::from_secs(1)) {
        Ok(SdrMessage::Samples(samples)) => Ok(samples),
        Ok(other) => Err(format!("Unexpected message: {:?}", other)),
        Err(e) if e.is_connection_error() => {
            // Subprocess crashed
            eprintln!("Subprocess crashed: {}", e);

            // Restart subprocess
            self.subprocess = None;
            self.ensure_running()?;

            // Retry
            Err("Subprocess restarted, retry operation")
        }
        Err(e) => Err(e),
    }
}
```

### Handling Subprocess Hangs

**Symptoms**:
- No samples received for extended period
- Commands don't get acknowledged

**Detection**:
```rust
pub struct WatchdogTimer {
    last_sample_time: Instant,
    timeout: Duration,
}

impl WatchdogTimer {
    pub fn check_timeout(&self) -> bool {
        self.last_sample_time.elapsed() > self.timeout
    }

    pub fn pet(&mut self) {
        self.last_sample_time = Instant::now();
    }
}

// In main loop
if watchdog.check_timeout() {
    eprintln!("SDR subprocess hung, restarting");
    subprocess.kill()?;
    subprocess = SdrSubprocess::spawn()?;
}
```

## Logging and Debugging

### Separate Log Files

```rust
let log_file = format!("/tmp/scanner-worker-{}.log", std::process::id());

let child = Command::new("./sdr-worker")
    .arg("--log-file")
    .arg(&log_file)
    .stderr(Stdio::null())  // Don't inherit stderr
    .spawn()?;

// Main process logs to /tmp/scanner.log
// Subprocess logs to /tmp/scanner-worker-<PID>.log
```

**Benefits**:
- Easier to debug (separate logs for each component)
- Can tail subprocess log independently
- Subprocess crashes leave log file for postmortem

### Debug Communication

Add logging to all IPC operations:

```rust
fn send_command(&self, cmd: &SdrCommand) -> Result<()> {
    debug!("Sending command to subprocess: {:?}", cmd);
    // ... send logic ...
    debug!("Command sent successfully");
    Ok(())
}

fn recv_message(&self) -> Result<SdrMessage> {
    debug!("Waiting for message from subprocess");
    let msg = // ... receive logic ...
    debug!("Received message: {:?}", msg);
    Ok(msg)
}
```

## Integration with ECS

### ECS Component

```rust
pub struct SdrSubprocessComponent {
    pub manager: SdrSubprocessManager,
    pub socket: UnixStream,
    pub command_queue: VecDeque<SdrCommand>,
}
```

### ECS System: Send Commands

```rust
fn send_sdr_commands_system(world: &mut World) {
    let subprocess = world.get_mut::<SdrSubprocessComponent>();

    while let Some(cmd) = subprocess.command_queue.pop_front() {
        if let Err(e) = subprocess.send_command(&cmd) {
            error!("Failed to send command: {}", e);
            // Try to restart subprocess
            let _ = subprocess.manager.ensure_running();
        }
    }
}
```

### ECS System: Receive Samples

```rust
fn recv_sdr_samples_system(world: &mut World) {
    let subprocess = world.get_mut::<SdrSubprocessComponent>();
    let broadcast_hub = world.get_mut::<BroadcastHub>();

    // Non-blocking receive
    match subprocess.try_recv_message() {
        Ok(Some(SdrMessage::Samples(samples))) => {
            broadcast_hub.broadcast(&samples);
        }
        Ok(Some(SdrMessage::Error(e))) => {
            error!("SDR error: {}", e);
        }
        Ok(None) => {
            // No message available
        }
        Err(e) => {
            error!("Receive error: {}", e);
        }
    }
}
```

## Performance Considerations

### Serialization Overhead

**Measurements** (1024 Complex<f32> samples = 8 KB):
- `bincode`: ~10 μs serialize + deserialize
- `serde_json`: ~100 μs
- `msgpack`: ~20 μs

**Recommendation**: Use `bincode` for binary efficiency

### Sample Chunking

Larger chunks amortize IPC overhead:
- Small chunks (128 samples): High IPC overhead, low latency
- Large chunks (8192 samples): Low overhead, higher latency

**Recommendation**: 1024-2048 samples per chunk (good balance)

### Non-blocking I/O

Use non-blocking socket operations to avoid stalling ECS loop:

```rust
socket.set_nonblocking(true)?;

match socket.read(&mut buffer) {
    Ok(n) => { /* process n bytes */ }
    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
        // No data available, continue
    }
    Err(e) => {
        // Actual error
    }
}
```

## Scanner-Specific Implementation

See `src/subprocess/` for reference implementation:
- `src/subprocess/manager.rs`: Subprocess lifecycle management
- `src/subprocess/protocol.rs`: Command/message definitions
- `src/subprocess/worker.rs`: Subprocess main loop (if applicable)
- IPC via Unix domain socket at `/tmp/scanner-worker-{pid}.sock`

**Key patterns**:
1. Subprocess spawned on first SDR operation
2. Commands queued in ECS component, sent by system
3. Samples received non-blocking, broadcast to consumers
4. Graceful shutdown on application exit
5. Automatic restart on crash (up to max attempts)
