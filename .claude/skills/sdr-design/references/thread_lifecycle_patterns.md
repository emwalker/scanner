# Thread Lifecycle Patterns in SDR Applications with ECS

This reference covers thread management patterns for SDR processing in applications using Entity-Component-System (ECS) architecture.

## Core Principles

### Thread Ownership Models

**Long-lived Worker Threads** (GNU Radio pattern)
- Created once during application startup or flowgraph initialization
- Persist throughout application lifetime
- Suitable for continuous processing tasks (SDR reading, demodulation)
- Lower overhead from thread creation/teardown
- Requires robust shutdown coordination

**Ephemeral Task Threads** (On-demand pattern)
- Created and destroyed per task or session
- Suitable for temporary operations (single station decode, file processing)
- Higher overhead but cleaner resource management
- Simpler shutdown - just wait for thread completion

**Hybrid Pattern** (Recommended for Scanner)
- Long-lived threads for core SDR pipeline (hardware interface, broadcast channel)
- Ephemeral threads for per-station processing (audio decode, quality analysis)
- Balances performance with resource management

## ECS State Machine Design

### Component Pattern for Thread Lifecycle

Use components to track thread state and resources:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioThreadState {
    Idle,           // No thread exists
    Starting,       // Thread spawn initiated
    Running,        // Thread actively processing
    Stopping,       // Shutdown requested
    Failed(ErrorCode),
}

pub struct AudioThreadComponent {
    pub state: AudioThreadState,
    pub handle: Option<JoinHandle<Result<(), AudioError>>>,
    pub control_tx: Option<Sender<AudioControl>>,
    pub shutdown_flag: Arc<AtomicBool>,
}
```

### State Machine Transitions

Model thread lifecycle as explicit state transitions:

```
Idle → Starting → Running → Stopping → Idle
          ↓          ↓
       Failed    Failed
```

**Transition Rules:**
- `Idle → Starting`: ECS system spawns thread, stores handle
- `Starting → Running`: Thread sends "ready" signal on channel
- `Running → Stopping`: Shutdown requested, set atomic flag
- `Stopping → Idle`: Thread joined successfully, resources cleaned
- `* → Failed`: Error detected, thread panicked or returned error

### ECS System Organization

**Thread Spawner System**
```rust
fn spawn_audio_thread_system(world: &mut World) {
    // Query entities with AudioThreadComponent where state == Idle
    // and some condition requires thread (e.g., StationTuned event)
    for (entity, thread_comp) in query_idle_audio_threads(world) {
        if should_spawn_audio_thread(world, entity) {
            let (control_tx, control_rx) = channel();
            let shutdown_flag = Arc::new(AtomicBool::new(false));

            let handle = spawn_audio_worker(control_rx, shutdown_flag.clone());

            thread_comp.state = AudioThreadState::Starting;
            thread_comp.handle = Some(handle);
            thread_comp.control_tx = Some(control_tx);
            thread_comp.shutdown_flag = shutdown_flag;
        }
    }
}
```

**Thread Monitor System**
```rust
fn monitor_audio_thread_system(world: &mut World) {
    for (entity, thread_comp) in query_active_audio_threads(world) {
        match thread_comp.state {
            AudioThreadState::Starting => {
                // Check if thread sent ready signal
                if thread_sent_ready_on_channel() {
                    thread_comp.state = AudioThreadState::Running;
                }
            }
            AudioThreadState::Running => {
                // Check if thread is still alive
                if thread_comp.handle.is_finished() {
                    thread_comp.state = AudioThreadState::Failed(ErrorCode::UnexpectedExit);
                }
            }
            AudioThreadState::Stopping => {
                // Poll join handle with timeout
                if let Some(handle) = thread_comp.handle.take() {
                    match handle.join_timeout(Duration::from_millis(10)) {
                        Ok(Ok(())) => {
                            thread_comp.state = AudioThreadState::Idle;
                            thread_comp.control_tx = None;
                        }
                        Ok(Err(e)) => {
                            thread_comp.state = AudioThreadState::Failed(e.into());
                        }
                        Err(_timeout) => {
                            // Still stopping, put handle back
                            thread_comp.handle = Some(handle);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
```

**Thread Shutdown System**
```rust
fn shutdown_audio_thread_system(world: &mut World) {
    // Run on application shutdown or when tuning away from station
    for (entity, thread_comp) in query_running_audio_threads(world) {
        if should_stop_thread(world, entity) {
            // Set shutdown flag (non-blocking)
            thread_comp.shutdown_flag.store(true, Ordering::SeqCst);

            // Send shutdown message on control channel
            if let Some(tx) = &thread_comp.control_tx {
                let _ = tx.send(AudioControl::Shutdown);
            }

            thread_comp.state = AudioThreadState::Stopping;
        }
    }
}
```

## Thread Design Patterns

### GNU Radio Thread-per-Block Pattern

Each processing block runs in dedicated thread:
- **Pros**: Maximum parallelism, independent block scheduling
- **Cons**: High thread count, synchronization overhead
- **Use when**: Many independent processing stages, CPU cores available

### Pipeline Thread Pattern

Single thread processes entire pipeline sequentially:
- **Pros**: Minimal context switching, cache-friendly
- **Cons**: No parallelism, blocking stages stall pipeline
- **Use when**: Simple pipeline, latency-sensitive, limited cores

### Stage-based Threading (Recommended)

Group related operations into stages, one thread per stage:
- SDR Reader Thread: Hardware interface, broadcast channel feeding
- Peak Detection Thread: Scan FFT, peak finding
- Audio Thread(s): Per-station demodulation and playback
- **Pros**: Balanced parallelism and overhead
- **Cons**: Requires careful stage boundary design

## Shutdown Coordination

### Shutdown Flag Pattern

```rust
pub struct WorkerThread {
    shutdown: Arc<AtomicBool>,
    control_rx: Receiver<Control>,
}

impl WorkerThread {
    fn run(&self) {
        while !self.shutdown.load(Ordering::SeqCst) {
            match self.control_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Control::Process(data)) => self.process(data),
                Ok(Control::Shutdown) => break,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        // Cleanup
    }
}
```

### Coordinated Shutdown Sequence

1. **ECS Main Loop**: Set shutdown flag, send control messages
2. **Worker Threads**: Check flag in loop, drain queues, exit gracefully
3. **ECS Monitor System**: Poll join handles with timeout
4. **Timeout Handling**: Log warning, continue shutdown (don't block)

### Avoiding Shutdown Deadlocks

**Common Deadlock**: Worker thread blocks on full channel while ECS waits for thread to exit

**Solution**: Use try_send or send_timeout in worker threads
```rust
// In worker thread producing results
match result_tx.send_timeout(result, Duration::from_millis(100)) {
    Ok(()) => {},
    Err(SendTimeoutError::Timeout(_)) => {
        if self.shutdown.load(Ordering::SeqCst) {
            return; // Exit instead of blocking
        }
    }
    Err(SendTimeoutError::Disconnected(_)) => return,
}
```

## Audio Thread Specific Patterns

### Continuous Audio Thread (Recommended for Scanner)

**Lifecycle**: Created when station tuned, destroyed when tuning away

**Responsibilities**:
- Receive FM-demodulated samples from broadcast channel
- Apply de-emphasis filter
- Resample to 48kHz audio rate
- Feed audio playback device

**State transitions**:
```
StationTuned event → spawn thread
Audio quality good → keep running
Tune to different station → shutdown old, spawn new
Application exit → shutdown all
```

### Shared Audio Thread (Alternative)

Single audio thread handles all stations, switches active source:
- **Pros**: No thread creation overhead during scanning
- **Cons**: Complex internal state machine, can't play multiple stations

## Worker Thread Communication Patterns

### Command Pattern
```rust
pub enum AudioCommand {
    Start { station_id: StationId },
    UpdateVolume(f32),
    Shutdown,
}
```

### Result Pattern
```rust
pub enum AudioEvent {
    Started { station_id: StationId },
    QualityUpdate { snr_db: f32 },
    Error(AudioError),
    Stopped,
}
```

Send results back to ECS via channel, process in ECS system:
```rust
fn process_audio_events_system(world: &mut World) {
    while let Ok(event) = audio_event_rx.try_recv() {
        match event {
            AudioEvent::QualityUpdate { snr_db } => {
                // Update ECS component
            }
            AudioEvent::Error(e) => {
                // Transition state machine to Failed
            }
            _ => {}
        }
    }
}
```

## Performance Considerations

### Thread Priority (Linux)

Use thread priority for real-time audio thread:
```rust
use libc::{pthread_self, pthread_setschedparam, sched_param, SCHED_FIFO};

unsafe {
    let mut param = sched_param { sched_priority: 10 };
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
}
```

Requires CAP_SYS_NICE capability or root. Consider `nice` value instead for non-root.

### CPU Affinity

Pin SDR reader to dedicated core to avoid scheduling jitter:
```rust
use core_affinity::{CoreId, set_for_current};

// In SDR reader thread
set_for_current(CoreId { id: 0 });
```

### Thread Count Guidelines

- **SDR Reader**: 1 thread (hardware interface)
- **Peak Detection**: 1 thread (or same as SDR reader if fast enough)
- **Audio Threads**: 1 per active station (typically 1, max 2-3)
- **ECS Main Loop**: 1 thread
- **Total**: ~3-6 threads typical, avoid exceeding CPU core count

## Reference Implementations

- **GNU Radio**: Thread-per-block, see `gnuradio-runtime/lib/scheduler.h`
- **RustRadio**: Async/await single-threaded runtime with cooperative scheduling
- **Scanner (this project)**: Hybrid pattern with long-lived SDR thread + ephemeral audio threads
