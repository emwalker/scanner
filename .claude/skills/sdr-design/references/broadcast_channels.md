# Broadcast Channel Patterns for Multi-Consumer SDR Pipelines

This reference covers design patterns for broadcast channels that distribute SDR data to multiple concurrent consumers (peak detection, signal quality analysis, FM demodulation).

## Broadcast Channel Fundamentals

### Why Broadcast Channels?

In SDR applications, raw or preprocessed samples often feed multiple independent processing paths:
- **Peak detection**: Find strong signals for scanning
- **Signal quality**: Measure SNR, detect interference
- **Demodulation**: Extract audio or data from specific frequency

**Problem**: Single producer (SDR reader), multiple consumers with different rates

**Solution**: Broadcast channel allows one-to-many sample distribution

### Rust Channel Options

**crossbeam::channel** (recommended):
```rust
use crossbeam::channel::{unbounded, Sender, Receiver};

let (tx, rx) = unbounded::<Vec<Complex<f32>>>();

// Multiple receivers via clone
let rx1 = rx.clone();  // Peak detector
let rx2 = rx.clone();  // Signal quality
let rx3 = rx.clone();  // FM demod
```

**std::sync::mpsc** (single consumer only):
```rust
// Can't clone receiver - not suitable for broadcast
```

**tokio::sync::broadcast**:
```rust
use tokio::sync::broadcast;

let (tx, _rx) = broadcast::channel(16);
let rx1 = tx.subscribe();  // Peak detector
let rx2 = tx.subscribe();  // Signal quality
let rx3 = tx.subscribe();  // FM demod
```

**Comparison**:
| Feature | crossbeam | tokio::broadcast |
|---------|-----------|------------------|
| Async-friendly | No | Yes |
| Cloneable receivers | Yes | Yes (via subscribe) |
| Bounded/unbounded | Both | Bounded only |
| Lagging behavior | Blocks sender | Drops old messages |
| Use case | Sync threads | Async tasks |

## Multi-Consumer Patterns

### Pattern 1: Cloned Receivers (crossbeam)

```rust
use crossbeam::channel::{bounded, Sender, Receiver};

pub struct BroadcastChannel<T> {
    tx: Sender<T>,
    // Don't store receivers - hand them out to consumers
}

impl<T: Clone> BroadcastChannel<T> {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = bounded(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> Receiver<T> {
        // This doesn't work with crossbeam - need different approach
        // See Pattern 2
    }

    pub fn send(&self, item: T) -> Result<(), SendError<T>> {
        self.tx.send(item)
    }
}
```

**Problem**: crossbeam receivers can't be cloned after creation

**Solution**: Use multiple independent channels (Pattern 2) or tokio::broadcast

### Pattern 2: Multiple Independent Channels (Recommended for Scanner)

```rust
use crossbeam::channel::{bounded, Sender, Receiver, SendError};

pub struct BroadcastHub<T> {
    senders: Vec<Sender<T>>,
}

impl<T: Clone> BroadcastHub<T> {
    pub fn new() -> Self {
        Self { senders: Vec::new() }
    }

    pub fn add_subscriber(&mut self, capacity: usize) -> Receiver<T> {
        let (tx, rx) = bounded(capacity);
        self.senders.push(tx);
        rx
    }

    pub fn broadcast(&self, item: &T) -> Result<(), BroadcastError> {
        let mut failed = Vec::new();

        for (idx, sender) in self.senders.iter().enumerate() {
            match sender.try_send(item.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    // Consumer is lagging, decide on policy
                    failed.push((idx, "full"));
                }
                Err(TrySendError::Disconnected(_)) => {
                    failed.push((idx, "disconnected"));
                }
            }
        }

        if failed.is_empty() {
            Ok(())
        } else {
            Err(BroadcastError::SomeFailed(failed))
        }
    }

    pub fn remove_disconnected(&mut self) {
        self.senders.retain(|tx| !tx.is_disconnected());
    }
}
```

**Advantages**:
- Each consumer gets independent channel with own buffer
- Can tune buffer sizes per consumer
- Easy to handle lagging consumers (drop, block, or remove)

**Disadvantages**:
- Must clone data for each consumer (CPU cost)
- More complex management

### Pattern 3: Shared Ring Buffer (Zero-copy)

For high sample rates where cloning is too expensive:

```rust
use std::sync::Arc;
use parking_lot::RwLock;

pub struct RingBuffer<T> {
    buffer: Vec<T>,
    write_pos: AtomicUsize,
    capacity: usize,
}

pub struct RingBufferReader {
    read_pos: AtomicUsize,
    buffer: Arc<RingBuffer<Complex<f32>>>,
}

impl RingBufferReader {
    pub fn read(&self, output: &mut [Complex<f32>]) -> usize {
        let read_pos = self.read_pos.load(Ordering::Acquire);
        let write_pos = self.buffer.write_pos.load(Ordering::Acquire);

        let available = write_pos.wrapping_sub(read_pos);
        let to_read = available.min(output.len());

        for i in 0..to_read {
            let idx = (read_pos + i) % self.buffer.capacity;
            output[i] = self.buffer.buffer[idx];
        }

        self.read_pos.store(read_pos + to_read, Ordering::Release);
        to_read
    }
}
```

**Advantages**:
- Zero-copy, very efficient
- Suitable for very high data rates

**Disadvantages**:
- Complex implementation
- Readers can be overrun if writer laps them
- Requires careful synchronization

## Backpressure Handling

### Problem: Slow Consumer

If one consumer can't keep up, what happens?

**Option 1: Block all consumers** (synchronized)
```rust
// sender.send(data) blocks until ALL consumers have capacity
// Slowest consumer determines throughput
```
**Use when**: All consumers are equally important, must not lose data

**Option 2: Drop messages for slow consumer** (tokio::broadcast)
```rust
match rx.recv().await {
    Ok(data) => process(data),
    Err(RecvError::Lagged(n)) => {
        eprintln!("Missed {} messages", n);
        // Continue with latest data
    }
}
```
**Use when**: Latest data more important than all data (real-time visualization)

**Option 3: Remove slow consumer**
```rust
pub fn broadcast(&mut self, item: &T) -> Result<(), BroadcastError> {
    self.senders.retain(|tx| {
        match tx.try_send(item.clone()) {
            Ok(()) => true,  // Keep sender
            Err(TrySendError::Full(_)) => {
                eprintln!("Removing slow consumer");
                false  // Remove sender
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    });
    Ok(())
}
```
**Use when**: Optional consumers, core pipeline must not be slowed

**Option 4: Adaptive decimation**
```rust
// If consumer is lagging, send every Nth sample instead of all
pub struct AdaptiveConsumer {
    rx: Receiver<Vec<Complex<f32>>>,
    decimation: AtomicUsize,  // 1 = all samples, 2 = every other, etc.
}

impl AdaptiveConsumer {
    pub fn recv(&self) -> Option<Vec<Complex<f32>>> {
        let decim = self.decimation.load(Ordering::Relaxed);

        // Skip (decim - 1) samples
        for _ in 0..(decim - 1) {
            let _ = self.rx.try_recv();
        }

        // Return the decimated sample
        self.rx.recv().ok()
    }

    pub fn adjust_decimation(&self) {
        let pending = self.rx.len();
        if pending > 100 {
            // Lagging, increase decimation
            self.decimation.fetch_add(1, Ordering::Relaxed);
        } else if pending < 10 && self.decimation.load(Ordering::Relaxed) > 1 {
            // Caught up, decrease decimation
            self.decimation.fetch_sub(1, Ordering::Relaxed);
        }
    }
}
```
**Use when**: Want graceful degradation, some data loss acceptable

## Warm-up Strategies

### Problem: Initial Transients

Many SDR processing blocks have internal state (filters, AGC, PLLs) that requires time to settle.

**Symptoms**:
- First few seconds of audio are distorted
- False peak detections at startup
- Incorrect SNR measurements initially

### Solution 1: Pre-fill with Silence

```rust
pub fn warm_up_broadcast_channel(
    tx: &Sender<Vec<Complex<f32>>>,
    sample_rate: f32,
    warmup_duration_sec: f32,
) {
    let num_samples = (sample_rate * warmup_duration_sec) as usize;
    let chunk_size = 1024;
    let silence = vec![Complex::new(0.0, 0.0); chunk_size];

    for _ in 0..(num_samples / chunk_size) {
        let _ = tx.send(silence.clone());
    }
}

// Call before starting actual SDR reading
warm_up_broadcast_channel(&tx, 2_048_000.0, 0.5);  // 500ms warmup
```

**Advantages**:
- Simple
- Ensures all consumers start with settled filters

**Disadvantages**:
- Delays startup
- Wastes computation on silent samples

### Solution 2: Discard Initial Samples

```rust
pub struct WarmupConsumer {
    rx: Receiver<Vec<Complex<f32>>>,
    samples_to_discard: AtomicUsize,
}

impl WarmupConsumer {
    pub fn new(rx: Receiver<Vec<Complex<f32>>>, warmup_samples: usize) -> Self {
        Self {
            rx,
            samples_to_discard: AtomicUsize::new(warmup_samples),
        }
    }

    pub fn recv(&self) -> Option<Vec<Complex<f32>>> {
        let mut data = self.rx.recv().ok()?;

        let to_discard = self.samples_to_discard.load(Ordering::Relaxed);
        if to_discard > 0 {
            let discarded = to_discard.min(data.len());
            self.samples_to_discard.fetch_sub(discarded, Ordering::Relaxed);

            if discarded == data.len() {
                // Entire chunk discarded, recv next
                return self.recv();
            } else {
                // Partial chunk, remove prefix
                data.drain(0..discarded);
            }
        }

        Some(data)
    }
}
```

**Advantages**:
- No startup delay
- Consumer decides its own warmup period

**Disadvantages**:
- More complex consumer logic

### Solution 3: State Indicators

```rust
pub enum SdrData {
    Warmup(Vec<Complex<f32>>),  // Still warming up
    Ready(Vec<Complex<f32>>),   // Stable data
}

pub struct SdrBroadcaster {
    tx: Sender<SdrData>,
    samples_sent: AtomicUsize,
    warmup_samples: usize,
}

impl SdrBroadcaster {
    pub fn send(&self, data: Vec<Complex<f32>>) {
        let sent = self.samples_sent.fetch_add(data.len(), Ordering::Relaxed);
        let msg = if sent < self.warmup_samples {
            SdrData::Warmup(data)
        } else {
            SdrData::Ready(data)
        };
        let _ = self.tx.send(msg);
    }
}

// Consumers can choose to ignore Warmup data or process differently
```

**Advantages**:
- Explicit state communication
- Consumers can choose behavior

## Channel Sizing

### Buffer Size Calculation

```rust
pub fn calculate_buffer_size(
    sample_rate: f32,
    chunk_size: usize,
    processing_time_ms: f32,
    safety_factor: f32,
) -> usize {
    // How many chunks arrive during processing time?
    let chunks_per_sec = sample_rate / chunk_size as f32;
    let chunks_during_processing = chunks_per_sec * (processing_time_ms / 1000.0);

    // Add safety factor
    let buffer_chunks = (chunks_during_processing * safety_factor).ceil() as usize;

    buffer_chunks.max(2)  // Minimum 2 for ping-pong
}

// Example: 2 MHz sample rate, 1024 sample chunks, 5ms processing time
// chunks_per_sec = 2_000_000 / 1024 = 1953
// chunks_during_processing = 1953 * 0.005 = 9.77
// buffer_size = 9.77 * 2.0 = 20 chunks
```

### Per-Consumer Sizing

Different consumers have different requirements:

```rust
pub struct BroadcastConfig {
    peak_detector_buffer: usize,    // Fast consumer, small buffer
    signal_quality_buffer: usize,   // Medium speed, medium buffer
    fm_demod_buffer: usize,         // Slow consumer, large buffer
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            peak_detector_buffer: 4,      // Very fast
            signal_quality_buffer: 16,    // Moderate
            fm_demod_buffer: 64,          // Can be slow
        }
    }
}
```

## Integration with ECS

### ECS Component Pattern

```rust
pub struct SdrBroadcastComponent {
    pub hub: Arc<Mutex<BroadcastHub<Vec<Complex<f32>>>>>,
    pub sample_rate: f32,
    pub chunk_size: usize,
}

pub struct PeakDetectorSubscription {
    pub rx: Receiver<Vec<Complex<f32>>>,
    pub warmup_remaining: usize,
}

pub struct FmDemodSubscription {
    pub rx: Receiver<Vec<Complex<f32>>>,
    pub station_id: StationId,
}
```

### ECS System: Subscribe to Broadcast

```rust
fn subscribe_to_sdr_broadcast_system(world: &mut World) {
    // Find entities that need SDR data but don't have subscription yet
    for (entity, peak_detector, _) in query_unsubscribed_peak_detectors(world) {
        let broadcast = world.get_resource::<SdrBroadcastComponent>();
        let mut hub = broadcast.hub.lock();

        let rx = hub.add_subscriber(16);  // 16 chunk buffer
        world.add_component(entity, PeakDetectorSubscription {
            rx,
            warmup_remaining: broadcast.sample_rate as usize,  // 1 second warmup
        });
    }
}
```

### ECS System: Cleanup Disconnected Subscribers

```rust
fn cleanup_broadcast_subscribers_system(world: &mut World) {
    let broadcast = world.get_resource::<SdrBroadcastComponent>();
    let mut hub = broadcast.hub.lock();

    // Remove disconnected channels
    hub.remove_disconnected();
}
```

## Performance Considerations

### Cloning Cost

For large sample chunks, cloning has significant CPU cost:

```rust
// 1024 Complex<f32> samples = 1024 * 8 bytes = 8 KB
// 3 consumers = 3 clones = 24 KB copied per chunk
// At 2 MHz sample rate with 1024 chunks: 2000 chunks/sec
// Total cloning: 48 MB/sec
```

**Mitigation**:
- Use Arc<Vec<T>> instead of Vec<T> (reference counted, cheap clone)
- Use shared ring buffer for very high rates
- Reduce chunk size (but increases channel overhead)

### Arc-based Broadcast

```rust
use std::sync::Arc;

pub struct BroadcastHub<T> {
    senders: Vec<Sender<Arc<T>>>,  // Send Arc instead of T
}

impl<T> BroadcastHub<T> {
    pub fn broadcast(&self, item: Arc<T>) -> Result<(), BroadcastError> {
        for sender in &self.senders {
            // Arc::clone is cheap (just increment ref count)
            let _ = sender.try_send(Arc::clone(&item));
        }
        Ok(())
    }
}

// Producer wraps data in Arc
let data = vec![/* samples */];
hub.broadcast(Arc::new(data));

// Consumers receive Arc<Vec<T>>
let arc_data = rx.recv().unwrap();
// Can deref to access data: &arc_data[0]
```

**Advantages**:
- O(1) clone cost regardless of data size
- Automatic cleanup when all consumers done

**Disadvantages**:
- Slightly more complex API
- Can't mutate received data (it's shared)

## Error Handling

### Producer Side

```rust
pub fn sdr_reader_thread(hub: Arc<Mutex<BroadcastHub<Vec<Complex<f32>>>>>) {
    loop {
        let samples = read_from_sdr(1024);

        let hub = hub.lock();
        match hub.broadcast(&samples) {
            Ok(()) => {}
            Err(BroadcastError::AllDisconnected) => {
                eprintln!("All consumers disconnected, exiting");
                break;
            }
            Err(BroadcastError::SomeFailed(failed)) => {
                eprintln!("Some consumers failed: {:?}", failed);
                // Continue anyway
            }
        }
    }
}
```

### Consumer Side

```rust
pub fn peak_detector_thread(rx: Receiver<Vec<Complex<f32>>>) {
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(samples) => {
                detect_peaks(&samples);
            }
            Err(RecvTimeoutError::Timeout) => {
                eprintln!("No data for 1 second");
                // Decide: continue waiting or exit?
            }
            Err(RecvTimeoutError::Disconnected) => {
                eprintln!("Producer disconnected, exiting");
                break;
            }
        }
    }
}
```

## Reference Implementations

- GNU Radio: Uses custom buffer class with multiple readers
- RustRadio: Async streams with tokio::broadcast
- Scanner: BroadcastHub with multiple crossbeam channels (see `src/hardware/broadcast.rs`)
