# Buffer Sizing for SDR Pipelines

This reference covers buffer sizing calculations, tradeoffs, and strategies for SDR processing pipelines to avoid underruns, overruns, and excessive latency.

## Buffer Fundamentals

### Why Buffers Matter

**Problem**: Producer and consumer run at different, variable rates
- SDR produces samples at constant rate (hardware clock)
- Consumer processes at variable rate (CPU scheduling, GC pauses, cache misses)

**Solution**: Buffer absorbs timing variations
- Producer writes to buffer when samples arrive
- Consumer reads from buffer when ready to process
- Buffer size determines tolerance to timing jitter

### Buffer Types

**Circular Buffer** (Ring Buffer):
```rust
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn write(&mut self, item: T) -> Result<(), BufferError> {
        let next_write = (self.write_pos + 1) % self.capacity;
        if next_write == self.read_pos {
            return Err(BufferError::Full);
        }

        self.buffer[self.write_pos] = item;
        self.write_pos = next_write;
        Ok(())
    }

    pub fn read(&mut self) -> Option<T> {
        if self.read_pos == self.write_pos {
            return None;  // Empty
        }

        let item = self.buffer[self.read_pos].clone();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        Some(item)
    }

    pub fn available(&self) -> usize {
        (self.write_pos + self.capacity - self.read_pos) % self.capacity
    }
}
```

**Advantages**:
- Fixed memory allocation
- O(1) read/write operations
- No memory fragmentation

**Channel Buffer** (crossbeam, std::sync::mpsc):
```rust
use crossbeam::channel::bounded;

let (tx, rx) = bounded(1024);  // 1024-item buffer
```

**Advantages**:
- Thread-safe built-in
- Blocking/non-blocking variants
- Simpler API than manual circular buffer

**Disadvantages**:
- Allocates memory per item (unless using Arc)
- May have higher overhead than raw circular buffer

## Buffer Size Calculation

### Latency-Based Sizing

```rust
pub fn buffer_size_for_latency(
    sample_rate: f32,
    target_latency_ms: f32,
) -> usize {
    let latency_sec = target_latency_ms / 1000.0;
    let samples = (sample_rate * latency_sec).ceil() as usize;
    samples
}

// Example: 2 MHz sample rate, 50 ms target latency
// buffer_size = 2_000_000 * 0.05 = 100_000 samples
```

**Use for**: Audio buffers where latency matters

### Processing-Time-Based Sizing

```rust
pub fn buffer_size_for_processing_time(
    sample_rate: f32,
    chunk_size: usize,
    processing_time_ms: f32,
    safety_factor: f32,
) -> usize {
    // How many chunks arrive during processing time?
    let chunks_per_sec = sample_rate / chunk_size as f32;
    let chunks_during_processing = chunks_per_sec * (processing_time_ms / 1000.0);

    // Add safety factor for jitter
    let buffer_chunks = (chunks_during_processing * safety_factor).ceil() as usize;

    // Minimum 2 chunks (ping-pong)
    buffer_chunks.max(2)
}

// Example: 2 MHz sample rate, 1024-sample chunks, 10 ms processing time, 2x safety
// chunks_per_sec = 2_000_000 / 1024 = 1953
// chunks_during_processing = 1953 * 0.010 = 19.53
// buffer_size = 19.53 * 2.0 = 39 chunks ≈ 40_000 samples
```

**Use for**: Processing pipeline stages

### Jitter-Based Sizing

```rust
pub fn buffer_size_for_jitter(
    sample_rate: f32,
    nominal_processing_time_ms: f32,
    max_jitter_ms: f32,
    percentile: f32,  // e.g., 0.99 for 99th percentile
) -> usize {
    // Assume jitter follows normal distribution
    // For 99th percentile, use ~2.33 standard deviations

    let std_dev_factor = match percentile {
        p if p >= 0.999 => 3.3,  // 99.9th percentile
        p if p >= 0.99 => 2.33,   // 99th percentile
        p if p >= 0.95 => 1.64,   // 95th percentile
        _ => 1.0,
    };

    let max_processing_time = nominal_processing_time_ms + (max_jitter_ms * std_dev_factor);
    let samples = (sample_rate * max_processing_time / 1000.0).ceil() as usize;
    samples
}

// Example: 2 MHz sample rate, 5ms nominal, 3ms jitter, 99th percentile
// max_processing_time = 5 + (3 * 2.33) = 12ms
// buffer_size = 2_000_000 * 0.012 = 24_000 samples
```

**Use for**: Real-time systems with measured jitter

### GNU Radio Approach

GNU Radio uses **page-aligned buffers** for efficient memory mapping:

```rust
pub fn gnuradio_buffer_size(
    min_samples: usize,
    sample_size_bytes: usize,
) -> usize {
    let page_size = 4096;  // Typical on Linux
    let min_bytes = min_samples * sample_size_bytes;

    // Round up to next multiple of page size
    let pages = (min_bytes + page_size - 1) / page_size;
    let buffer_bytes = pages * page_size;

    buffer_bytes / sample_size_bytes
}

// Example: Need 10,000 Complex<f32> (8 bytes each)
// min_bytes = 10_000 * 8 = 80_000
// pages = 80_000 / 4096 = 20
// buffer_bytes = 20 * 4096 = 81_920
// buffer_samples = 81_920 / 8 = 10_240
```

**Advantages**:
- Efficient virtual memory usage
- Cache-aligned for better performance

## Tradeoffs

### Latency vs Robustness

| Buffer Size | Latency | Robustness | Use Case |
|-------------|---------|------------|----------|
| Small (1-2ms) | Low | Poor | Interactive audio, gaming |
| Medium (10-50ms) | Acceptable | Good | Most SDR applications |
| Large (100+ms) | High | Excellent | Batch processing, recording |

**Example Calculation**:
```
FM broadcast: 200 kHz signal, 48 kHz audio output
- Small: 48 samples = 1ms latency (may underrun)
- Medium: 2400 samples = 50ms latency (recommended)
- Large: 9600 samples = 200ms latency (very safe but noticeable delay)
```

### Memory vs Performance

**Memory considerations**:
```rust
// Complex<f32> = 8 bytes
// 2 MHz sample rate, 100 ms buffer
let bytes = (2_000_000.0 * 0.1 * 8.0) as usize;  // 1.6 MB

// Multiple stages in pipeline
let num_stages = 5;
let total_memory = bytes * num_stages;  // 8 MB

// Multiple consumers (broadcast)
let num_consumers = 3;
let total_memory_broadcast = bytes * num_consumers;  // 4.8 MB per stage
```

**Guideline**: Keep total pipeline memory under 100 MB on embedded systems, unlimited on desktop

### CPU Cache Effects

**Cache-friendly buffer sizes**:
```rust
// L1 cache: ~32 KB per core
// L2 cache: ~256 KB per core
// L3 cache: ~8 MB shared

// Keep working set in L2 cache if possible
pub fn cache_friendly_chunk_size() -> usize {
    let l2_cache_bytes = 256 * 1024;
    let sample_bytes = 8;  // Complex<f32>

    // Use ~half of L2 cache (leave room for code, stack)
    (l2_cache_bytes / 2) / sample_bytes  // ~16K samples
}
```

**Impact**: Processing chunks that fit in cache can be 2-10x faster

## Buffer Strategies by Stage

### SDR Reader Buffer

**Goal**: Never miss samples from hardware

```rust
pub struct SdrReaderConfig {
    // Hardware buffer in driver (rtl-sdr, SoapySDR)
    pub driver_buffer_size: usize,  // Large, 256K-1M samples

    // Application buffer (from driver to processing)
    pub app_buffer_size: usize,  // Medium, 16K-64K samples
}

impl Default for SdrReaderConfig {
    fn default() -> Self {
        Self {
            driver_buffer_size: 262144,  // 256K samples = 128ms @ 2MHz
            app_buffer_size: 32768,      // 32K samples = 16ms @ 2MHz
        }
    }
}
```

**Rationale**:
- Driver buffer is large to tolerate OS scheduling jitter
- App buffer is medium to keep latency reasonable
- Total buffering: ~150ms, acceptable for scanning

### Decimation Filter Buffer

**Goal**: Match decimation ratio

```rust
pub struct DecimationStage {
    pub input_buffer: usize,    // Before decimation
    pub output_buffer: usize,   // After decimation
    pub decimation: usize,
}

impl DecimationStage {
    pub fn new(decimation: usize, input_samples: usize) -> Self {
        Self {
            input_buffer: input_samples,
            output_buffer: input_samples / decimation,
            decimation,
        }
    }
}

// Example: 2 MHz → 512 kHz (decim 4)
// Input buffer: 8192 samples = 4ms @ 2MHz
// Output buffer: 2048 samples = 4ms @ 512kHz
// Same latency, proportional sizes
```

### Audio Output Buffer

**Goal**: Prevent clicks/pops from audio system

```rust
pub struct AudioOutputConfig {
    pub buffer_size_samples: usize,
    pub num_buffers: usize,  // Ping-pong or triple buffering
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        // 48 kHz sample rate
        // 1024 samples per buffer = 21ms latency
        // 3 buffers for triple buffering
        Self {
            buffer_size_samples: 1024,
            num_buffers: 3,
        }
    }
}
```

**Trade-off**:
- Smaller buffers: Lower latency, higher CPU (more frequent callbacks)
- Larger buffers: Higher latency, lower CPU

**Typical values**:
- Low latency (music production): 64-256 samples (1-5ms)
- Normal (SDR, media playback): 512-2048 samples (10-40ms)
- High latency (robustness): 4096+ samples (85+ms)

### Broadcast Channel Buffers

**Goal**: Independent buffer sizing per consumer

```rust
pub struct BroadcastConfig {
    pub peak_detector_buffer: usize,
    pub signal_quality_buffer: usize,
    pub fm_demod_buffer: usize,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            // Peak detector is fast, small buffer
            peak_detector_buffer: calculate_buffer_size(2.0e6, 1024, 2.0, 1.5),  // ~6 chunks

            // Signal quality moderate speed
            signal_quality_buffer: calculate_buffer_size(2.0e6, 1024, 5.0, 2.0),  // ~20 chunks

            // FM demod can be slow, large buffer
            fm_demod_buffer: calculate_buffer_size(2.0e6, 1024, 10.0, 3.0),  // ~60 chunks
        }
    }
}

fn calculate_buffer_size(sample_rate: f32, chunk_size: usize, proc_time_ms: f32, safety: f32) -> usize {
    let chunks_per_sec = sample_rate / chunk_size as f32;
    let chunks = chunks_per_sec * (proc_time_ms / 1000.0) * safety;
    chunks.ceil() as usize
}
```

## Monitoring and Adaptation

### Buffer Occupancy Metrics

```rust
pub struct BufferMetrics {
    pub current_occupancy: usize,
    pub max_occupancy: usize,
    pub capacity: usize,
    pub underruns: usize,
    pub overruns: usize,
}

impl BufferMetrics {
    pub fn utilization(&self) -> f32 {
        self.current_occupancy as f32 / self.capacity as f32
    }

    pub fn is_healthy(&self) -> bool {
        self.underruns == 0 && self.overruns == 0 && self.utilization() < 0.9
    }
}
```

### Adaptive Sizing

```rust
pub struct AdaptiveBuffer<T> {
    buffer: Vec<T>,
    target_occupancy: f32,  // e.g., 0.5 (keep buffer half full)
    resize_threshold: usize,
}

impl<T: Clone> AdaptiveBuffer<T> {
    pub fn maybe_resize(&mut self) {
        let occupancy = self.current_occupancy() as f32 / self.capacity() as f32;

        if occupancy > 0.9 {
            // Running too full, increase size
            self.grow();
        } else if occupancy < 0.1 && self.capacity() > self.min_capacity {
            // Running too empty, decrease size to save memory
            self.shrink();
        }
    }

    fn grow(&mut self) {
        let new_capacity = (self.capacity() as f32 * 1.5) as usize;
        self.buffer.reserve(new_capacity - self.capacity());
    }

    fn shrink(&mut self) {
        let new_capacity = (self.capacity() as f32 * 0.75) as usize;
        self.buffer.shrink_to(new_capacity);
    }
}
```

**Caution**: Only resize during quiet periods (not while processing critical data)

## Common Buffer Problems

### Underrun (Buffer Starvation)

**Symptoms**:
- Clicks/pops in audio
- Gaps in spectrum display
- "Buffer underrun" warnings

**Causes**:
- Consumer too fast
- Producer too slow (CPU overload, I/O delay)
- Buffer too small

**Solutions**:
1. Increase buffer size
2. Reduce processing load (fewer filter taps, lower sample rate)
3. Optimize consumer code (SIMD, algorithmic improvements)
4. Increase thread priority for producer

### Overrun (Buffer Overflow)

**Symptoms**:
- Old data not consumed
- Increasing latency
- Eventually full buffer, dropped samples

**Causes**:
- Consumer too slow
- Producer too fast
- Buffer too small

**Solutions**:
1. Increase buffer size
2. Drop old samples (for real-time, latest data more important)
3. Optimize consumer code
4. Reduce producer rate (decimate earlier in pipeline)

### Excessive Latency

**Symptoms**:
- Noticeable delay between tuning and audio
- Slow response to signal changes

**Causes**:
- Buffers too large
- Too many pipeline stages

**Solutions**:
1. Reduce buffer sizes (if underruns not occurring)
2. Combine pipeline stages to reduce buffering points
3. Use smaller chunks (more frequent processing)

## Reference Values

### Scanner Application

```rust
// Tested and working configuration for Scanner

pub struct ScannerBufferConfig;

impl ScannerBufferConfig {
    pub const SDR_DRIVER_BUFFER: usize = 262144;  // 256K samples (hardware)
    pub const SDR_APP_BUFFER: usize = 32768;       // 32K samples (app-side)

    pub const BROADCAST_PEAK_DETECT: usize = 8;    // 8 chunks
    pub const BROADCAST_SIGNAL_QUALITY: usize = 16; // 16 chunks
    pub const BROADCAST_FM_DEMOD: usize = 64;       // 64 chunks

    pub const AUDIO_OUTPUT_CHUNK: usize = 1024;    // 1024 samples
    pub const AUDIO_OUTPUT_BUFFERS: usize = 3;     // Triple buffer

    pub const DECIMATION_STAGE_BUFFER: usize = 8192; // Per stage
}
```

**Sample rates**:
- SDR input: 2.048 MHz
- After first decimation: 512 kHz
- After FM demod: 256 kHz
- Audio output: 48 kHz

**Total latency**: ~150ms (acceptable for scanning, not real-time voice)

### GNU Radio Typical

```
SDR Source: 1M samples (varies by hardware)
FIR Filter: 8192 samples
FFT: 8192 samples (match FFT size)
Audio Sink: 2048-4096 samples

Total: ~200-300ms latency
```

### GQRX SDR Software

```
Input buffer: 100ms worth of samples
FFT buffer: 8192-32768 samples
Audio buffer: 50ms (2400 samples @ 48kHz)

Latency: 150-200ms
```

## Testing Buffer Sizes

### Stress Test

```rust
pub fn stress_test_buffers() {
    let mut underruns = 0;
    let mut max_occupancy = 0;

    for _ in 0..1000 {
        // Simulate variable processing time
        let jitter_ms = rand::random::<f32>() * 10.0;
        thread::sleep(Duration::from_millis(jitter_ms as u64));

        let occupancy = buffer.available();
        max_occupancy = max_occupancy.max(occupancy);

        if occupancy == 0 {
            underruns += 1;
        }
    }

    println!("Underruns: {} / 1000", underruns);
    println!("Max occupancy: {} / {}", max_occupancy, buffer.capacity());
    println!("Utilization: {:.1}%", max_occupancy as f32 / buffer.capacity() as f32 * 100.0);
}
```

**Target**: 0 underruns, <80% max utilization

## Further Reading

- GNU Radio buffer architecture: https://www.gnuradio.org/news/2017-01-05-buffers/
- Jack Audio latency guide: https://jackaudio.org/faq/latency.html
- Real-time audio programming: Ross Bencina's articles
