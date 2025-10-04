# SDR Sample Batching: Performance Analysis

**Date**: 2025-10-03
**Status**: Analysis & Recommendation
**Issue**: High CPU usage from numerous small receives on broadcast channels

## Problem Statement

Current implementation sends/receives **individual Complex samples** over `tokio::sync::broadcast` channels:
- Sample rate: ~2 MHz (2,000,000 samples/second)
- Each sample: ~8-16 bytes (Complex<f32>)
- **2 million send/receive operations per second**
- Observable: CPU fan spins up during scanning

## Current Architecture

### BroadcastSink (Lines 30-58)
```rust
// Sends ONE sample at a time
for sample in samples {
    match self.sender.send(*sample) {  // Individual send
        Ok(_) => sent += 1,
        ...
    }
}
```

### BroadcastSource (Lines 97-167)
```rust
// Receives ONE sample at a time
while n < batch_size {
    match self.receiver.try_recv() {  // Individual receive
        Ok(sample) => {
            out_slice[n] = sample;
            n += 1;
        }
        ...
    }
}
```

**Analysis**: Receiver already attempts batching, but sender sends individually. This creates a **hot loop** with 2M system calls/second.

## Research Findings

### Tokio Broadcast Limitations

From research:
1. **No `recv_many` for broadcast**: Only available for `mpsc` channels
2. **No bulk send API**: Broadcast only has single-item `send()`
3. **Ring buffer implementation**: Values stored once, cloned per receiver
4. **Known performance issue**: Quadratic slowdown with >40 receivers

### Batching Benefits (from research)

**General Principles:**
- ✅ Reduces per-operation overhead (syscalls, locks, cache misses)
- ✅ Better CPU cache utilization (sequential access)
- ✅ Amortizes fixed costs (function call overhead)
- ✅ Lock-free approaches outperform mutex-based

**From SDR domain:**
- Packet buffers critical at high sample rates
- CPU at 40-60% can still drop samples (synchronization overhead)
- FPGA adapters reduce protocol overhead with larger chunks
- GNU Radio is 25x faster than OSSIE (better batching)

**From Rust ecosystem:**
- `recv_many` improves throughput for batched workloads
- Crossbeam/Flume channels designed for high-performance
- Lock-free queues eliminate contention

### Key Quote from Research

> "Functions with high startup costs and linear complexity benefit from batching to reduce the fixed overhead of function invocations."

**This exactly describes our scenario**:
- High startup cost: Atomic operations, potential cache misses per send/recv
- Linear complexity: Processing N samples
- Solution: Send/receive in batches

## Proposed Solutions

### Option 1: Packet-Based Broadcast (Recommended)

**Concept**: Send `Vec<Complex>` instead of individual `Complex`

```rust
// Current
pub struct BroadcastSink {
    sender: broadcast::Sender<Complex>,  // Individual samples
}

// Proposed
pub struct BroadcastSink {
    sender: broadcast::Sender<Arc<Vec<Complex>>>,  // Packets
    packet_size: usize,  // e.g., 1024 samples
}
```

**Implementation:**
```rust
impl Block for BroadcastSink {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();

        // Batch into packets
        for chunk in samples.chunks(self.packet_size) {
            let packet = Arc::new(chunk.to_vec());
            match self.sender.send(packet) {
                Ok(_) => { /* sent ~1024 samples in one call */ }
                Err(_) => break,
            }
        }
        ...
    }
}

impl Block for BroadcastSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        // Receive packets
        match self.receiver.try_recv() {
            Ok(packet) => {
                // Write entire packet to output
                // ~1024 samples in one receive
                out_slice[..packet.len()].copy_from_slice(&packet);
            }
            ...
        }
    }
}
```

**Pros:**
- ✅ **Minimal code changes**: ~50 lines modified
- ✅ **Use existing broadcast**: No new dependencies
- ✅ **Arc avoids copies**: Single allocation, cloned Arc per receiver
- ✅ **Tunable packet size**: Adjust for latency/throughput tradeoff
- ✅ **Backwards compatible**: Can wrap in same trait

**Cons:**
- ⚠️ **Latency increase**: Packets add ~0.5ms latency (1024 samples @ 2MHz)
- ⚠️ **Memory overhead**: More Vec allocations
- ⚠️ **Arc overhead**: Reference counting per packet

**Performance Estimate:**
- Current: 2,000,000 ops/sec
- With 1024-sample packets: ~2,000 ops/sec
- **1000x reduction in channel operations**

### Option 2: Replace with High-Performance Channel

**Concept**: Use Crossbeam or custom lock-free queue

```rust
use crossbeam::channel;

pub struct BroadcastSink {
    // One sender, cloned for multiple receivers
    senders: Vec<channel::Sender<Vec<Complex>>>,
}
```

**Pros:**
- ✅ Lock-free performance
- ✅ Purpose-built for high throughput
- ✅ Proven in production

**Cons:**
- ❌ Requires significant refactoring
- ❌ Loses broadcast semantics (need manual fan-out)
- ❌ New dependency

### Option 3: Custom Ring Buffer

**Concept**: Shared memory ring buffer with atomic pointers

```rust
pub struct SampleRingBuffer {
    buffer: Vec<Complex>,
    write_pos: AtomicUsize,
    read_pos: Vec<AtomicUsize>,  // Per reader
}
```

**Pros:**
- ✅ Zero-copy for readers
- ✅ Maximum performance
- ✅ No channel overhead

**Cons:**
- ❌ Complex implementation
- ❌ Hard to get right (race conditions)
- ❌ Slow reader blocks others
- ❌ Not needed for current scale

## Detailed Analysis: Option 1 (Packet-Based)

### Packet Size Tradeoffs

| Packet Size | Ops/Second | Latency | Memory |
|-------------|------------|---------|--------|
| 256 samples | ~8,000 | 0.128ms | Lower |
| 512 samples | ~4,000 | 0.256ms | Medium |
| 1024 samples | ~2,000 | 0.512ms | Medium |
| 2048 samples | ~1,000 | 1.024ms | Higher |
| 4096 samples | ~500 | 2.048ms | Higher |

**Recommendation**: Start with **1024 samples**
- Good balance of throughput and latency
- 0.5ms latency acceptable for scanning (not real-time audio)
- 2000 ops/sec is manageable

### Memory Impact

**Current:**
```
Per sample: 8 bytes (Complex<f32>)
Broadcast buffer: 524,288 samples × 8 bytes = 4 MB
Per receiver: Negligible (just pointers)
```

**With packets (1024 samples):**
```
Per packet: 1024 × 8 bytes = 8 KB
Arc overhead: 16 bytes
Total per packet: ~8 KB

Broadcast buffer: 512 packets (to match ~524K samples)
Memory: 512 × 8 KB = 4 MB (same!)
Arc clones per receiver: 16 bytes each (negligible)
```

**Conclusion**: Minimal memory increase, Arc keeps it efficient

### Implementation Checklist

1. **Define packet type**
   ```rust
   pub type SamplePacket = Arc<Vec<Complex>>;
   ```

2. **Update BroadcastSink**
   - Add packet size configuration
   - Batch samples into Vec
   - Wrap in Arc
   - Send packets

3. **Update BroadcastSource**
   - Receive SamplePacket
   - Unpack and write to output stream
   - Handle partial packets at boundaries

4. **Update channel creation**
   - Adjust buffer size for packets (divide by packet_size)
   - Document packet size choice

5. **Add configuration**
   ```rust
   pub struct ScanningConfig {
       ...
       pub packet_size: usize,  // Default: 1024
   }
   ```

6. **Testing**
   - Verify no sample drops
   - Measure CPU usage improvement
   - Check latency impact
   - Benchmark throughput

## Alternative: Hybrid Approach

Keep current implementation for low-latency paths, use packets for high-throughput:

```rust
pub enum SampleStream {
    Individual(broadcast::Sender<Complex>),
    Batched(broadcast::Sender<SamplePacket>),
}
```

**Use cases:**
- Individual: Audio playback (low latency critical)
- Batched: Peak detection, FFT (throughput critical)

## Research-Based Recommendations

Based on search findings:

### DO:
1. ✅ **Batch aggressively for CPU-bound work**: Our FFT/detection is CPU-bound
2. ✅ **Use Arc for shared data**: Avoids copies while maintaining broadcast
3. ✅ **Tune packet size empirically**: Profile with different sizes
4. ✅ **Consider lock-free alternatives**: If batching insufficient

### DON'T:
1. ❌ **Don't expect `recv_many` on broadcast**: It doesn't exist
2. ❌ **Don't over-optimize prematurely**: Start with simple packet approach
3. ❌ **Don't ignore latency**: Monitor impact on responsiveness

## Expected Performance Improvement

**Current bottleneck:**
- 2M `send()` calls/second
- 2M `try_recv()` calls/second
- Atomic operations, cache misses, overhead per call

**With 1024-sample packets:**
- ~2K `send()` calls/second (1000x reduction)
- ~2K `try_recv()` calls/second (1000x reduction)
- CPU can process samples instead of managing channels

**Estimated CPU reduction**: 30-50%
- Based on research: Batching reduces overhead by orders of magnitude
- GNU Radio's 25x speedup over OSSIE largely due to batching
- Our receiver already tries to batch, but sender negates it

## Implementation Priority

**Phase 1: Prototype** (1-2 days)
- Implement packet-based broadcast
- Use fixed 1024 packet size
- Test with current workload

**Phase 2: Tune** (2-3 days)
- Profile CPU usage
- Experiment with packet sizes
- Measure latency impact

**Phase 3: Optimize** (optional)
- If still bottlenecked, consider Crossbeam
- Add adaptive packet sizing
- Implement zero-copy paths

## Risks & Mitigations

### Risk 1: Increased Latency
- **Impact**: Slower response to signals
- **Mitigation**: Acceptable for scanning (not real-time)
- **Threshold**: <2ms added latency is fine

### Risk 2: Memory Pressure
- **Impact**: More Vec allocations
- **Mitigation**: Arc keeps overhead low, reuse buffers
- **Monitoring**: Track allocator stats

### Risk 3: Complexity
- **Impact**: Harder to debug
- **Mitigation**: Add packet tracing, clear documentation
- **Fallback**: Keep individual sample path as option

## Decision Matrix

| Criterion | Current | Option 1 (Packets) | Option 2 (Crossbeam) | Option 3 (Ring) |
|-----------|---------|-------------------|---------------------|-----------------|
| Effort | N/A | Low | Medium | High |
| CPU Gain | 0% | 30-50% | 40-60% | 50-70% |
| Latency | Low | +0.5ms | +0.3ms | Minimal |
| Complexity | Simple | Simple | Medium | Complex |
| Risk | N/A | Low | Medium | High |
| **Score** | - | **9/10** | 7/10 | 5/10 |

## Recommendation

**Implement Option 1: Packet-Based Broadcast**

**Justification:**
1. Research proves batching is effective for this workload
2. Minimal code changes, low risk
3. Expected 30-50% CPU reduction
4. Latency increase acceptable for scanning
5. Can iterate to other options if needed

**Success Metrics:**
- CPU usage reduction: >30%
- No sample drops at 2 MHz
- Latency increase: <1ms
- Code changes: <100 lines

## Code Example: Complete Implementation

```rust
// Type alias for clarity
pub type SamplePacket = Arc<Vec<Complex>>;

// Updated BroadcastSink
pub struct BroadcastSink {
    input: ReadStream<Complex>,
    sender: broadcast::Sender<SamplePacket>,
    packet_size: usize,
    buffer: Vec<Complex>,  // Accumulator
}

impl BroadcastSink {
    pub fn new(
        input: ReadStream<Complex>,
        sender: broadcast::Sender<SamplePacket>,
        packet_size: usize,
    ) -> Self {
        Self {
            input,
            sender,
            packet_size,
            buffer: Vec::with_capacity(packet_size),
        }
    }
}

impl Block for BroadcastSink {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();

        let mut consumed = 0;
        for &sample in samples {
            self.buffer.push(sample);
            consumed += 1;

            if self.buffer.len() >= self.packet_size {
                let packet = Arc::new(std::mem::replace(
                    &mut self.buffer,
                    Vec::with_capacity(self.packet_size)
                ));

                if self.sender.send(packet).is_err() {
                    // No receivers, consume rest and exit
                    consumed = samples.len();
                    break;
                }
            }
        }

        input_buf.consume(consumed);
        Ok(BlockRet::Again)
    }
}

// Updated BroadcastSource
pub struct BroadcastSource {
    output: WriteStream<Complex>,
    receiver: broadcast::Receiver<SamplePacket>,
    leftover: Option<(SamplePacket, usize)>,  // (packet, offset)
}

impl Block for BroadcastSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let mut out = self.output.write_buf()?;
        let out_slice = out.slice();
        let mut written = 0;

        // Write leftover from previous packet first
        if let Some((packet, offset)) = &self.leftover {
            let remaining = &packet[*offset..];
            let to_write = remaining.len().min(out_slice.len());
            out_slice[..to_write].copy_from_slice(&remaining[..to_write]);
            written += to_write;

            if to_write >= remaining.len() {
                self.leftover = None;
            } else {
                self.leftover = Some((packet.clone(), offset + to_write));
            }
        }

        // Receive and write packets
        while written < out_slice.len() {
            match self.receiver.try_recv() {
                Ok(packet) => {
                    let to_write = packet.len().min(out_slice.len() - written);
                    out_slice[written..written + to_write]
                        .copy_from_slice(&packet[..to_write]);
                    written += to_write;

                    if to_write < packet.len() {
                        self.leftover = Some((packet, to_write));
                        break;
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(broadcast::error::TryRecvError::Closed) => {
                    return Ok(BlockRet::EOF);
                }
            }
        }

        if written > 0 {
            out.produce(written, &[]);
            Ok(BlockRet::Again)
        } else {
            std::thread::yield_now();
            Ok(BlockRet::Again)
        }
    }
}
```

## Next Steps

1. Create feature branch: `feature/packet-batching`
2. Implement packet-based broadcast
3. Add configuration for packet size
4. Run benchmarks and compare CPU usage
5. Iterate on packet size if needed
6. Merge when CPU reduction confirmed

## References

- [Tokio recv_many for batching](https://medium.com/@concisenotes/optimizing-batch-task-queues-with-tokio-channels-and-recv-many-d2c2f7ee204d)
- [Lock-free channels in Rust](https://www.slingacademy.com/article/building-concurrent-data-structures-in-rust-lock-free-approaches/)
- [SDR sample processing performance](https://dsp.stackexchange.com/questions/87710/sdr-lose-samples-at-high-bandwidth-sample-rate)
- [Crossbeam channels documentation](https://docs.rs/crossbeam-channel/)
- [Tokio broadcast limitations](https://docs.rs/tokio/latest/tokio/sync/broadcast/)
