# CPU Optimization - BroadcastSink Sample-by-Sample Iteration

**Date**: October 3, 2025
**Status**: Resolved
**Impact**: High - Reduced CPU usage from 45% to near-zero for SDR graph thread

## Problem

The scanner was consuming 45% CPU during audio playback in browsing mode, even after implementing packet batching for the broadcast channels. Multiple optimization attempts targeting different areas had no measurable effect:

- Reducing audio sample rate (48kHz → 24kHz): No effect
- Lowering thread priority: No effect
- Increasing BroadcastSource sleep duration: No effect
- Widening filter transition bandwidth: No effect

## Root Cause

The bottleneck was in `BroadcastSink::work()` at src/broadcast.rs:90-112. The code was iterating through incoming samples **one at a time**:

```rust
for &sample in samples {
    self.buffer.push(sample);
    consumed += 1;

    if self.buffer.len() >= self.packet_size {
        // Send packet...
    }
}
```

At 2 MHz sample rate, this created **2 million loop iterations per second**, with each iteration:
- Checking the buffer length
- Pushing a single sample
- Incrementing a counter
- Checking if packet is full

This was incredibly inefficient CPU-wise, as the loop overhead dominated the actual work.

## Solution

Replaced sample-by-sample iteration with batch operations using `extend_from_slice()`:

```rust
while consumed < samples.len() {
    let space_in_buffer = self.packet_size - self.buffer.len();
    let to_copy = space_in_buffer.min(samples.len() - consumed);

    self.buffer.extend_from_slice(&samples[consumed..consumed + to_copy]);
    consumed += to_copy;

    if self.buffer.len() >= self.packet_size {
        // Send packet...
    }
}
```

### Key Changes
1. Calculate how many samples can fit in buffer in one operation
2. Copy entire slice at once using `extend_from_slice()` (highly optimized)
3. Only loop when forming/sending packets

## Results

**Loop iterations reduced from ~2,000,000/sec to ~125/sec** (16,000x reduction)

With packet_size of 16384 samples:
- Before: 2,000,000 iterations/sec (one per sample)
- After: ~125 iterations/sec (one per packet)

### CPU Usage

**Before optimization:**
- Single scanner thread: 45% CPU (SDR graph spinning on sample-by-sample iteration)
- Total: ~45-50% CPU

**After optimization:**
```
PID     TID     COMM    %CPU
1570772 1571214 scanner 2.8   (audio processing)
1570772 1590850 scanner 1.6   (likely detection/analysis)
1570772 1571213 scanner 0.3   (SDR graph - now efficient!)
1570772 1590852 scanner 0.0
1570772 1590851 scanner 0.0
1570772 1570772 scanner 0.0
```
- Total: ~5% CPU (90% reduction)
- SDR graph thread: <1% CPU (previously 45%)

## Lessons Learned

### What We Learned

1. **Packet batching isn't enough** - Even with packet-based broadcast channels, per-sample processing in the producer can still dominate CPU usage

2. **Profile before optimizing** - Multiple optimization attempts targeted the wrong areas (audio decimation, thread priority, filter design) before finding the actual bottleneck

3. **Rust optimization patterns** - `extend_from_slice()` is vastly more efficient than repeated `push()` calls due to:
   - Single bounds check instead of N checks
   - Optimized memory copy operations
   - Better CPU cache utilization

4. **High sample rates amplify inefficiencies** - At 2 MHz, even trivial per-sample overhead becomes a major bottleneck

### Supporting Optimizations

While investigating, we also made several improvements that remain beneficial:

1. **Increased BroadcastSource sleep** (100μs → 1ms when idle) - Reduces polling overhead
2. **Wider filter transition bandwidth** (120kHz → 200kHz) - Fewer filter taps, less CPU per sample
3. **Limited packets per work() call** (max 64) - Prevents excessive spinning on large buffers
4. **Thread priority reduction** - Lower priority for audio processing threads
5. **Removed audio clipping detection** - Eliminated per-sample range checks in audio callback

## Related Files

- `src/broadcast.rs:79-125` - BroadcastSink implementation
- `src/fm/filter_config.rs:76-78` - Filter transition bandwidth
- `src/types.rs:324` - Packet size configuration (16384 samples)

## References

This optimization demonstrates the importance of batch processing at high sample rates in SDR applications. Similar principles apply throughout the codebase wherever samples are processed individually.
