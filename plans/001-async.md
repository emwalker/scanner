# Multi-SDR Architecture: Async vs Threads Analysis

**Date**: 2025-10-03
**Status**: Research & Planning
**Goal**: Determine optimal concurrency model for supporting multiple simultaneous SDRs

## Executive Summary

**Recommendation: Hybrid Architecture** - Use dedicated threads for blocking I/O, Rayon for CPU-bound work, and Tokio async for coordination.

This approach is validated by:
- Industry best practices (Tokio documentation, 2024)
- FutureSDR research (modern async SDR framework)
- Rust ecosystem patterns for mixed workloads

## Background

Current scanner uses single SDR with hybrid threading model:
- Main scan thread (sequential window processing)
- Per-SDR graph thread (SoapySDR streaming)
- Audio playback thread
- Detection threads (per-window candidate detection)
- TUI thread (terminal rendering)

**Goal**: Support multiple simultaneous SDRs for:
1. Direction finding (phase coherent)
2. Wide-band coverage (frequency mosaic)
3. Signal comparison/diversity
4. Distributed scanning (network SDRs)

## Research Findings

### FutureSDR (Modern Async SDR Framework)

FutureSDR is an experimental open-source SDR framework implemented in Rust using async/await.

**Key Findings:**
- ✅ **Async works well for SDR**: Real-time signal processing is viable with async
- ✅ **Performance is comparable**: Benchmarks showed "minor" differences between sync/async
- ✅ **Simpler architecture**: Concluded "not worth supporting sync implementations"
- ⚠️ **Still experimental**: Only ~5,100 LOC, far fewer features than GNU Radio

**Architecture:**
- Async runtime with executor threads
- When awaiting, executor continues useful work rather than stalling
- Can shift blocks between threads during runtime for free

**Limitations:**
- Minimal DSP block library
- Less mature than GNU Radio
- Thread-based GNU Radio approach considered "dead-end" by FutureSDR authors

### Tokio Best Practices (2024)

Critical guidelines for mixing blocking and async:

**DON'T:**
- ❌ Use `spawn_blocking` for CPU-intensive work (can exhaust thread pool)
- ❌ Block async runtime threads (>10-100μs between `.await` points)
- ❌ Use nested runtimes (`block_on` from within async context)

**DO:**
- ✅ Use `spawn_blocking` for blocking I/O only (file systems, C libraries)
- ✅ Use Rayon for CPU-bound computation
- ✅ Use dedicated threads for long-running sync work
- ✅ Limit CPU parallelism with semaphores

**Pattern:**
```rust
// For CPU-bound computation: use Rayon
// For blocking IO: use spawn_blocking
// For sync work that runs forever: spawn dedicated thread
```

### Hybrid Pattern Validation

The hybrid approach is **standard practice** in Rust:
- Separate thread pools for different workload types
- Async runtime for I/O and coordination
- Bridge via async channels (`tokio::mpsc`)

## Architecture Options

### Option 1: Pure Threaded (Current + Scaled)

**Structure:**
- One thread per SDR for sample streaming
- Shared broadcast channels for distribution
- Central coordinator thread

**Pros:**
- ✅ Natural fit for blocking I/O (SDR hardware)
- ✅ Simple mental model (each SDR = one thread)
- ✅ OS scheduler handles fairness
- ✅ Great for CPU-bound work (FFT, demodulation)
- ✅ Existing rustradio compatibility
- ✅ Easy debugging (standard tools)

**Cons:**
- ❌ Thread overhead (~1-2MB stack per thread)
- ❌ Context switching overhead
- ❌ Coordination complexity (synchronizing N threads)
- ❌ Channel saturation with multiple broadcasts

**Best for:** 2-8 SDRs, independent scanning, CPU-intensive processing

### Option 2: Pure Async (Tokio-based)

**Structure:**
- Tokio runtime with async tasks per SDR
- `tokio::select!` for multi-device coordination

**Pros:**
- ✅ Efficient coordination (`tokio::select!` is elegant)
- ✅ Lower overhead (async tasks ~2KB vs thread ~2MB)
- ✅ Better for I/O-bound operations
- ✅ Built-in cancellation and timeouts
- ✅ Natural backpressure
- ✅ Could scale to 100s of devices

**Cons:**
- ❌ Blocking SDR APIs (need `spawn_blocking` - negates benefits)
- ❌ Runtime overhead and complexity
- ❌ Async "color" spreads through codebase
- ❌ Rustradio incompatibility (blocking `Graph::run()`)
- ❌ FFT/DSP work needs offloading
- ❌ Harder debugging

**Best for:** Pure I/O workloads, network-based SDRs, native async drivers

### Option 3: Hybrid (Recommended)

**Structure:**
```rust
struct HybridMultiSdrScanner {
    // Dedicated threads for SDR I/O (blocking C APIs)
    sdr_threads: Vec<std::thread::JoinHandle<()>>,

    // Rayon for CPU-intensive DSP/FFT
    rayon_pool: rayon::ThreadPool,

    // Tokio async runtime for coordination
    runtime: tokio::Runtime,

    // Async channels for coordination
    sample_channels: Vec<tokio::sync::mpsc::Receiver<Samples>>,
}
```

**Layer Separation:**

| Layer | Concurrency Model | Rationale |
|-------|------------------|-----------|
| **SDR I/O** | Dedicated threads | Blocking C APIs, runs forever |
| **DSP/FFT** | Rayon thread pool | CPU-bound, needs parallelism control |
| **Coordination** | Tokio async | Multi-way select, elegant control flow |
| **UI/Commands** | Tokio async | Low latency, responsive |
| **File I/O** | Tokio async-fs | True async I/O |

**Pros:**
- ✅ Keeps blocking I/O efficient
- ✅ Async coordination is elegant
- ✅ No rustradio rewrite needed
- ✅ Natural separation of concerns
- ✅ Can use `tokio::select!` for multi-SDR correlation
- ✅ Industry-validated pattern

**Cons:**
- ⚠️ More complex than pure approaches
- ⚠️ Need to understand multiple concurrency models

**Best for:** Production systems with mixed workloads (our use case)

## Specific Multi-SDR Use Cases

### 1. Direction Finding (Phase Coherent)
- **Need**: Synchronized sampling from 3+ SDRs
- **Best**: Threaded with hardware sync
- **Why**: Phase alignment requires hardware timing, not software

### 2. Wide-Band Coverage (Frequency Mosaic)
- **Need**: Scan different bands simultaneously
- **Best**: Hybrid - threads for streaming, async for aggregation
- **Why**: Independent operations, results need merging

### 3. Signal Comparison/Diversity
- **Need**: Same frequency on multiple SDRs
- **Best**: Hybrid - threads for capture, async for correlation
- **Why**: `tokio::select!` makes sample comparison elegant

### 4. Distributed Scanning (Network SDRs)
- **Need**: Multiple networked SDRs
- **Best**: Pure async
- **Why**: Network I/O is truly async, no blocking drivers

## Implementation Plan

### Phase 1: Keep Current Threaded Model
- Add multi-device support with additional threads
- One thread per SDR (as now)
- Use existing broadcast channels
- **Effort**: 1-2 weeks
- **Risk**: Low

### Phase 2: Add Rayon for CPU Work
```rust
// Replace CPU-bound work
rayon::scope(|s| {
    for chunk in sample_chunks {
        s.spawn(|_| process_fft(chunk));
    }
});
```
- **Effort**: 3-5 days
- **Risk**: Low
- **Benefit**: Better CPU utilization

### Phase 3: Add Async Coordinator
```rust
async fn coordinate_sdrs(
    sdr_receivers: Vec<tokio::sync::mpsc::Receiver<Complex>>,
    command_rx: mpsc::Receiver<Command>,
) {
    loop {
        tokio::select! {
            Some(samples_a) = sdr_receivers[0].recv() => {
                // Offload to Rayon
                let result = rayon_pool.install(|| compute_fft(samples_a));
            }
            Some(samples_b) = sdr_receivers[1].recv() => {
                // Cross-correlate
            }
            Some(cmd) = command_rx.recv() => {
                // Handle commands
            }
        }
    }
}
```
- **Effort**: 1-2 weeks
- **Risk**: Medium
- **Benefit**: Elegant multi-SDR coordination

### Phase 4: Migrate Control Flow
- Command handling → async
- UI events → async
- Keep SDR I/O and DSP threaded
- **Effort**: 2-3 weeks
- **Risk**: Medium

## Key Implementation Patterns

### Multi-SDR Correlation (Async)
```rust
async fn correlate_signals(
    mut sdr1: tokio::sync::mpsc::Receiver<Complex>,
    mut sdr2: tokio::sync::mpsc::Receiver<Complex>,
) {
    loop {
        tokio::select! {
            Some(sample1) = sdr1.recv() => {
                // Buffered correlation logic
            }
            Some(sample2) = sdr2.recv() => {
                // Compare with sdr1 buffer
            }
        }
    }
}
```

### Resource Management
```rust
struct SdrPool {
    devices: Vec<Arc<SdrDevice>>,
    // Tokio semaphore for rate limiting
    permits: Arc<tokio::sync::Semaphore>,
}
```

### CPU-Bound Processing
```rust
// Use Rayon, NOT spawn_blocking
fn process_samples(samples: Vec<Complex>) -> FftResult {
    rayon::spawn(|| {
        compute_fft(samples)
    })
}
```

## Current System Analysis

**Existing Architecture:**
- Main thread: Sequential window processing
- `SoapySdrManager`: Spawns thread for SDR graph (`soapy.rs:171`)
- `AudioSession`: Spawns thread for audio playback (`audio_session.rs:77`)
- Window processing: Spawns detection threads (`window.rs:232`)
- Communication: Mix of `tokio::sync::broadcast` and `std::sync::mpsc`

**What Works:**
- `tokio::sync::broadcast` for high-throughput SDR samples
- Dedicated threads for blocking operations
- Clear separation between streaming and processing

**What Could Improve:**
- Multi-way coordination (would benefit from `tokio::select!`)
- CPU-bound work (would benefit from Rayon)
- Command handling (would benefit from async)

## Critical Insights

### From FutureSDR
- **Async IS viable for SDR/DSP** - performance differences are "minor"
- Full async requires framework rewrite (not worth it for us)
- Async scheduling can be advantageous for dynamic workloads

### From Tokio Best Practices
- **Never use `spawn_blocking` for CPU work** - use Rayon instead
- **Dedicated threads for forever-running sync work** - our SDR I/O
- **10-100μs rule** - max time between `.await` points

### From Hybrid Pattern
- **Industry standard** for mixed workloads
- **Channel-based coordination** between concurrency models
- **Clear layer separation** by workload type

## Recommendations Summary

1. **Start with threaded multi-SDR** (Phase 1)
   - Lowest risk, works with existing code
   - One thread per SDR

2. **Add Rayon for CPU work** (Phase 2)
   - Easy win for FFT/DSP parallelism
   - Better CPU utilization

3. **Add Tokio coordinator when needed** (Phase 3)
   - For sophisticated multi-device correlation
   - Elegant `tokio::select!` for multi-way operations

4. **Keep blocking operations in threads**
   - SDR I/O: dedicated threads
   - DSP/FFT: Rayon
   - Don't use `spawn_blocking` for CPU work

5. **Don't rewrite rustradio**
   - Current blocking model is fine
   - FutureSDR shows async is possible, but not necessary

## Future Considerations

### If Going Full Async (Not Recommended Now)
- Would need async-native SDR driver
- Would need to fork/rewrite rustradio
- FutureSDR proves it's technically viable
- Performance gains are minor per research

### Alternative: Contribute to FutureSDR
- If we need pure async in future
- Already has async architecture
- Still needs DSP block library development

## References

- [FutureSDR](https://www.futuresdr.org/) - Async SDR framework in Rust
- [FutureSDR Benchmarking](https://www.futuresdr.org/blog/benchmarking/)
- [FutureSDR Sync vs Async](https://www.futuresdr.org/blog/sync-vs-async/)
- [Tokio spawn_blocking docs](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html)
- [Alice Ryhl: Async - What is Blocking?](https://ryhl.io/blog/async-what-is-blocking/)
- [Bridging Async and Sync in Rust](https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust)

## Conclusion

The hybrid architecture is the **pragmatic, research-validated choice** for multi-SDR support:

1. **Proven pattern**: Industry standard for mixed workloads
2. **Incremental adoption**: Can migrate piece by piece
3. **Best performance**: Threads for I/O, Rayon for CPU, async for coordination
4. **Maintainable**: Clear separation, easier to reason about
5. **Future-proof**: Can add more async as needed

The research strongly validates using **dedicated threads for SDR I/O**, **Rayon for CPU-bound work**, and **Tokio async for coordination** - exactly what the hybrid model provides.
