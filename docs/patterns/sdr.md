# Software-Defined Radio (SDR) Design Patterns

A comprehensive guide to architectural patterns used in SDR systems, based on GNU Radio, USRP, and industry best practices.

---

## 1. Dataflow/Pipeline Pattern

**Description:**

Stream-oriented signal processing where data flows through a directed graph of processing blocks. Each block reads samples from input ports, processes them, and writes to output ports. The runtime scheduler manages data movement between blocks through buffers.

**When to use:**

- Building signal processing chains (RF → baseband → demodulation → output)
- When processing continuous streams of samples
- When you need visual representation of signal flow (GNU Radio Companion flowgraphs)
- When blocks can be independently developed and tested
- For modular, composable signal processing architectures

**When NOT to use:**

- For non-streaming, batch-oriented processing (use offline tools)
- When you need complex control flow with conditionals and loops (use imperative code)
- For simple, single-operation processing (overhead not justified)
- When latency must be deterministic and bounded (scheduler introduces variability)
- For event-driven systems where samples don't flow continuously

---

## 2. Producer-Consumer with Ring Buffers

**Description:**

Lock-free circular buffers (ring buffers) enable efficient data passing between producer and consumer threads. Single-producer-single-consumer (SPSC) implementations use atomic operations on head/tail pointers, avoiding mutex overhead. Multi-producer or multi-consumer variants (MPMC) exist but have higher overhead.

**When to use:**

- Between acquisition thread and processing thread (samples from SDR hardware)
- When producer and consumer run at similar rates
- For low-latency inter-thread communication (microseconds)
- When buffer size is predictable and can be power-of-2 (enables bitwise masking)
- In real-time systems where lock contention is unacceptable

**When NOT to use:**

- When producer/consumer rates differ significantly (buffer will fill or starve)
- For many-to-many communication patterns (use broadcast or work queues)
- When message sizes vary dramatically (fixed-slot rings are inefficient)
- If you need backpressure or flow control (ring buffers drop or block)
- When debugging is priority (lock-free code is harder to debug)

---

## 3. Zero-Copy/DMA Pattern

**Description:**

Data moves from SDR hardware to user space without CPU-mediated copying, using Direct Memory Access (DMA) and scatter-gather operations. The kernel maps physically non-contiguous pages into a virtually contiguous buffer, allowing efficient data transfer with ~3% overhead for DMA descriptor fetches.

**When to use:**

- High sample rate applications (>10 MS/s) where CPU is bottleneck
- When moving data between IIO subsystem and network/storage
- For USRP and FMCOMMS platforms with scatter-gather DMA support
- When you need sustained throughput with minimal CPU usage
- For FPGA-accelerated processing where CPU only coordinates

**When NOT to use:**

- Low sample rates where CPU copy overhead is negligible (<1 MS/s)
- When hardware doesn't support scatter-gather DMA (requires contiguous buffers)
- If you need to inspect/modify data en route (defeats zero-copy purpose)
- For bursty traffic where setup overhead dominates (DMA works best for sustained streams)
- When memory bandwidth is the bottleneck (zero-copy doesn't help)

---

## 4. Hierarchical Block Composition

**Description:**

Complex signal processing functions are built by composing simpler, reusable blocks into hierarchical structures. A hierarchical block encapsulates a sub-flowgraph, presenting a simplified interface while hiding internal complexity. This is the primary reuse mechanism in SDR.

**When to use:**

- Creating reusable waveforms (FM receiver, BPSK demodulator)
- When a processing function appears multiple times in different flowgraphs
- To simplify complex flowgraphs (group related blocks into logical units)
- For building libraries of common SDR operations
- When you want to A/B test different implementations behind the same interface

**When NOT to use:**

- For one-off, custom processing (adds unnecessary abstraction)
- When performance is critical and abstraction overhead matters
- If the "block" has complex state that crosses boundaries (hard to encapsulate)
- When debugging flowgraphs (hierarchy hides what's happening inside)
- For trivial operations that don't benefit from reuse

---

## 5. Dynamic Reconfiguration Pattern

**Description:**

Signal processing chains adapt at runtime without stopping the flowgraph. Small-scale reconfiguration changes block parameters (gain, frequency); large-scale reconfiguration swaps entire processing chains using lock/unlock on the flowgraph. FPGA implementations use partial reconfiguration to swap hardware blocks.

**When to use:**

- Cognitive radio applications that adapt to spectrum conditions
- Multi-standard radios switching between protocols (WiFi ↔ LTE ↔ 5G)
- When different channels require different processing chains
- For frequency-hopping systems coordinating with external timing
- To time-share limited FPGA resources between processing functions

**When NOT to use:**

- When reconfiguration time exceeds acceptable latency (can be 100ms+)
- For parameter changes that can be done live (use set_* methods instead)
- If flowgraph structure is truly static (adds complexity for no benefit)
- When partial reconfiguration overhead (3-5% throughput) is unacceptable
- For safety-critical systems (reconfiguration introduces failure modes)

---

## 6. Polyphase Filterbank/Channelizer Pattern

**Description:**

Efficiently separates a wideband signal into multiple narrowband channels using a prototype lowpass filter and FFT. The polyphase implementation interleaves filtering and FFT operations, providing flat channel response and excellent out-of-band rejection (~60 dB) while being computationally efficient compared to per-channel filtering.

**When to use:**

- Multi-channel receivers processing multiple signals simultaneously
- When you need to extract specific frequency bands from wideband capture
- For cognitive radio scanning multiple channels in parallel
- When filter response quality matters (polyphase has better characteristics than DFT)
- If you have more bandwidth than needed and want to focus on specific channels

**When NOT to use:**

- Single-channel applications (use frequency xlating FIR filter instead)
- When channel spacing is irregular (polyphase assumes uniform channelization)
- If computational resources are extremely limited (FFT overhead)
- For very narrowband signals where simple decimation suffices
- When channel bandwidth requirements vary (polyphase produces equal-width channels)

---

## 7. Scheduler Throughput vs Latency Pattern

**Description:**

GNU Radio's dynamic scheduler optimizes for throughput by passing large chunks of samples (often thousands) to blocks, but this increases latency. The max_noutput_items parameter limits chunk size, trading throughput for lower latency. Tagged stream blocks are input-driven (process complete PDUs) rather than output-driven.

**When to use:**

**High throughput mode (large buffers, no limits):**
- Offline processing of recorded files
- When end-to-end latency doesn't matter
- For computationally-heavy blocks that benefit from large chunks
- When optimizing for samples-per-second metric

**Low latency mode (small max_noutput_items):**
- Interactive applications (user controls, real-time feedback)
- Closed-loop control systems
- When latency is more important than throughput
- For bursty processing where large chunks cause stalls

**When NOT to use:**

- Don't use low latency mode if throughput is primary concern (scheduler overhead increases)
- Don't use high throughput mode if you need bounded, predictable latency
- Avoid limiting noutput_items if you haven't profiled latency (premature optimization)

---

## 8. Tagged Stream Pattern

**Description:**

Metadata tags propagate alongside sample streams, marking specific samples with information like frequency, timestamp, or packet boundaries. Tags enable blocks to coordinate without out-of-band communication. PDU-length tags tell blocks how many samples to process as a unit, enabling packet-oriented processing in a stream system.

**When to use:**

- Packet-based protocols where frame boundaries matter
- When sample rate, frequency, or gain changes mid-stream
- For timestamp synchronization (tx_time tags for coordinated transmission)
- When blocks need to coordinate on sample-accurate events
- If you need to annotate samples with metadata (SNR, frequency offset)

**When NOT to use:**

- For simple continuous streams where all samples are equivalent
- When tag propagation overhead matters (every block must handle tags)
- If you need very high tag density (becomes inefficient)
- For out-of-band control (use message passing instead)
- When debugging (tags are harder to visualize than explicit signals)

---

## 9. SIMD Vectorization (VOLK) Pattern

**Description:**

The Vector-Optimized Library of Kernels (VOLK) provides hand-written SIMD implementations for common DSP operations, with runtime selection of the optimal kernel for the host architecture (SSE, AVX, NEON). This enables portable code that runs efficiently on diverse platforms without recompilation.

**When to use:**

- Performance-critical inner loops (FIR filters, FFT, correlation)
- When the same operation processes many samples (vectorization shines)
- For arithmetic-heavy operations (multiply-accumulate, complex operations)
- When you need portability across x86, ARM, and other architectures
- If ~10-50% performance improvement justifies the complexity

**When NOT to use:**

- For control logic or conditional operations (SIMD requires uniform operations)
- When data is not contiguous in memory (gather/scatter is expensive)
- If the operation is memory-bound rather than compute-bound
- For short vectors (SIMD overhead dominates)
- When code clarity is priority over performance (SIMD intrinsics are hard to read)

---

## 10. Hardware Abstraction Layer (HAL) Pattern

**Description:**

A software layer isolates waveform applications from hardware specifics, enabling the same waveform to run on different SDR platforms. The HAL has a core module (common to all platforms) and custom modules (platform-specific), presenting uniform interfaces for device control, sample access, and FPGA communication.

**When to use:**

- Building portable waveforms that run on multiple SDR platforms
- When you want to separate waveform development from hardware concerns
- For systems that need to support new hardware without modifying applications
- If you're building a framework or library used by others
- When different developers own waveform vs platform layers

**When NOT to use:**

- For single-platform applications (abstraction adds overhead)
- When you need platform-specific optimizations (HAL prevents direct access)
- If portability is not a requirement (YAGNI principle)
- For performance-critical paths where abstraction overhead matters
- When the "abstraction" doesn't actually hide meaningful differences

---

## 11. Pool Pattern (Resource Management)

**Description:**

A pool manages a set of tuners, allocating them to channels based on requirements (frequency, bandwidth, sample rate) and releasing them when no longer needed. The pool dynamically adjusts center frequencies to maximize spectrum coverage, re-tuning even locked tuners if necessary to accommodate new channels.

**When to use:**

- Multi-tuner SDR systems (multiple RTL-SDRs, multi-channel USRPs)
- When multiple channels share limited tuner resources
- For trunked radio systems with dynamic channel allocation
- When center frequency selection needs to be automatic and optimal
- If you need graceful degradation when resources are exhausted

**When NOT to use:**

- Single-tuner systems (pool overhead not justified)
- When tuner allocation is static and determined at startup
- If each channel requires dedicated hardware (no sharing possible)
- For latency-critical applications (allocation takes time)
- When simple, explicit tuner assignment is clearer (KISS principle)

---

## 12. State Machine Pattern

**Description:**

Control logic for synchronization, timing recovery, and frequency hopping is implemented as explicit finite state machines. The state machine coordinates complex sequences (acquire → track → decode) with clear state transitions, often implemented in FPGA hardware for deterministic timing.

**When to use:**

- Timing recovery (symbol synchronization state machine)
- Carrier frequency synchronization with acquisition and tracking states
- Frequency hopping coordination (idle → transmit → hop → idle)
- Protocol state machines (header search → payload decode → CRC check)
- When control flow has clear states and transitions

**When NOT to use:**

- For simple linear control flow (if-else is clearer)
- When state explosion makes the machine unmanageable (>10-15 states)
- If state is implicit in data flow (don't duplicate state)
- For highly dynamic control that doesn't fit state model
- When debugging requires understanding complex state interactions

---

## 13. Backpressure/Flow Control Pattern

**Description:**

Mechanisms to handle sample rate mismatches between producer and consumer. When the producer is faster, buffers fill and samples are dropped; when slower, buffers starve and underruns occur. Solutions include adaptive buffering, rate matching (ASRC with PID control), and explicit flow control signals.

**When to use:**

- When sample rates are not perfectly matched (clock drift)
- For network streaming where jitter and latency vary
- When processing can't keep up with acquisition rate (need to drop intelligently)
- If you need to adapt to varying system load
- For audio applications where dropouts are unacceptable (ASRC)

**When NOT to use:**

- When rates are guaranteed to match (hardware-locked clocks)
- If dropping samples is acceptable and simpler (lossy systems)
- For offline processing where there's no real-time constraint
- When latency introduced by buffering is unacceptable
- If the rate mismatch is large (flow control can't bridge large gaps)

---

## 14. Hybrid HW/SW Pipeline Pattern

**Description:**

Signal processing is split between hardware accelerators (FPGA, GPU) and software (CPU), with interposers allowing the CPU to intercept data between pipeline stages via DMA. This enables flexible control while offloading heavy computation to specialized hardware.

**When to use:**

- High sample rate systems (>100 MS/s) where CPU alone can't keep up
- When part of the processing benefits from hardware (FFT, FIR) and part needs flexibility (control logic)
- For prototyping where some blocks are in FPGA and others in development
- When you need to inspect intermediate results for debugging or adaptation
- If FPGA resources are limited and CPU can handle non-critical stages

**When NOT to use:**

- When CPU can handle the entire workload (added complexity not justified)
- If the interface overhead between HW/SW dominates (minimize crossings)
- For latency-critical paths where DMA introduces jitter
- When the processing is clearly all-HW or all-SW (hybrid adds complexity)
- If debugging across HW/SW boundary is too difficult

---

## Pattern Interactions

These patterns often work together:

- **Dataflow + Tagged Stream**: Packet processing in streaming systems
- **Polyphase Channelizer + Pool Pattern**: Multi-channel receiver with tuner management
- **Zero-Copy + Ring Buffer**: High-throughput acquisition pipeline
- **HAL + Dynamic Reconfiguration**: Portable, adaptive waveforms
- **Hierarchical Blocks + VOLK**: Reusable, optimized components
- **State Machine + Backpressure**: Robust synchronization under varying conditions

Choose patterns based on specific requirements, not trends. Simpler is better when it meets needs.

---

## References

- GNU Radio Manual and Wiki (https://wiki.gnuradio.org)
- VOLK: Vector-Optimized Library of Kernels (https://www.libvolk.org)
- Software Communications Architecture (SCA) specification
- "Software-defined Radios: Architecture, State-of-the-art, and Challenges" (Computer Communications, 2018)
- SDRTrunk: Multi-tuner resource management (https://github.com/DSheirer/sdrtrunk)
- USRP Hardware Driver (UHD) documentation
- Academic research on SDR architectures and FPGA partial reconfiguration
