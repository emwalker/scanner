---
name: sdr-design
description: This skill should be used when designing or implementing SDR (Software-Defined Radio) processing pipelines in Rust applications with ECS architecture. Apply when working with RustRadio or GNURadio patterns, designing thread lifecycles for audio/SDR workers, implementing broadcast channels for multi-consumer sample distribution, managing subprocesses for hardware interface, designing FM/AM/SSB demodulation chains, or optimizing filter taps and decimation strategies. Essential for questions about state machines for pipeline states, buffer sizing, CPU optimization in DSP code, and multi-mode demodulation architecture.
---

# SDR Design

## Overview

This skill provides comprehensive guidance for designing and implementing Software-Defined Radio (SDR) processing pipelines in Rust applications using Entity-Component-System (ECS) architecture. It covers patterns from established SDR frameworks (GNURadio, RustRadio) and applies them to multithreaded Rust applications with specific focus on thread lifecycle management, multi-consumer data distribution, signal processing optimization, and demodulation architecture.

## When to Use This Skill

Apply this skill when:

- **Thread Management**: Designing thread lifecycle for SDR reader, audio playback, or signal processing workers in ECS context
- **Pipeline Architecture**: Structuring SDR processing chains (SDR → decimation → demodulation → audio output)
- **Broadcast Channels**: Implementing multi-consumer sample distribution (peak detection, quality analysis, demodulation)
- **Subprocess Delegation**: Managing subprocess for SDR hardware interface with clean IPC
- **Filter Design**: Calculating filter taps, decimation chains, and CPU optimization for FM/AM/SSB demodulation
- **Buffer Sizing**: Determining buffer sizes to prevent underruns while minimizing latency
- **Multi-Mode Demodulation**: Implementing extensible demodulator architecture (FM, AM, USB, LSB, digital modes)
- **State Machines**: Designing ECS-based state machines for pipeline lifecycle (idle → starting → running → stopping)

## Integration with ECS Architecture

This skill complements the `ecs-design` skill. When designing SDR pipelines:

1. **First consult `ecs-design`** for ECS fundamentals (EntityWorld, components, systems)
2. **Then apply `sdr-design`** for SDR-specific patterns (thread workers, broadcast channels, demodulation)

Key integration points:
- SDR worker threads communicate with ECS via channels (commands in, events out)
- ECS components track thread state and resources (handles, shutdown flags)
- ECS systems implement state machine transitions (spawning, monitoring, shutdown)
- Broadcast hub lives in ECS component, managed by systems

## Core Capabilities

### 1. Thread Lifecycle Management

**Reference**: `references/thread_lifecycle_patterns.md`

Design patterns for managing SDR worker threads in ECS context:

- **Long-lived vs Ephemeral threads**: When to keep threads running vs spawn/destroy per task
- **State machine design**: ECS components and systems for explicit state transitions
- **Shutdown coordination**: Atomic flags, non-blocking shutdown, avoiding deadlocks
- **Audio thread patterns**: Continuous vs shared audio thread architectures

**When to use**: Designing how audio threads, SDR reader threads, or processing workers should be managed throughout application lifecycle.

**Example scenario**: "Should I spawn a new audio thread every time I tune to a station, or keep one running throughout the app lifetime?"

**Asset**: `assets/state_machine_example.rs` - Complete working example of ECS-based thread lifecycle with state machine

### 2. Broadcast Channel Architecture

**Reference**: `references/broadcast_channels.md`

Patterns for distributing SDR samples to multiple concurrent consumers:

- **BroadcastHub pattern**: Independent channels per consumer with different buffer sizes
- **Backpressure handling**: What to do when consumers lag (block, drop, remove, adaptive decimation)
- **Warmup strategies**: Pre-filling buffers, discarding initial samples, state indicators
- **Arc-based sharing**: Minimizing clone overhead for high sample rates

**When to use**: Implementing the broadcast channel that sends SDR data to peak detection, signal quality analysis, and FM demodulation simultaneously.

**Example scenario**: "How do I cleanly broadcast SDR samples to multiple consumers with different processing speeds, and how do I warm up the pipeline so downstream consumers have good data?"

**Asset**: `assets/broadcast_setup_example.rs` - Multi-consumer broadcast channel with warmup support and example consumer threads

### 3. Subprocess Management

**Reference**: `references/subprocess_management.md`

Focused guidance for delegating SDR hardware interface to subprocess:

- **When to use subprocess vs threads**: Process isolation, crash recovery, language boundaries
- **IPC mechanisms**: Unix domain sockets (recommended), pipes, shared memory
- **Lifecycle management**: Startup sequence, graceful shutdown, restart on failure
- **Error recovery**: Handling crashes and hangs with watchdog timers

**When to use**: Delegating to and managing the subprocess that interfaces with SDR hardware.

**Example scenario**: "How should I structure the subprocess that reads from the SDR, and how do I handle crashes or hangs?"

### 4. FM Demodulation and Filter Design

**Reference**: `references/fm_demodulation.md`
**Script**: `scripts/filter_designer.py`

Deep dive into FM broadcast demodulation with CPU optimization:

- **Decimation chains**: Multi-stage vs single-stage, calculating optimal tap counts
- **Filter specification**: Passband, stopband, transition width, tap estimation formulas
- **Quadrature demodulation**: Phase discrimination algorithm and fast approximations
- **De-emphasis filtering**: 75μs time constant for US broadcasts
- **CPU optimization**: SIMD, polyphase decimation, avoiding underruns

**When to use**: Finding the right combination of filter taps, high-pass filtering, decimation to get good FM demodulated signal without excessive CPU, lag, or distortion.

**Example scenario**: "How do I design a decimation chain from 2.048 MHz SDR rate to 48 kHz audio without causing CPU issues or audio artifacts?"

**Usage**:
```bash
# Design single decimation filter
python3 scripts/filter_designer.py --mode decimation \
  --sample-rate 2048000 --decimation 4 --passband 100000 \
  --plot --rust --name fm_stage1

# Design complete decimation chain
python3 scripts/filter_designer.py --mode chain \
  --input-rate 2048000 --output-rate 48000 \
  --plot

# Design de-emphasis filter
python3 scripts/filter_designer.py --mode deemphasis \
  --sample-rate 48000 --time-constant 75 --plot
```

### 5. Multi-Mode Demodulation

**Reference**: `references/multi_mode_demod.md`
**Asset**: `assets/demodulator_trait.rs`

Extensible architecture for supporting multiple demodulation modes:

- **Trait-based design**: Pluggable demodulators via common interface
- **SSB demodulation**: Hilbert transform / phasing method for USB/LSB
- **AM demodulation**: Envelope detection with DC blocking
- **Digital modes**: PSK31, FT8 architecture patterns
- **Mode auto-detection**: Heuristic-based signal classification

**When to use**: Implementing demodulation for signals other than FM (upper/lower sideband, AM, digital modes).

**Example scenario**: "How do I demodulate SSB signals, and how should I structure the code to support adding new modes later?"

**Asset usage**: `assets/demodulator_trait.rs` provides complete trait definition, example implementations (FM, AM, SSB), registry pattern, and mode selection helpers. Copy and adapt for your application.

### 6. Buffer Sizing Strategy

**Reference**: `references/buffer_sizing.md`

Comprehensive guide to calculating buffer sizes:

- **Sizing formulas**: Latency-based, processing-time-based, jitter-based calculations
- **Tradeoffs**: Latency vs robustness, memory vs performance
- **Per-stage sizing**: SDR reader, decimation filters, audio output, broadcast channels
- **Problem diagnosis**: Underruns, overruns, excessive latency symptoms and solutions

**When to use**: Determining buffer sizes for any stage in the pipeline, diagnosing clicks/pops/lag issues.

**Example scenario**: "My audio has clicks and pops during scanning. How should I size my buffers?"

## Workflow Decision Tree

```
┌─ Designing new SDR feature? ────────────────────────────────┐
│                                                              │
├─ Thread lifecycle question? ─────────────────────────────────┤
│  → Read references/thread_lifecycle_patterns.md             │
│  → Review assets/state_machine_example.rs                   │
│  → Consider: Long-lived vs ephemeral? ECS state machine?    │
│                                                              │
├─ Multi-consumer data distribution? ──────────────────────────┤
│  → Read references/broadcast_channels.md                    │
│  → Review assets/broadcast_setup_example.rs                 │
│  → Consider: Buffer sizes per consumer? Backpressure policy?│
│                                                              │
├─ Subprocess for SDR hardware? ───────────────────────────────┤
│  → Read references/subprocess_management.md                 │
│  → Consider: IPC mechanism? Crash recovery? Log files?      │
│                                                              │
├─ Filter design / decimation chain? ──────────────────────────┤
│  → Read references/fm_demodulation.md                       │
│  → Run scripts/filter_designer.py for coefficient design    │
│  → Consider: Multi-stage decimation? CPU budget? Quality?   │
│                                                              │
├─ Adding new demodulation mode? ──────────────────────────────┤
│  → Read references/multi_mode_demod.md                      │
│  → Review assets/demodulator_trait.rs                       │
│  → Implement Demodulator trait for new mode                 │
│                                                              │
├─ Buffer sizing / underrun issues? ───────────────────────────┤
│  → Read references/buffer_sizing.md                         │
│  → Calculate sizes using formulas provided                  │
│  → Consider: Latency requirements? Processing jitter?       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Common Patterns

### Pattern: Spawning Audio Thread with ECS State Machine

1. Read `references/thread_lifecycle_patterns.md` for architecture
2. Review `assets/state_machine_example.rs` for implementation
3. Define state enum: `Idle → Starting → Running → Stopping → Failed`
4. Create ECS component with state, handle, channels, shutdown flag
5. Implement spawner system (Idle → Starting transition)
6. Implement monitor system (Starting → Running → Stopping transitions)
7. Implement shutdown system (Running → Stopping trigger)

### Pattern: Setting Up Broadcast Channel

1. Read `references/broadcast_channels.md` for options
2. Review `assets/broadcast_setup_example.rs` for implementation
3. Create BroadcastHub in ECS component
4. Add subscribers with appropriate buffer sizes (use `buffer_sizing.md` formulas)
5. Wrap receivers with WarmupConsumer for filter settling
6. Spawn consumer threads with receivers
7. Implement cleanup system to remove disconnected subscribers

### Pattern: Designing FM Decimation Chain

1. Read `references/fm_demodulation.md` for theory
2. Determine input rate (SDR) and output rate (audio): e.g., 2.048 MHz → 48 kHz
3. Run filter designer to see recommended chain:
   ```bash
   python3 scripts/filter_designer.py --mode chain --input-rate 2048000 --output-rate 48000
   ```
4. Review output: number of stages, tap counts, computational cost
5. If CPU too high: increase transition widths, reduce taps, adjust decimation factors
6. Generate Rust code for each stage:
   ```bash
   python3 scripts/filter_designer.py --mode decimation --sample-rate 2048000 \
     --decimation 4 --passband 100000 --rust --name fm_stage1
   ```
7. Integrate generated filter taps into your decimation blocks

### Pattern: Adding New Demodulation Mode

1. Read `references/multi_mode_demod.md` for architecture
2. Copy `assets/demodulator_trait.rs` as starting point
3. Implement `Demodulator` trait for new mode:
   - `name()`: Human-readable name
   - `bandwidth()`: Expected signal bandwidth
   - `required_sample_rate()`: Input sample rate needed
   - `demodulate()`: Core demodulation algorithm
   - `reset()`: Clear internal state
   - `metrics()`: Return SNR, signal strength, etc.
4. Register in DemodulatorRegistry
5. Add mode selection logic in `determine_mode_for_frequency()`
6. Test with real signals

## Reference Summary

All reference files provide deep technical detail. Consult them when designing or debugging specific aspects:

| Reference | Focus | Use When |
|-----------|-------|----------|
| `thread_lifecycle_patterns.md` | ECS thread management, state machines, shutdown | Designing worker thread lifecycle |
| `broadcast_channels.md` | Multi-consumer distribution, backpressure | Broadcasting samples to multiple consumers |
| `subprocess_management.md` | IPC, crash recovery, lifecycle | Delegating SDR hardware to subprocess |
| `fm_demodulation.md` | Filters, decimation, demod algorithms, CPU optimization | Implementing FM demodulation chain |
| `multi_mode_demod.md` | SSB, AM, digital mode demodulation, trait design | Adding non-FM demodulation modes |
| `buffer_sizing.md` | Latency, robustness, underrun prevention | Sizing buffers at any pipeline stage |

## Resources

This skill includes three types of bundled resources:

### scripts/

**`filter_designer.py`**: Python script for designing FIR filters and decimation chains.

Calculate optimal filter coefficients, estimate computational cost, generate Rust code, plot frequency responses. Essential tool for any DSP chain design.

**Dependencies**: scipy, numpy, matplotlib

**Usage examples**:
```bash
# Design decimation filter
python3 scripts/filter_designer.py --mode decimation --sample-rate 2048000 \
  --decimation 4 --passband 100000 --plot --rust

# Design Hilbert transformer for SSB
python3 scripts/filter_designer.py --mode hilbert --num-taps 65 --plot

# Design complete chain
python3 scripts/filter_designer.py --mode chain --input-rate 2048000 \
  --output-rate 48000
```

### references/

Detailed technical documentation covering:
- Thread lifecycle patterns with ECS integration
- Broadcast channel architectures and backpressure handling
- Subprocess management for SDR workers
- FM demodulation with deep DSP coverage
- Multi-mode demodulation (SSB, AM, digital)
- Buffer sizing formulas and strategies

Load these files into context when working on specific aspects of SDR pipeline design.

### assets/

Working Rust code examples:

- **`state_machine_example.rs`**: Complete ECS-based thread lifecycle implementation
- **`broadcast_setup_example.rs`**: Multi-consumer broadcast channel with warmup
- **`demodulator_trait.rs`**: Trait-based demodulator architecture with FM/AM/SSB examples

Copy and adapt these examples directly into your project.

## Integration with Other Skills

- **Use `ecs-design` first** for ECS architecture fundamentals, then `sdr-design` for SDR-specific patterns
- **Use `rust-analyzer-tools`** for refactoring demodulator implementations or finding references to filter coefficients
- **Use `systematic-debugging`** when diagnosing audio artifacts, underruns, or CPU issues in DSP chains

## Quick Start Examples

**Scenario 1**: "I need to design a thread to handle audio playback. Should it live throughout the app or be created per-station?"

→ Read `references/thread_lifecycle_patterns.md` section "Audio Thread Specific Patterns"
→ Review `assets/state_machine_example.rs`
→ Decision: Ephemeral per-station (created on tune, destroyed on tune-away)

**Scenario 2**: "I need to broadcast SDR samples to peak detection, signal quality, and FM demod. How?"

→ Read `references/broadcast_channels.md` section "Multi-Consumer Patterns"
→ Review `assets/broadcast_setup_example.rs`
→ Implement BroadcastHub with different buffer sizes per consumer

**Scenario 3**: "My FM audio is distorted and CPU usage is high."

→ Read `references/fm_demodulation.md` section "CPU Optimization Techniques"
→ Run `scripts/filter_designer.py --mode chain` to see current cost
→ Reduce tap counts or increase transition widths
→ Check buffer sizes in `references/buffer_sizing.md`

**Scenario 4**: "I want to add SSB demodulation support."

→ Read `references/multi_mode_demod.md` section "SSB Demodulation"
→ Copy `assets/demodulator_trait.rs` and implement SsbDemodulator
→ Run `scripts/filter_designer.py --mode hilbert` for Hilbert transformer coefficients

## Final Notes

This skill synthesizes patterns from:
- **GNU Radio**: Thread-per-block model, buffer architecture, scheduler design
- **RustRadio**: Async/await and iterator-based processing patterns
- **DSP literature**: Filter design formulas, decimation strategies, demodulation algorithms

Apply these proven patterns to your multithreaded Rust SDR application, adapting them to your ECS architecture and specific requirements.
