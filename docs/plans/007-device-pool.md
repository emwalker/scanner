# Plan 007: Tuner Pool with RAII

**Date**: October 2025
**Status**: In Progress (Phase 1d - Window Conversion)
**Dependencies**: ✅ Plan 005, ✅ Plan 006
**Related Plans**: Plan 004 (parent), Plan 011 (hot-plug)
**Enables**: Plans 009, 010

## Executive Summary

Implement dynamic tuner inventory using Rust's RAII pattern to guarantee proper resource management.

**Key features**:
- Tuners automatically return to pool when dropped (impossible to leak)
- Multi-tuner devices (e.g., RSPduo with 2 tuners) fully supported
- Controlled rollout via `PoolFilter` (single-tuner → multi-tuner)
- Shutdown-safe using `AtomicBool` and `try_lock()` pattern
- Activity tracking (replaces `ActiveTuners`)

**Design validation**: Architecture mirrors SDRTrunk's proven pool-based approach and follows Rust best practices (`object-pool`, `lockfree-object-pool`).

## Replaces ActiveTuners

The pool replaces the existing `ActiveTuners` struct with automatic state management:

| ActiveTuners | Tuner Pool |
|--------------|------------|
| Manual vector manipulation | Automatic RAII return on drop |
| `scanning: Vec<TunerId>` | `activity: TunerActivity::Scanning` |
| `listening: Vec<TunerId>` | `activity: TunerActivity::Listening` |
| Single tuner only | Full multi-tuner support |
| `send_active_tuners_update()` | `pool.status()` for TUI |

## Problem & Solution

**Current issues**:
- Manual device/tuner management (easy to forget to release)
- Multi-tuner devices underutilized (RSPduo has 2 tuners, we only use 1)
- No tuner-level tracking for UI
- Resource leaks possible

**Solution**: Self-managing pool with RAII guarantees:
```rust
let tuner = pool.acquire(requirements, TunerActivity::Scanning)?;
// ... use tuner ...
// ← Automatically returned to pool when out of scope!
```

## Design

See `src/pool/mod.rs` for full implementation.

### Key Types

**`Pool`**: Thread-safe tuner inventory with filtering
- Uses `Arc<Mutex<PoolInner>>` for shared ownership
- `PoolFilter` enables gradual single→multi-tuner rollout
- `AtomicBool` shutdown flag for lock-free checks

**`pool::Tuner`**: RAII wrapper auto-returns tuner on drop
- Wraps `Arc<Mutex<Box<dyn DeviceTrait>>>`
- Explicit methods (`tune()`, `set_gain()`) vs Deref to encapsulate channel logic
- `try_lock()` in Drop prevents shutdown deadlocks

**`TunerId`**: Composite of `device_id` + `channel_index`
- Uniquely identifies each tuner (RSPduo has 2: channel 0 and 1)

**`TunerActivity`**: Tracks what tuner is doing
- `Scanning | Listening | Other`
- Replaces `ActiveTuners.scanning/listening` vectors

**Lock ordering** (prevents deadlocks):
1. Device lock first
2. Pool lock second
3. Never hold both simultaneously

## Migration Strategy

**Phase 1** (current): Single-tuner mode
- Filter: `PoolFilter::new().with_driver("sdrplay").with_mode(TuningMode::SingleTuner)`
- Only one tuner allocatable at a time
- Benefits: RAII cleanup, capability matching, shutdown safety

**Phase 2** (future): Selective multi-tuner
- Relax filter constraints gradually
- Test with specific devices/backends

**Phase 3** (Plan 010): Full multi-tuner orchestration
- Concurrent scan + listen
- Priority management
- UI/UX for multi-tuner operations

## Benefits

**RAII guarantees**: Impossible to leak, compiler enforced, exception safe, scoped lifetime

**Multi-tuner support**: All device tuners exposed (RSPduo→2, RTL-SDR→1)

**Shutdown safety**: `try_lock()` in Drop, atomic shutdown flag, lock-free checks

**Thread-safe**: Standard `Arc<Mutex<>>` pattern with documented lock ordering

**Smart allocation**: Capability matching, best-fit selection, activity tracking

## Usage Examples

**Basic usage**:
```rust
let tuner = pool.acquire(&requirements, TunerActivity::Scanning)?;
let mut graph = Graph::new();
let stream = tuner.add_source_to_graph(&mut graph, 88.9e6, 2.4e6, 20.0)?;
// Automatically returned when tuner drops
```

**Multi-tuner** (RSPduo with 2 channels):
```rust
let tuner1 = pool.acquire(&scan_req, TunerActivity::Scanning)?;
let tuner2 = pool.acquire(&audio_req, TunerActivity::Listening)?;
// Both tuners from same device, running simultaneously
```

**Graceful degradation**:
```rust
let scan_tuner = pool.acquire(&scan_req, TunerActivity::Scanning)?;
if let Some(audio_tuner) = pool.try_acquire(&audio_req, TunerActivity::Listening) {
    spawn_audio_task(audio_tuner);  // Parallel operation
}
// Falls back to scan-only if no second tuner available
```

## File Structure

```
src/pool/
  mod.rs              # Core implementation
  types.rs            # TunerId, TaskRequirements, PoolStatus, TunerActivity
```

## Shutdown Safety ✅

**Problem**: Drop-in-drop chains and lock contention during teardown can cause deadlocks.

**Solution**: Multiple layers of protection (all implemented):

1. **`try_lock()` in Drop**: Non-blocking, gracefully handles lock contention
2. **`AtomicBool` shutdown flag**: Lock-free state checks before operations  
3. **Shutdown-aware operations**: All pool methods check shutdown flag first
4. **`Pool::is_shutdown()`**: Lock-free query for scanner loops
5. **Lock ordering**: Device lock → Pool lock (never both simultaneously)

**Key insight**: Use `AtomicBool` instead of bool in `PoolInner` to avoid needing pool lock during shutdown.

See CLAUDE.md shutdown safety section for detailed patterns and examples.

## Implementation Status

### ✅ Phase 1a-1c: Pool Infrastructure & Integration

**Core pool** (`src/pool/mod.rs`): Complete with RAII, shutdown safety, activity tracking, filtering (21 tests passing)

**MainThread integration** (`src/main_thread.rs`): Pool initialized with SingleTuner filter, populated on startup

**Critical discovery**: SDR hardware requires exclusive ownership - cannot run pool and legacy paths simultaneously. Must convert Window before using pool.

### 🚧 Phase 1d: Window Conversion (Current)

**Need**: Implement `pool::Segment` to bridge pool and Window
- Acquires `pool::Tuner` from pool
- Creates rustradio graph with `tuner.add_source_to_graph()`
- Sets up `broadcast::channel` for samples (see `SoapySdrManager` pattern)
- Implements `Segment` trait for Window compatibility
- RAII cleanup via Drop

**Tasks**:
1. ✅ Create `Window::for_station_with_pool()` constructor
2. ✅ **Implement `pool::Segment`** (~100 lines, reference: `src/soapy.rs:132-245`)
3. Update `scan_stations()` to use pool-based Window
4. Remove legacy device opening code

**Then Phase 1e**: Convert `scan_band()` similarly

### ⏳ Deferred Phases

**Phase 2 - Pool-Based Allocation**:
- Replace `ActiveTuners` with `pool.acquire()`
- TUI uses `pool.status()`

**Phase 3 - Discovery Integration**:
- Hot-plug support deferred to Plan 011
- Current: Initial device added at startup only

**Phase 4 - Multi-Tuner Enablement**:
- Change filter to `MultiTuner` mode
- Test concurrent scan + listen

## Next Steps

1. **Plan 009**: Task Abstraction
2. **Plan 010**: Multi-SDR Orchestration
3. **Plan 011**: Hot-Plug Support
