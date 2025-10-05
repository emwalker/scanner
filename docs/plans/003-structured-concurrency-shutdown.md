# Plan 003: Structured Concurrency for Shutdown Safety

**Date**: October 2025
**Status**: Implemented
**Related Issues**: `docs/issues/2025-10-shutdown/`
**Related Plans**: `docs/plans/001-async.md` (Hybrid Architecture - Future)

## Executive Summary

The shutdown bug we encountered (missing shutdown check in pause loop) revealed a key architectural weakness: it was too easy to forget shutdown checks when adding new code paths. This plan implements **structured concurrency** to make shutdown architecturally impossible to forget.

**Key Learning**: We don't need async/await to achieve structured concurrency. Using `tokio-util`'s `CancellationToken` with regular threads provides the same guarantees without async complexity.

## Problem Analysis

### What Made the Bug Easy to Introduce

The shutdown issue occurred because:

1. **Scattered Shutdown Checks**: Shutdown state was checked in 10+ places across the codebase
2. **Untracked Thread Spawns**: Using raw `std::thread::spawn` with manual joins
3. **No Architectural Guarantee**: Easy to forget shutdown checks in new code paths
4. **Complex Drop Order**: AudioSession required careful manual cleanup ordering

### Root Cause

This was an **architectural problem**: the architecture made it easy to introduce the bug and hard to detect it.

## Solution: Structured Concurrency WITHOUT Async

We implemented structured concurrency using synchronous Rust + `tokio-util`:

### 1. Centralized Shutdown Coordination

**Before** (scattered, error-prone):
```rust
// 10+ places checking shutdown_listener.is_triggered()
if self.shutdown_listener.is_triggered() {
    break;
}
```

**After** (centralized):
```rust
pub struct ShutdownCoordinator {
    token: CancellationToken,
    thread_handles: Mutex<Vec<JoinHandle<()>>>,
}

impl ShutdownCoordinator {
    pub fn shutdown(&self) {
        self.token.cancel(); // Propagates to ALL child tokens
    }

    pub fn wait(self) -> Result<()> {
        // Joins ALL threads, guaranteed
    }
}
```

### 2. Automatic Thread Tracking

**Before** (manual, forgettable):
```rust
let handle = std::thread::spawn(|| { /* work */ });
// ... somewhere else, might forget to join
```

**After** (automatic, impossible to forget):
```rust
coordinator.spawn_sdr_thread(|cancel_token| {
    while !cancel_token.is_cancelled() {
        // work - shutdown check is ALWAYS available
    }
});

// Later: shutdown ALL threads
coordinator.shutdown();
coordinator.wait() // All threads guaranteed joined
```

### 3. Single Source of Truth

One shutdown call propagates everywhere:

```rust
// In bin/scanner.rs
let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

ctrlc::set_handler(move || {
    signal_coordinator.shutdown(); // ← Single call
})?;

// Propagates to:
// - MainThread scan loop
// - All Window processing
// - Audio playback
// - TUI display
```

## Implementation Details

### Core Components

1. **ShutdownCoordinator** (`src/shutdown.rs`)
   - Tracks all spawned threads automatically
   - Provides cancellation tokens to all components
   - Guarantees all threads are joined on shutdown

2. **CancellationToken Integration**
   - Every component gets a token from coordinator
   - Checks `token.is_cancelled()` at strategic points
   - No scattered shutdown state

3. **State Machine Enforcement**
   - Existing `ScannerState` enum with exhaustive matching
   - Compiler enforces handling `ShuttingDown` state
   - Works perfectly with structured concurrency

### Key Design Decisions

**Why NOT async/await?**
- SDR I/O is blocking (SoapySDR C API)
- FFT/DSP is CPU-bound (not I/O-bound)
- No benefit from async overhead
- `CancellationToken` works in sync code

**What we DO use from tokio**:
- `tokio-util` only (not full `tokio`)
- `CancellationToken` - works without async runtime
- No `#[tokio::main]`, no async/await needed

**Result**: Structured concurrency guarantees with zero async complexity.

## Testing Strategy

### 1. Integration Tests (`tests/shutdown_test.rs`)

Comprehensive shutdown scenarios:
- Shutdown while paused
- Shutdown while scanning
- Shutdown during window processing
- Immediate shutdown (before work starts)
- Signal propagation (multiple tokens)
- Concurrent shutdown detection
- Cleanup ordering verification

All tests use regular `#[test]` (not async) with explicit timeouts.

### 2. Property-Based Tests (proptest)

Random shutdown timing tests:
- Arbitrary shutdown delays (0-500ms)
- Different scanner states
- Concurrent thread detection
- Window processing interruption

Tests 20 random scenarios per run to catch edge cases.

### 3. State Transition Tests

Using existing `ScannerState` enum:
- Verify shutdown works in all states
- Test pause/resume with shutdown
- Validate state machine transitions

## Migration Results

### Changes Made

1. **Added Dependencies** (Cargo.toml)
   ```toml
   tokio-util = { version = "0.7", features = ["rt"] }
   ```
   Note: `tokio` was already a dependency (for other features)

2. **Created ShutdownCoordinator** (src/shutdown.rs)
   - Central coordination
   - Automatic thread tracking
   - Unified cancellation

3. **Updated Components**
   - `MainThread`: Uses coordinator for all operations
   - `AudioSession`: Spawns threads via coordinator
   - `Window`: Uses cancellation tokens
   - `TuiProgressDisplay`: Checks token for shutdown
   - `bin/scanner.rs`: Creates and uses coordinator

4. **Removed `triggered` Dependency**
   - Completely replaced with `CancellationToken`
   - Simpler API, better integration

5. **Added Tests**
   - 16 shutdown integration tests
   - Property-based timing tests
   - All passing

### Test Results

```
Running 16 shutdown tests... ok (5.68s)
Running 163 library tests... ok (11.24s)
```

Zero regressions. All tests passing.

## Benefits Achieved

### Immediate Benefits

✅ **Impossible to forget shutdown**: Coordinator tracks everything automatically
✅ **Single shutdown call**: `coordinator.shutdown()` stops everything
✅ **Guaranteed cleanup**: `coordinator.wait()` ensures all threads joined
✅ **No async complexity**: Pure synchronous code
✅ **Compiler enforcement**: State machine ensures exhaustive handling

### Code Quality Improvements

✅ **Simpler drop order**: Coordinator handles cleanup
✅ **Fewer shutdown checks**: 10+ scattered checks → 2 strategic checks
✅ **Better error messages**: Centralized error handling
✅ **Self-documenting**: Coordinator makes threading explicit

### Multi-SDR Ready (Future)

When adding multiple SDRs:
```rust
for sdr in sdrs {
    coordinator.spawn_sdr_thread(|cancel_token| {
        run_sdr(sdr, cancel_token)
    });
}

// Shutdown ALL SDRs with single call
coordinator.shutdown();
coordinator.wait() // All guaranteed stopped
```

## Compatibility with Plan 001 (Hybrid Architecture)

This implementation is **100% compatible** with Plan 001's hybrid approach:

### What Plan 001 Proposes (Future)
- Dedicated threads for SDR I/O ✅ (we have this)
- Rayon for CPU-bound work
- Tokio async for multi-device coordination

### What Plan 003 Provides (Implemented)
- ShutdownCoordinator for thread tracking ✅
- CancellationToken for unified shutdown ✅
- State machine for compiler enforcement ✅
- Works in sync OR async contexts ✅

### Integration Path (When Plan 001 is Implemented)

```rust
struct HybridMultiSdrScanner {
    // SDR I/O threads (blocking)
    shutdown_coordinator: Arc<ShutdownCoordinator>,

    // Async coordination (future)
    tokio_runtime: tokio::Runtime,
    task_tracker: TaskTracker,

    // Works in both contexts
    state_machine: ScannerState,
}
```

The `ShutdownCoordinator` we built works perfectly with or without async.

## Lessons Learned

### Async is NOT Required for Structured Concurrency

**Before this work**: Assumed structured concurrency = async/await
**After this work**: `CancellationToken` provides structured concurrency in sync code

**Key Insight**: Structured concurrency is about **tracking and coordinating** concurrent work, not about async vs threads.

### CancellationToken Works Everywhere

- Works in blocking threads ✅
- Works in async tasks ✅
- Works with existing state machines ✅
- No runtime overhead ✅

### Incremental Adoption Works

We migrated incrementally:
1. Added coordinator alongside `triggered::Listener`
2. Migrated components one by one
3. Removed old dependency when done
4. Zero downtime, all tests passing

## Future Enhancements

### When Adding Multi-SDR (Plan 001)
- Coordinator already supports multiple threads
- Just spawn more threads via `spawn_sdr_thread()`
- Shutdown automatically coordinates all SDRs

### If Adding Async Coordination (Plan 001 Phase 3)
- Add `TaskTracker` for async tasks
- Keep `ShutdownCoordinator` for SDR threads
- Use same `CancellationToken` for both
- State machine works in both contexts

### Advanced Testing (Optional)
- Loom tests for concurrency validation
- Chaos testing with fault injection
- Performance benchmarks for shutdown latency

## References

### Documentation
- [tokio-util CancellationToken](https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html)
- [tokio-util TaskTracker](https://docs.rs/tokio-util/latest/tokio_util/task/struct.TaskTracker.html)
- [Tokio Graceful Shutdown](https://tokio.rs/tokio/topics/shutdown)

### Related Documents
- [Shutdown Issue Root Cause](../issues/2025-10-shutdown/01-shutting-down.md)
- [Shutdown Fix Plan](../issues/2025-10-shutdown/02-fix-plan.md)
- [Implementation Checklist](../issues/2025-10-shutdown/03-structured-concurrency-implementation.md)
- [Plan 001: Hybrid Architecture](001-async.md)

## Conclusion

We successfully implemented structured concurrency for shutdown safety **without async/await complexity**. The solution:

1. **Makes shutdown architecturally impossible to forget**
2. **Works with existing synchronous code**
3. **Prepares for future multi-SDR support**
4. **Maintains compatibility with Plan 001**
5. **Adds zero async overhead**

**Status**: ✅ Implemented and tested. All 179 tests passing.
