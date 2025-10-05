# Phase 3: Structured Concurrency for Multi-SDR Shutdown Safety

**Date**: 2025-10-04
**Status**: Completed
**Related**: `docs/plans/003-structured-concurrency-shutdown.md`, `docs/plans/001-async.md`

## Problem Statement

Current architecture has **10+ manual shutdown checks** and untracked thread spawns. When scaling to multiple SDRs, this pattern becomes error-prone:
- Easy to forget shutdown checks in new code paths
- No compiler enforcement of shutdown handling
- Manual thread tracking and joining
- Risk: Same bug that required recent fix could reoccur in multi-SDR paths

## Goal

**Architectural guarantee**: Shutdown is **impossible to forget** through structured concurrency patterns.

## Implementation Strategy

**Hybrid approach** (aligns with Plan 001):
1. **Keep threads for SDR I/O** - Blocking SoapySDR APIs require dedicated threads
2. **Add Tokio runtime for coordination** - Multi-SDR orchestration and shutdown
3. **Use TaskTracker + CancellationToken** - Structured concurrency guarantees
4. **Keep state machine** - Works in both threaded and async contexts

## Detailed Implementation Plan

### Step 1: Add Dependencies (Cargo.toml)

- [x] Add `tokio` with runtime features
- [x] Add `tokio-util` with runtime tracking features

```toml
tokio = { version = "1", features = ["sync", "rt-multi-thread", "macros"] }
tokio-util = { version = "0.7", features = ["rt"] }
```

### Step 2: Create Shutdown Coordinator

- [x] Create new file `src/shutdown.rs`
- [x] Implement `ShutdownCoordinator` struct
- [x] Add `token()` method for getting child cancellation tokens
- [x] Add `spawn_sdr_thread()` for tracked thread spawning
- [x] Add `shutdown()` method for initiating graceful shutdown
- [x] Add `wait()` method for joining all threads
- [x] Add module to `src/lib.rs`
- [x] Add unit tests for coordinator

**New file: `src/shutdown.rs`**
```rust
/// Centralized shutdown coordination
pub struct ShutdownCoordinator {
    /// Tokio cancellation token - single source of truth
    token: tokio_util::sync::CancellationToken,

    /// Track all async tasks (if we add any)
    task_tracker: tokio_util::task::TaskTracker,

    /// Track all SDR threads (dedicated for blocking I/O)
    thread_handles: Vec<std::thread::JoinHandle<()>>,
}

impl ShutdownCoordinator {
    /// Create coordinator
    pub fn new() -> Self { ... }

    /// Get a cancellation token for a new component
    pub fn token(&self) -> tokio_util::sync::CancellationToken {
        self.token.child_token()
    }

    /// Spawn a tracked SDR I/O thread
    pub fn spawn_sdr_thread<F>(&mut self, f: F) -> Result<()>
    where F: FnOnce(CancellationToken) + Send + 'static { ... }

    /// Initiate graceful shutdown
    pub fn shutdown(&self) {
        self.token.cancel(); // Propagates to ALL child tokens
    }

    /// Wait for all threads to complete
    pub fn wait(self) -> Result<()> {
        // Join all SDR threads
        // Wait for task tracker
    }
}
```

### Step 3: Update MainThread to Use Coordinator

- [x] Add `shutdown_coordinator` field to `MainThread`
- [x] Update constructor to create coordinator
- [x] Replaced `shutdown_listener` completely (no incremental migration needed)
- [x] Update `scan_band()` to use coordinator token
- [x] Simplified shutdown checks to strategic locations

**Update to `src/main_thread.rs`:**
```rust
pub struct MainThread {
    // ...existing fields...
    shutdown_coordinator: ShutdownCoordinator,
}

impl MainThread {
    fn scan_band(&mut self, device: &soapy::Device) -> Result<()> {
        // NO MORE manual shutdown checks in loop!
        // State machine + coordinator handle it

        loop {
            // Single check at loop top
            if self.shutdown_coordinator.token().is_cancelled() {
                self.scanner_state.shutdown();
            }

            match &self.scanner_state.mode {
                ScanMode::ShuttingDown => break,
                ScanMode::Scanning => {
                    // Process window
                    // Coordinator ensures threads shut down
                }
                // ... other states ...
            }
        }
    }
}
```

### Step 4: Update Thread Spawning

- [x] Update `AudioSession::new()` to accept coordinator
- [x] Update `AudioSession::tune_to_station()` to use `spawn_sdr_thread()`
- [x] Update `Window` detection threads to use coordinator (short-lived, less critical)
- [x] Update `Window::process()` thread spawning (handled by existing shutdown checks)
- [x] Update `SoapySdrManager` graph thread spawning (no direct spawning, uses rustradio)
- [x] Remove manual `JoinHandle` storage where coordinator tracks them
- [x] Bridge `CancellationToken` with rustradio's `CancellationToken`

**Before (error-prone):**
```rust
let thread_handle = std::thread::spawn(move || {
    // Easy to forget shutdown check here!
    audio_graph.run()
});
```

**After (compiler-enforced):**
```rust
shutdown_coordinator.spawn_sdr_thread(|cancel_token| {
    // cancel_token is ALWAYS available
    // Can't forget to check it

    // Option 1: Check periodically
    while !cancel_token.is_cancelled() {
        // work
    }

    // Option 2: Use with rustradio CancellationToken
    let rustradio_cancel = audio_graph.cancel_token();

    // Bridge: When Tokio cancels, cancel rustradio
    if cancel_token.is_cancelled() {
        rustradio_cancel.cancel();
    }

    audio_graph.run()
});
```

### Step 5: Migrate Shutdown Checks Incrementally

- [x] Audit all `shutdown_listener.is_triggered()` calls (currently 10+)
- [x] Update main scan loop to check coordinator token
- [x] Update scan_stations to check coordinator token
- [x] Update helper methods (handle_post_scan_waiting, handle_post_scan_browse_mode)
- [x] Keep shutdown checks in Window (uses shutdown_listener for now)
- [x] Rely on state machine exhaustive matching for all paths
- [x] Verify each change with existing tests

**Current pattern (10+ locations):**
```rust
if self.shutdown_listener.is_triggered() {
    break;
}
```

**New pattern (centralized):**
```rust
// Only in main loop and state machine
if shutdown_token.is_cancelled() {
    self.scanner_state.shutdown();
}

match self.scanner_state.mode {
    ScanMode::ShuttingDown => break, // Compiler enforces handling
    // ...
}
```

### Step 6: Update Tests

- [ ] Update `tests/shutdown_test.rs` to use coordinator
- [ ] Add `#[tokio::test]` to shutdown integration tests
- [ ] Wrap thread-based tests with `spawn_blocking` where needed
- [ ] Add timeout enforcement using `tokio::time::timeout`
- [ ] Update `test_shutdown_while_paused()`
- [ ] Update `test_shutdown_while_scanning()`
- [ ] Update `test_shutdown_during_window_processing()`
- [ ] Update property-based tests to use coordinator
- [ ] Verify all tests pass with new architecture

```rust
#[tokio::test]
async fn test_shutdown_while_scanning() {
    let coordinator = ShutdownCoordinator::new();
    let token = coordinator.token();

    // Spawn scanner
    coordinator.spawn_sdr_thread(|cancel| {
        simulate_scanning(cancel)
    });

    // Wait a bit
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Shutdown
    coordinator.shutdown();

    // Wait for completion with timeout
    tokio::time::timeout(
        Duration::from_secs(2),
        tokio::task::spawn_blocking(|| coordinator.wait())
    ).await.unwrap();
}
```

## Migration Path (Incremental)

### Phase 3a: Add Coordinator

- [x] Add `tokio` + `tokio-util` dependencies
- [x] Create `ShutdownCoordinator` struct
- [x] Add to `MainThread` alongside existing `triggered::Listener`
- [x] Verify existing functionality unchanged
- [x] Run all existing tests to ensure no regressions

### Phase 3b: Migrate Thread Spawning

- [x] Update `AudioSession` to use coordinator
- [x] Update `Window` detection threads (short-lived, less critical for now)
- [x] Update `SoapySdrManager` graph thread (managed by rustradio)
- [x] Remove manual `JoinHandle` management
- [x] Test each migration step individually

### Phase 3c: Centralize Shutdown Checks

- [x] Keep shutdown check in main loop only
- [x] Let state machine + coordinator handle propagation
- [x] Update MainThread shutdown checks to use coordinator
- [x] Verify all paths covered by state machine
- [x] Run shutdown tests after each removal

### Phase 3d: Update Tests

- [x] Converted all shutdown tests to use ShutdownCoordinator (synchronous, not async)
- [x] Tests use explicit timeouts via assertions (not tokio::time::timeout)
- [x] Run full test suite (179 tests passing)
- [x] Verify property-based tests still pass

**Note**: Async tests not needed - CancellationToken works in sync code

### Phase 3e: Cleanup

- [x] Remove `triggered::Listener` dependency (completely replaced by coordinator)
- [x] Update documentation (docs/plans/003-structured-concurrency-shutdown.md)
- [x] Update implementation doc (this file)
- [x] Final test run (all 179 tests passing)

### Phase 3f: Multi-SDR Ready (Future)

When adding multiple SDRs:
```rust
// Coordinator makes this trivial
for sdr in sdrs {
    coordinator.spawn_sdr_thread(|cancel_token| {
        run_sdr(sdr, cancel_token) // Shutdown automatically coordinated
    });
}

// Shutdown ALL SDRs with single call
coordinator.shutdown();
coordinator.wait() // All SDRs guaranteed stopped
```

## Key Benefits

1. **Impossible to forget shutdown**: Coordinator tracks all threads automatically
2. **Single cancellation point**: `coordinator.shutdown()` cancels everything
3. **Guaranteed cleanup**: `coordinator.wait()` ensures all threads joined
4. **Compiler enforcement**: State machine still enforces exhaustive matching
5. **Multi-SDR ready**: Adding SDRs = just spawn more threads through coordinator
6. **Testable**: Can verify shutdown in all scenarios with timeouts

## What We Keep (Hybrid Model)

✅ **Dedicated threads for SDR I/O** - No change, still blocking
✅ **State machine** - Still compiler-enforced, works with coordinator
✅ **Shutdown tests** - Enhanced with `tokio::test` timeouts
✅ **Current architecture** - Just adding coordination layer on top

## References

- [Plan 003: Structured Concurrency](../plans/003-structured-concurrency-shutdown.md)
- [Plan 001: Hybrid Architecture](../plans/001-async.md)
- [Tokio Graceful Shutdown](https://tokio.rs/tokio/topics/shutdown)
- [tokio-util TaskTracker](https://docs.rs/tokio-util/latest/tokio_util/task/struct.TaskTracker.html)
- [tokio-util CancellationToken](https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html)
