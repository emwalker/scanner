# Plan 003: Structured Concurrency and Shutdown Testing

**Date**: October 2025
**Status**: Proposal
**Related Issues**: `docs/issues/2025-10-shutdown/`
**Related Plans**: `docs/plans/001-async.md` (Hybrid Architecture)

## Executive Summary

The shutdown bug we encountered (missing shutdown check in pause loop) revealed architectural weaknesses in our concurrency model. This document proposes architectural improvements and testing strategies to prevent similar issues and make shutdown handling more robust and maintainable.

**Compatibility with Plan 001 (Hybrid Architecture)**: This plan is **highly compatible** (~95%) with the hybrid async/thread architecture proposed in `001-async.md`. The state machine, structured concurrency patterns, and testing strategies work seamlessly with the hybrid model:
- State machines work in both threaded and async contexts
- CancellationToken/TaskTracker integrate with Tokio coordination layer
- Testing strategies (Loom, proptest) validate both thread and async code
- Only constraint: Keep SDR I/O in dedicated threads (both plans agree)

## Problem Analysis

### What Made This Bug Hard to Find

The shutdown issue was difficult to debug because:

1. **Scattered Shutdown Checks**: Shutdown state was checked in multiple places (window loop, candidate processing, audio playback) but not in the pause loop. There was no architectural guarantee that all code paths checked for shutdown.

2. **Unstructured Concurrency**: Using raw `std::thread::spawn` with manual joins meant:
   - No automatic tracking of spawned threads
   - Easy to forget shutdown checks in new code paths
   - Complex drop order dependencies (AudioSession)
   - Race conditions between thread completion and resource cleanup

3. **Implicit State Machine**: The scanner has multiple states (Running, Paused, Processing Window, Playing Audio) but this was implicit in boolean flags and control flow, not explicit in the type system.

4. **No Systematic Testing**: We had no tests for shutdown scenarios, so the bug only appeared during manual testing.

### Root Cause Classification

This was fundamentally an **architectural problem**, not just a bug:
- The architecture made it easy to forget shutdown checks
- The architecture made shutdown paths hard to test
- The architecture required manual coordination of multiple threads and resources

## Architectural Improvements

### 1. Structured Concurrency with Tokio

**Current State**: Unstructured concurrency with `std::thread::spawn`
```rust
// Current pattern - easy to leak threads
let handle = std::thread::spawn(|| { /* work */ });
// ... somewhere else, might forget to join
```

**Proposed**: Tokio structured concurrency
```rust
use tokio_util::task::TaskTracker;
use tokio_util::sync::CancellationToken;

let tracker = TaskTracker::new();
let token = CancellationToken::new();

// All tasks are tracked automatically
tracker.spawn(async move {
    tokio::select! {
        _ = token.cancelled() => {
            // Shutdown detected - cleanup
            return;
        }
        result = do_work() => {
            // Normal completion
        }
    }
});

// Shutdown: cancel all tasks and wait
token.cancel();
tracker.close();
tracker.wait().await; // Guaranteed: all tasks finished
```

**Benefits**:
- **Automatic tracking**: Can't leak threads
- **Unified cancellation**: One token propagates to all tasks
- **Natural shutdown points**: Every `.await` is a potential cancellation point
- **Scope-based cleanup**: When tracker is dropped, all tasks are awaited

### 2. Explicit State Machine

**Current State**: Implicit state in boolean flags
```rust
let mut paused = false;
if paused {
    // Easy to forget shutdown check here
    std::thread::sleep(100ms);
    continue;
}
```

**Proposed**: Explicit state enum with exhaustive matching
```rust
enum ScannerState {
    Running,
    Paused { audio_session: AudioSession },
    ProcessingWindow { window_id: usize },
    ShuttingDown,
}

match state {
    ScannerState::Running => {
        if shutdown.is_triggered() {
            state = ScannerState::ShuttingDown;
        }
        // process window
    }
    ScannerState::Paused { .. } => {
        if shutdown.is_triggered() {  // ← Compiler forces us to handle this
            state = ScannerState::ShuttingDown;
        }
        // handle pause
    }
    ScannerState::ShuttingDown => {
        // cleanup and exit
        break;
    }
}
```

**Benefits**:
- **Exhaustive matching**: Compiler enforces handling all states
- **Clear transitions**: Can see all possible state changes
- **Testable**: Can unit test state transitions
- **Self-documenting**: State machine is explicit in code

### 3. Async/Await Architecture

**Current State**: Thread-based with manual synchronization
- Hard to coordinate shutdown across threads
- Complex drop order dependencies
- No natural cancellation points

**Proposed**: Async/await with Tokio
```rust
async fn run_scanner(
    stations: Vec<f64>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut state = ScannerState::Running;

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                // Shutdown always handled, at every await point
                state = ScannerState::ShuttingDown;
            }
            result = process_next_window(&mut state) => {
                // Normal processing
            }
        }

        if matches!(state, ScannerState::ShuttingDown) {
            break;
        }
    }

    // Cleanup happens here, guaranteed
    Ok(())
}
```

**Benefits**:
- **Natural cancellation**: Every `.await` checks for cancellation
- **No drop order issues**: Async runtime handles cleanup order
- **Better error propagation**: `?` operator works across await boundaries
- **Easier to test**: `#[tokio::test]` provides test runtime

### 4. Resource Lifecycle Management

**Current Issue**: AudioSession drop order was critical
```rust
// Current: Manual drop order to avoid use-after-free
if let Some((cancel, handle)) = graph.take() {
    cancel.cancel();
    handle.join();  // MUST happen before segment drop
}
drop(segment);
```

**Proposed**: RAII with async drop (using async-dropper crate)
```rust
struct AudioSession {
    graph: AsyncDrop<AudioGraph>,
    segment: AsyncDrop<SdrSegment>,
}

// Drop order is automatic and safe:
// 1. graph is dropped (awaited) first
// 2. segment is dropped second
// Compiler enforces correct order via struct field order
```

## Testing Strategies

### 1. Loom for Concurrency Testing

**Tool**: [Loom](https://github.com/tokio-rs/loom) - Deterministic concurrency testing

```rust
#[cfg(loom)]
mod loom_tests {
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;

    #[test]
    fn test_shutdown_all_orderings() {
        loom::model(|| {
            let shutdown = Arc::new(AtomicBool::new(false));
            let paused = Arc::new(AtomicBool::new(false));

            // Main thread
            let s1 = shutdown.clone();
            let p1 = paused.clone();
            let t1 = thread::spawn(move || {
                loop {
                    if p1.load(Ordering::SeqCst) {
                        if s1.load(Ordering::SeqCst) {
                            break;  // ← Must detect shutdown when paused
                        }
                        thread::yield_now();
                        continue;
                    }
                    // work
                }
            });

            // Control thread
            paused.store(true, Ordering::SeqCst);
            shutdown.store(true, Ordering::SeqCst);

            t1.join().unwrap();
        });
    }
}
```

**What it tests**:
- All possible thread interleavings
- Race conditions between pause and shutdown
- Memory ordering issues
- Deadlocks and livelocks

**Configuration**:
```bash
RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=2 cargo test --test loom_tests
```

### 2. Integration Tests for Each Shutdown Path

**Test Suite Structure**:
```rust
mod shutdown_tests {
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_shutdown_while_paused() {
        let scanner = Scanner::new(config);

        // Start scanning
        let handle = tokio::spawn(scanner.run());

        // Pause
        scanner.send_command(Command::Pause).await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Trigger shutdown while paused
        scanner.shutdown().await;

        // Assert: exits within 1 second
        let result = timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "Scanner should exit promptly when shutdown while paused");
    }

    #[tokio::test]
    async fn test_shutdown_during_window_processing() {
        let scanner = Scanner::new(config);
        let handle = tokio::spawn(scanner.run());

        // Let it start processing
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Shutdown mid-window
        scanner.shutdown().await;

        let result = timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "Scanner should exit during window processing");
    }

    #[tokio::test]
    async fn test_shutdown_during_audio_playback() {
        let scanner = Scanner::new(config);
        scanner.send_command(Command::Pause).await;
        scanner.tune_to(88.9e6).await;

        // Audio is playing
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shutdown during audio
        scanner.shutdown().await;

        let result = timeout(Duration::from_secs(1), scanner.wait()).await;
        assert!(result.is_ok(), "Scanner should exit during audio playback");
    }

    #[tokio::test]
    async fn test_double_shutdown() {
        let scanner = Scanner::new(config);

        // First shutdown - graceful
        scanner.shutdown().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Second shutdown - force
        scanner.force_shutdown().await;

        let result = timeout(Duration::from_millis(500), scanner.wait()).await;
        assert!(result.is_ok(), "Force shutdown should exit immediately");
    }
}
```

### 3. Property-Based Testing with Proptest

**Tool**: [proptest](https://github.com/proptest-rs/proptest) - Property-based testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn shutdown_at_arbitrary_time(
        shutdown_delay_ms in 0u64..1000,
        in_pause_state in proptest::bool::ANY,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let scanner = Scanner::new(config);
            let handle = tokio::spawn(scanner.run());

            if in_pause_state {
                scanner.send_command(Command::Pause).await;
            }

            tokio::time::sleep(Duration::from_millis(shutdown_delay_ms)).await;
            scanner.shutdown().await;

            // Property: ALWAYS exits within timeout, regardless of timing
            let result = timeout(Duration::from_secs(2), handle).await;
            prop_assert!(result.is_ok(), "Failed to shutdown after {}ms", shutdown_delay_ms);
        });
    }
}
```

**What it tests**:
- Shutdown at random points in execution
- Different scanner states
- Edge cases we didn't think of

### 4. Timeout Enforcement

**Pattern**: Every test has a timeout
```rust
// Option 1: Test attribute (requires test framework support)
#[tokio::test]
#[timeout(5000)]  // 5 seconds max
async fn test_shutdown() {
    // If this hangs, test fails
}

// Option 2: Explicit timeout
#[tokio::test]
async fn test_shutdown() {
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        actual_test_logic()
    ).await;

    assert!(result.is_ok(), "Test timed out - shutdown hung");
}

// Option 3: Test helper
async fn with_timeout<F, T>(timeout_secs: u64, f: F) -> T
where
    F: Future<Output = T>,
{
    tokio::time::timeout(Duration::from_secs(timeout_secs), f)
        .await
        .expect("Test timed out")
}

#[tokio::test]
async fn test_shutdown() {
    with_timeout(5, async {
        // test logic
    }).await;
}
```

## Compatibility with Plan 001: Hybrid Architecture

This plan is **highly compatible** with the hybrid async/thread architecture proposed in `docs/plans/001-async.md`. The two plans are complementary:

### What Plan 001 Provides (Architecture)
- **SDR I/O**: Dedicated threads for blocking SoapySDR APIs
- **CPU-bound work**: Rayon thread pool for FFT/DSP
- **Coordination**: Tokio async runtime for multi-device control
- **Focus**: Multi-SDR support and performance

### What Plan 003 Provides (Correctness & Testing)
- **Shutdown handling**: CancellationToken, TaskTracker
- **State management**: Explicit state machine
- **Testing**: Loom, property-based tests, integration tests
- **Focus**: Graceful shutdown and bug prevention

### Combined Architecture

```rust
// Hybrid multi-SDR scanner with structured concurrency:
struct HybridMultiSdrScanner {
    // From 001: Dedicated threads for SDR I/O (blocking APIs)
    sdr_threads: Vec<std::thread::JoinHandle<()>>,

    // From 001: Rayon for CPU-bound work
    rayon_pool: rayon::ThreadPool,

    // From 001: Tokio async runtime for coordination
    runtime: tokio::Runtime,

    // From 003: Structured concurrency for async tasks
    task_tracker: TaskTracker,
    shutdown_token: CancellationToken,

    // From 003: Explicit state machine
    state: Arc<Mutex<ScannerState>>,
}

// Async coordination with structured concurrency:
async fn coordinate_sdrs(
    sdr_receivers: Vec<tokio::sync::mpsc::Receiver<Complex>>,
    shutdown: CancellationToken,  // ← From 003
    tracker: TaskTracker,          // ← From 003
) {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                break;  // Graceful shutdown
            }
            Some(samples_a) = sdr_receivers[0].recv() => {
                // Spawn tracked task - can't leak
                tracker.spawn(async move {
                    // Offload CPU work to Rayon (from 001)
                    rayon_pool.install(|| compute_fft(samples_a))
                });
            }
            Some(samples_b) = sdr_receivers[1].recv() => {
                // Cross-correlate for direction finding
            }
        }
    }

    // Wait for all tasks to complete
    tracker.wait().await;
}
```

### State Machine Works in Both Contexts

The explicit state machine from this plan works seamlessly in both threaded and async code:

```rust
enum ScannerState {
    Running,
    Paused { audio_session: AudioSession },
    ProcessingWindow { window_id: usize },
    ShuttingDown,
}

// In threaded context (current):
loop {
    match state {
        ScannerState::Paused { .. } => {
            if shutdown.is_triggered() {  // ← Compiler enforces
                state = ScannerState::ShuttingDown;
            }
        }
        // ...
    }
}

// In async context (future):
async fn run_scanner(state: &mut ScannerState) {
    match state {
        ScannerState::Running => { /* async work */ }
        ScannerState::Paused { .. } => {
            // Same state machine, async context
        }
        // Compiler still enforces exhaustive matching
    }
}
```

### Testing Validates Entire Stack

The testing strategies from this plan validate both threaded and async layers:

```rust
// Loom tests work for both:
#[cfg(loom)]
#[test]
fn test_hybrid_shutdown() {
    loom::model(|| {
        // Test thread coordination (SDR I/O layer)
        let sdr_thread = loom::thread::spawn(|| { /* blocking I/O */ });

        // Test async coordination (Tokio layer)
        // Loom supports both models

        // Test shutdown propagation across both layers
    });
}

// Property-based tests work regardless of concurrency model:
proptest! {
    fn shutdown_at_arbitrary_time_hybrid(delay_ms: u64) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // Start hybrid architecture
            let sdr_thread = std::thread::spawn(|| { /* SDR I/O */ });
            let async_coord = tokio::spawn(async { /* coordination */ });

            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            shutdown_token.cancel();

            // Both layers should exit cleanly
        });
    }
}
```

### Where You Must Choose: SDR I/O Layer Only

There is **one constraint** both plans agree on:

**SDR I/O must use dedicated threads** (not async `spawn_blocking`)

From Plan 001:
> "DON'T: Use `spawn_blocking` for CPU-intensive work (can exhaust thread pool)"
> "DO: Use dedicated threads for long-running sync work"

This plan's async recommendations apply to the **coordination layer only**, not SDR I/O. The layers are:

| Layer | Concurrency Model | From Plan | Why |
|-------|------------------|-----------|-----|
| **SDR I/O** | Dedicated threads | 001 | Blocking C APIs |
| **CPU/DSP** | Rayon | 001 | CPU-bound parallelism |
| **Coordination** | Tokio + TaskTracker | 001 + 003 | Elegant control flow + safety |
| **State Logic** | State Machine | 003 | Works in any context |
| **Testing** | Loom + proptest | 003 | Validates all layers |

### Incremental Adoption Path

You can adopt both plans **incrementally and independently**:

**Phase 1: State Machine (This Plan, Immediate)**
- Add explicit `ScannerState` enum
- Works with current threaded code
- No async required

**Phase 2: Multi-SDR with Threads (Plan 001, Phase 1)**
- Add more SDR threads (001 approach)
- Keep state machine from this plan
- Use `CancellationToken` for shutdown

**Phase 3: Add Async Coordination (Plan 001, Phase 3)**
- Add Tokio runtime for coordination
- Use `TaskTracker` from this plan (003)
- Keep SDR threads, keep state machine

**Phase 4: Testing (This Plan, Phase 1)**
- Add Loom, proptest, integration tests
- Tests validate entire hybrid stack

### Benefits of Combined Approach

**Performance** (from 001):
- ✅ Efficient SDR I/O (threads)
- ✅ CPU parallelism (Rayon)
- ✅ Elegant multi-device coordination (Tokio)

**Correctness** (from 003):
- ✅ No task leaks (TaskTracker)
- ✅ Guaranteed shutdown (CancellationToken)
- ✅ Compiler-enforced state (state machine)
- ✅ Comprehensive testing (Loom, proptest)

**Together**:
- ✅ Best of both worlds: Fast AND correct
- ✅ Clear separation: Threads for I/O, async for coordination, state machine for logic
- ✅ Testable at all layers
- ✅ Incremental adoption

### Summary: Complementary Plans

Think of these plans as:
- **Plan 001**: The **architecture** (structure of concurrency)
- **Plan 003**: The **quality assurance** (correctness and testing)

**Compatibility**: ~95%
- State machine: 100% compatible
- Structured concurrency: 100% compatible
- Testing: 100% compatible
- Only constraint: SDR I/O stays threaded (both agree)

**Recommendation**: Implement both. They solve different problems and reinforce each other.

## Implementation Roadmap

### Phase 1: Testing Infrastructure

**Goal**: Catch shutdown bugs before they reach production

1. **Add Integration Test Suite**
   - Create `tests/shutdown_tests.rs`
   - Test each shutdown scenario from the bug
   - Add timeout enforcement
   - CI: Run on every PR

2. **Add Property-Based Tests**
   - Install `proptest`
   - Fuzz shutdown timing
   - Test state transitions

3. **Add Loom Tests (Optional)**
   - Add `loom` dev dependency
   - Test critical shutdown paths
   - Document how to run: `RUSTFLAGS="--cfg loom" cargo test --test loom_tests`

**Deliverable**: Comprehensive test suite that catches missing shutdown checks

### Phase 2: State Machine Refactor

**Goal**: Make shutdown checks impossible to miss

1. **Extract State Enum**
   - Define `ScannerState` enum with all states
   - Migrate from boolean flags to state machine
   - Use exhaustive matching

2. **Centralized State Transitions**
   - Create `fn transition(&mut self, event: Event) -> Result<()>`
   - All state changes go through one function
   - Add state transition tests

3. **Verify with Tests**
   - Ensure all existing tests still pass
   - Add state-specific tests
   - Document state machine (diagrams)

**Deliverable**: Explicit state machine that the compiler enforces

### Phase 3: Hybrid Architecture Integration (Aligns with Plan 001)

**Goal**: Integrate structured concurrency with hybrid async/thread architecture

**Note**: This phase aligns with Plan 001 (Hybrid Architecture). SDR I/O remains threaded, async is used for coordination only.

1. **Add Tokio Coordinator** (Plan 001 Phase 3)
   - Add Tokio runtime for coordination layer
   - Keep SDR I/O in dedicated threads (Plan 001 requirement)
   - Use `tokio::select!` for multi-device coordination

2. **Structured Concurrency for Async Layer**
   - Add `TaskTracker` for all Tokio tasks (from this plan)
   - Use `CancellationToken` for unified shutdown (from this plan)
   - Ensure no async task leaks

3. **Bridge Layers**
   - Use `tokio::sync::mpsc` channels between threads and async
   - State machine works in both threaded and async contexts
   - Shutdown propagates across all layers

4. **Update Tests**
   - Test hybrid shutdown (threads + async)
   - Verify coordination layer
   - Benchmark: validate performance

**Deliverable**: Hybrid architecture with structured concurrency guarantees

### Phase 4: Advanced Testing (Long Term - Ongoing)

1. **Chaos Engineering**
   - Random delays in test harness
   - Fault injection (simulated errors)
   - Load testing with shutdown

2. **Performance Testing**
   - Shutdown latency benchmarks
   - Resource cleanup verification
   - Memory leak detection

3. **Documentation**
   - Update architecture docs
   - Create shutdown guide
   - Example code for contributors

## Migration Strategy

### Incremental Adoption

We don't need to rewrite everything at once:

1. **Test First**: Add tests with current architecture
2. **State Machine**: Refactor state management
3. **Async Gradually**: Convert modules one by one
   - Start with MainThread
   - Then Window processing
   - Finally AudioSession
4. **Verify**: Keep old code alongside new, compare behavior

### Compatibility

- Keep sync APIs for now
- Async internals with sync wrappers
- Gradual migration of callsites

## Benefits Summary

### Short Term (Testing Infrastructure)
- ✅ Catch shutdown bugs in CI
- ✅ Prevent regressions
- ✅ Faster debugging (tests reproduce issues)

### Medium Term (State Machine)
- ✅ Impossible to miss shutdown checks (compiler enforced)
- ✅ Clear code structure
- ✅ Better error messages

### Long Term (Async + Structured Concurrency)
- ✅ No thread leaks (automatic tracking)
- ✅ No manual drop ordering (async handles it)
- ✅ Natural cancellation (every await point)
- ✅ Better performance (async is more efficient)
- ✅ Industry best practices (Tokio patterns)

## Risks and Mitigation

### Risk 1: Complexity of Async
**Mitigation**:
- Incremental migration
- Team training on async Rust
- Start with simple conversions

### Risk 2: Performance Changes
**Mitigation**:
- Benchmark before and after
- Async can be faster for I/O bound work
- Profile real workloads

### Risk 3: Testing Coverage
**Mitigation**:
- Start with high-value tests (shutdown scenarios)
- Use coverage tools to find gaps
- Make tests fast (run often)

## References

### Tokio Documentation
- [Graceful Shutdown](https://tokio.rs/tokio/topics/shutdown)
- [Structured Concurrency RFC](https://github.com/tokio-rs/tokio/issues/2592)
- [Testing Async Code](https://tokio.rs/tokio/topics/testing)

### Crates
- [tokio](https://crates.io/crates/tokio) - Async runtime
- [tokio-util](https://crates.io/crates/tokio-util) - TaskTracker, CancellationToken
- [loom](https://crates.io/crates/loom) - Concurrency testing
- [proptest](https://crates.io/crates/proptest) - Property-based testing
- [async-dropper](https://crates.io/crates/async-dropper) - Async drop support

### Articles
- [Properly Testing Concurrent Data Structures](https://matklad.github.io/2024/07/05/properly-testing-concurrent-data-structures.html)
- [Structured Concurrency in Rust](https://without.boats/blog/the-scoped-task-trilemma/)

### Related Issues
- [Shutdown Issue Root Cause](docs/issues/2025-10-shutdown/01-shutting-down.md)
- [Shutdown Fix Plan](docs/issues/2025-10-shutdown/02-fix-plan.md)

## Conclusion

The shutdown bug revealed that our current architecture makes it too easy to introduce shutdown-related bugs. The proposed improvements—testing infrastructure, state machines, and structured concurrency—will:

1. **Prevent similar bugs** through architectural constraints
2. **Catch bugs earlier** through comprehensive testing
3. **Make the code more maintainable** through clear patterns
4. **Align with industry best practices** using modern Rust async patterns

We recommend starting with Phase 1 (testing infrastructure) immediately, as it provides value with minimal risk. Phases 2-4 can be planned based on team capacity and priorities.
