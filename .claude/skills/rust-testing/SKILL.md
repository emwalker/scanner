---
name: rust-testing
description: Use when writing integration tests for multi-threaded Rust applications, when tests contain sleep calls, when testing concurrent code or shutdown behavior, or when tests are flaky or non-deterministic. Provides expert patterns for dependency injection, channel-based synchronization, property-based testing, loom testing, and avoiding common testing anti-patterns. Apply when designing test architecture for async/concurrent systems or debugging flaky tests.
---

# Rust Testing for Multi-Threaded Applications

Comprehensive guidance for writing effective integration tests for multi-threaded Rust applications with a focus on avoiding sleeps, dependency injection, and deterministic testing.

## When to Use This Skill

Use this skill when:

- Writing integration tests for multi-threaded or concurrent Rust code
- Tests contain `std::thread::sleep()` or `tokio::time::sleep()` for synchronization
- Testing shutdown behavior and cleanup paths
- Tests are flaky or non-deterministic
- Testing code with `Arc<Mutex<T>>`, channels, or other concurrency primitives
- Designing test architecture for async/concurrent systems
- Using proptest for property-based testing
- Using loom for concurrency verification
- Debugging race conditions or deadlocks in tests
- Refactoring code to make it more testable

## Core Principles

### 1. Avoid Sleep-Based Synchronization

Sleep in tests is a code smell indicating improper synchronization. Instead:

- **Use channels** to signal completion or state changes
- **Join threads** to wait for completion
- **Use barriers** to synchronize thread starts
- **Use condition variables** for complex synchronization
- **For async code**, use tokio's paused time feature

**Never do this:**
```rust
std::thread::sleep(Duration::from_millis(100)); // Hope work finishes
assert!(result.is_ready()); // Flaky!
```

**Do this instead:**
```rust
let result = rx.recv_timeout(Duration::from_secs(5))?; // Wait for signal
assert!(result.is_ok()); // Deterministic!
```

### 2. Inject Dependencies

Structure code to accept dependencies as constructor parameters rather than creating them internally. This enables:

- Swapping real implementations for test doubles
- Testing with controlled/deterministic behavior
- Testing error conditions

**Pattern:**
```rust
// Accept CancellationToken instead of creating internally
struct Worker {
    shutdown_token: CancellationToken,
}

impl Worker {
    fn new(shutdown_token: CancellationToken) -> Self {
        Self { shutdown_token }
    }
}

#[test]
fn test_shutdown() {
    let token = CancellationToken::new();
    let worker = Worker::new(token.clone());

    token.cancel(); // Control shutdown in test

    // Verify behavior...
}
```

### 3. Test Shutdown Paths

Most bugs occur during shutdown. Always test:

- Shutdown while system is in various states (running, paused, etc.)
- Shutdown signal propagation to all threads
- Resource cleanup and deallocation
- No deadlocks or hangs during shutdown
- Graceful vs forced shutdown scenarios

### 4. Use Try-Lock in Drop

Blocking locks in Drop implementations cause deadlocks during shutdown.

**Never do this:**
```rust
impl Drop for Resource {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap(); // DEADLOCK RISK!
        pool.return_resource(self.id);
    }
}
```

**Do this instead:**
```rust
impl Drop for Resource {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pool.try_lock() {
            pool.return_resource(self.id);
        }
    }
}
```

### 5. Test Public Behavior, Not Implementation

Test through public APIs to make tests resilient to refactoring. Avoid testing private methods or implementation details.

### 6. Use Atomics for Verification

Use atomics to verify cross-thread behavior without blocking:

```rust
let completed = Arc::new(AtomicBool::new(false));
// Thread sets completed.store(true, ...)
assert!(completed.load(Ordering::SeqCst));
```

### 7. Property-Based Testing for Invariants

Use proptest to test that properties hold for arbitrary inputs:

```rust
proptest! {
    #[test]
    fn shutdown_at_arbitrary_time(delay_ms in 0u64..500) {
        // Test shutdown can happen at any time
    }
}
```

### 8. Loom for Critical Concurrency

Use loom to exhaustively test thread interleavings for critical concurrent code:

```rust
#[cfg(loom)]
#[test]
fn test_concurrent_access() {
    loom::model(|| {
        // Test all possible interleavings
    });
}
```

## Testing Workflow

### Step 1: Identify Test Requirements

Determine what needs to be tested:

- **Unit tests**: Individual functions/methods
- **Integration tests**: Multiple components working together
- **Concurrency tests**: Race conditions, deadlocks
- **Shutdown tests**: Cleanup and graceful termination
- **Property tests**: Invariants across many inputs

### Step 2: Choose Synchronization Strategy

Based on what you're testing, select appropriate synchronization:

- **Completion signals**: Use channels
- **Thread coordination**: Use barriers
- **Complex conditions**: Use condition variables
- **Cross-thread verification**: Use atomics
- **Async code**: Use tokio-test with paused time

### Step 3: Structure Test with Dependency Injection

Design the test to inject controllable dependencies:

1. Identify external dependencies (time, randomness, IO, shutdown signals)
2. Define traits for these dependencies
3. Inject them through constructors
4. Provide test implementations

### Step 4: Write the Test

Follow this structure:

```rust
#[test]
fn test_name() {
    // Setup: Create dependencies and system under test
    let shutdown_token = CancellationToken::new();
    let system = System::new(shutdown_token.clone());

    // Execute: Run the behavior
    let handle = std::thread::spawn(move || system.run());

    // Trigger: Cause state change
    shutdown_token.cancel();

    // Verify: Check results
    let result = handle.join().unwrap();
    assert!(result.is_ok());
}
```

### Step 5: Verify Determinism

Run the test multiple times to ensure it's not flaky:

```bash
# Run test 100 times
for i in {1..100}; do cargo test test_name; done

# Or use cargo-nextest with retries
cargo nextest run --retries 100
```

### Step 6: Add Property Tests (If Applicable)

If testing invariants, add proptest:

```rust
proptest! {
    #[test]
    fn invariant_holds(input in strategy()) {
        prop_assert!(check_invariant(input));
    }
}
```

### Step 7: Add Loom Test (For Critical Code)

For critical concurrent code, add loom test:

```rust
#[cfg(loom)]
#[test]
fn loom_test_critical_section() {
    loom::model(|| {
        // Test with all interleavings
    });
}
```

## Quick Reference: When to Use What

| Scenario | Tool/Pattern |
|----------|-------------|
| Wait for completion | Channel (mpsc/broadcast) |
| Synchronize thread starts | Barrier |
| Verify cross-thread state | Atomics (AtomicBool, AtomicUsize) |
| Test shutdown | CancellationToken injection |
| Prevent deadlock in Drop | try_lock() |
| Test async code | #[tokio::test] with paused time |
| Test properties/invariants | proptest |
| Find race conditions | loom |
| Mock external services | mockall or trait-based mocking |
| Temporary filesystem | tempfile |
| Test with different inputs | rstest or test-case |

## Common Patterns to Apply

### Pattern: Testing Worker Thread Completion

```rust
#[test]
fn test_worker_completes() {
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        do_work();
        tx.send(()).unwrap(); // Signal completion
    });

    // Wait for signal
    rx.recv_timeout(Duration::from_secs(5)).unwrap();
}
```

### Pattern: Testing Concurrent Access

```rust
#[test]
fn test_concurrent_increment() {
    let counter = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(5));

    let handles: Vec<_> = (0..5).map(|_| {
        let counter = counter.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait(); // Synchronize start
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), 5);
}
```

### Pattern: Testing Shutdown Signal Propagation

```rust
#[test]
fn test_shutdown_propagation() {
    let token = CancellationToken::new();
    let child_tokens: Vec<_> = (0..5).map(|_| token.clone()).collect();

    token.cancel();

    for child in child_tokens {
        assert!(child.is_cancelled());
    }
}
```

### Pattern: Testing Broadcast Channel Subscribers

```rust
#[test]
fn test_multiple_subscribers() {
    let (tx, mut rx1) = tokio::sync::broadcast::channel(10);
    let mut rx2 = tx.subscribe();

    tx.send(1).unwrap();
    tx.send(2).unwrap();

    assert_eq!(rx1.try_recv().unwrap(), 1);
    assert_eq!(rx2.try_recv().unwrap(), 1);
}
```

## Reference Material

Consult the bundled references for detailed information:

### references/patterns.md

Comprehensive patterns for:
- Dependency injection patterns
- Testing concurrent code with channels, barriers, atomics
- Shutdown testing patterns
- Property-based testing with proptest
- Loom testing patterns
- Async testing with tokio-test
- Mock and trait-based testing

Load this when: Implementing a specific testing pattern or need detailed examples.

### references/anti-patterns.md

Common mistakes to avoid:
- Sleep-based synchronization (with better alternatives)
- Flaky test patterns
- Blocking in Drop during shutdown
- Testing implementation details
- Over-mocking
- Ignoring shutdown paths
- Shared mutable state in tests

Load this when: Reviewing existing tests or debugging flaky/unreliable tests.

### references/tools.md

Testing tools and crates:
- loom for concurrency testing
- proptest for property-based testing
- tokio-test for async testing
- mockall for mocking
- serial_test, rstest, tempfile
- Debugging tools (cargo-nextest, deflake, ThreadSanitizer)

Load this when: Choosing tools for a new test suite or need detailed tool documentation.

## Summary

Key practices for effective multi-threaded Rust testing:

1. **Replace sleep with channels** for deterministic synchronization
2. **Inject dependencies** through constructors for testability
3. **Test shutdown paths** from all system states
4. **Use try_lock in Drop** to prevent deadlocks
5. **Test public behavior** not implementation details
6. **Use atomics** for cross-thread verification
7. **Use proptest** to test invariants with random inputs
8. **Use loom** to find race conditions in critical code
9. **Join all threads** to prevent test contamination
10. **Run tests multiple times** to verify determinism

Apply these principles to write fast, reliable, deterministic tests for concurrent Rust applications.
