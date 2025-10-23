# Rust Testing Patterns for Multi-Threaded Applications

This document contains proven patterns for writing effective integration tests for multi-threaded Rust applications, with a focus on avoiding sleeps and writing deterministic tests.

## Table of Contents

1. [Dependency Injection for Testability](#dependency-injection-for-testability)
2. [Testing Concurrent Code](#testing-concurrent-code)
3. [Shutdown Testing Patterns](#shutdown-testing-patterns)
4. [Channel-Based Synchronization](#channel-based-synchronization)
5. [Property-Based Testing](#property-based-testing)
6. [Loom Testing for Concurrency](#loom-testing-for-concurrency)
7. [Async Testing Patterns](#async-testing-patterns)
8. [Mock and Trait-Based Testing](#mock-and-trait-based-testing)

---

## Dependency Injection for Testability

Dependency injection in Rust follows the principle: separate object construction from usage. Pass dependencies as constructor parameters rather than creating them internally.

### Pattern: Constructor Injection with Traits

```rust
// Define a trait for the dependency
trait Clock {
    fn now(&self) -> std::time::Instant;
}

// Production implementation
struct RealClock;
impl Clock for RealClock {
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

// Test implementation
struct MockClock {
    time: std::time::Instant,
}
impl Clock for MockClock {
    fn now(&self) -> std::time::Instant {
        self.time
    }
}

// System under test accepts any Clock implementation
struct Timer<C: Clock> {
    clock: C,
    start: std::time::Instant,
}

impl<C: Clock> Timer<C> {
    fn new(clock: C) -> Self {
        let start = clock.now();
        Self { clock, start }
    }

    fn elapsed(&self) -> std::time::Duration {
        self.clock.now().duration_since(self.start)
    }
}

#[test]
fn test_timer_with_mock_clock() {
    let start_time = std::time::Instant::now();
    let mock_clock = MockClock { time: start_time + std::time::Duration::from_secs(5) };

    let timer = Timer::new(mock_clock);
    assert_eq!(timer.elapsed(), std::time::Duration::from_secs(5));
}
```

### Pattern: Dependency Injection for Shutdown Coordination

Pass shutdown tokens/coordinators as constructor parameters to enable testing of shutdown behavior.

```rust
use tokio_util::sync::CancellationToken;

struct Worker {
    shutdown_token: CancellationToken,
}

impl Worker {
    fn new(shutdown_token: CancellationToken) -> Self {
        Self { shutdown_token }
    }

    fn work(&self) -> Result<(), &'static str> {
        for i in 0..100 {
            if self.shutdown_token.is_cancelled() {
                return Err("Shutdown");
            }
            // Do work...
        }
        Ok(())
    }
}

#[test]
fn test_worker_respects_shutdown() {
    let token = CancellationToken::new();
    let worker = Worker::new(token.clone());

    // Start work in separate thread
    let handle = std::thread::spawn(move || worker.work());

    // Trigger shutdown
    std::thread::sleep(std::time::Duration::from_millis(10));
    token.cancel();

    // Verify shutdown was detected
    let result = handle.join().unwrap();
    assert!(result.is_err());
}
```

---

## Testing Concurrent Code

### Pattern: Atomics for Cross-Thread State Verification

Use atomics to verify state across threads without blocking.

```rust
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn test_cleanup_order() {
    let cleanup_order = Arc::new(AtomicUsize::new(0));
    let audio_stopped = Arc::new(AtomicBool::new(false));
    let sdr_stopped = Arc::new(AtomicBool::new(false));

    let order_clone = cleanup_order.clone();
    let audio_clone = audio_stopped.clone();
    let sdr_clone = sdr_stopped.clone();

    std::thread::spawn(move || {
        // Simulate cleanup
        audio_clone.store(true, Ordering::SeqCst);
        order_clone.fetch_add(1, Ordering::SeqCst);

        std::thread::sleep(std::time::Duration::from_millis(10));

        sdr_clone.store(true, Ordering::SeqCst);
        order_clone.fetch_add(1, Ordering::SeqCst);
    }).join().unwrap();

    assert!(audio_stopped.load(Ordering::SeqCst));
    assert!(sdr_stopped.load(Ordering::SeqCst));
    assert_eq!(cleanup_order.load(Ordering::SeqCst), 2);
}
```

### Pattern: Barrier for Synchronized Thread Start

Use `std::sync::Barrier` to synchronize multiple threads and test race conditions.

```rust
use std::sync::{Arc, Barrier};

#[test]
fn test_concurrent_access_with_barrier() {
    let barrier = Arc::new(Barrier::new(5));
    let shared_state = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..5 {
        let barrier_clone = barrier.clone();
        let state_clone = shared_state.clone();

        let handle = std::thread::spawn(move || {
            // Wait for all threads to reach this point
            barrier_clone.wait();

            // Now all threads execute simultaneously
            state_clone.fetch_add(1, Ordering::SeqCst);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(shared_state.load(Ordering::SeqCst), 5);
}
```

### Pattern: Try-Lock for Non-Blocking Tests

Use `try_lock()` instead of `lock()` to prevent deadlocks in tests and verify lock contention.

```rust
use std::sync::{Arc, Mutex};

#[test]
fn test_drop_doesnt_block_when_pool_locked() {
    struct Resource {
        pool: Arc<Mutex<Vec<String>>>,
        id: String,
    }

    impl Drop for Resource {
        fn drop(&mut self) {
            // Use try_lock instead of lock to avoid deadlock
            if let Ok(mut pool) = self.pool.try_lock() {
                pool.push(self.id.clone());
            }
        }
    }

    let pool = Arc::new(Mutex::new(Vec::new()));
    let resource = Resource {
        pool: pool.clone(),
        id: "test".to_string(),
    };

    // Lock the pool
    let _lock = pool.lock().unwrap();

    // Spawn thread to drop resource
    let handle = std::thread::spawn(move || {
        drop(resource); // Should not block!
    });

    // Verify thread completes without blocking
    let result = handle.join();
    assert!(result.is_ok());
}
```

---

## Shutdown Testing Patterns

### Pattern: Testing Shutdown Signal Propagation

Verify that shutdown signals propagate correctly to all threads.

```rust
use tokio_util::sync::CancellationToken;

#[test]
fn test_shutdown_signal_propagation() {
    let token = CancellationToken::new();

    // Create multiple child tokens
    let token1 = token.clone();
    let token2 = token.clone();
    let token3 = token.clone();

    // All should be not-cancelled initially
    assert!(!token1.is_cancelled());
    assert!(!token2.is_cancelled());
    assert!(!token3.is_cancelled());

    // Cancel parent
    token.cancel();

    // All children should see cancellation
    assert!(token1.is_cancelled());
    assert!(token2.is_cancelled());
    assert!(token3.is_cancelled());
}
```

### Pattern: Testing Shutdown in Different States

Test shutdown from multiple system states to ensure graceful cleanup everywhere.

```rust
#[test]
fn test_shutdown_while_paused() {
    let token = CancellationToken::new();
    let paused = Arc::new(AtomicBool::new(true));

    let token_clone = token.clone();
    let paused_clone = paused.clone();

    let handle = std::thread::spawn(move || {
        loop {
            if paused_clone.load(Ordering::SeqCst) {
                if token_clone.is_cancelled() {
                    return "shutdown_while_paused";
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            if token_clone.is_cancelled() {
                return "shutdown_while_running";
            }
        }
    });

    // Let thread reach paused state
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Shutdown while paused
    token.cancel();

    let result = handle.join().unwrap();
    assert_eq!(result, "shutdown_while_paused");
}
```

### Pattern: Testing Shutdown Never Blocks

Verify that shutdown operations complete quickly without blocking.

```rust
#[test]
fn test_shutdown_never_blocks() {
    let start = std::time::Instant::now();
    let token = CancellationToken::new();

    // Hold a lock that shutdown might need
    let pool = Arc::new(Mutex::new(Vec::new()));
    let _lock = pool.lock().unwrap();

    // Shutdown should not block even if lock is held
    token.cancel();

    let elapsed = start.elapsed();
    assert!(elapsed < std::time::Duration::from_millis(100));
}
```

---

## Channel-Based Synchronization

Channels provide deterministic synchronization without sleep-based timing.

### Pattern: Channel as Completion Signal

Use channels to signal test completion instead of sleep.

```rust
use std::sync::mpsc;

#[test]
fn test_worker_completion_signal() {
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        // Do work
        for i in 0..100 {
            // Simulate work
        }

        // Signal completion
        tx.send("done").unwrap();
    });

    // Wait for completion signal (no sleep needed!)
    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(result, "done");
}
```

### Pattern: Multiple Producers with Channel

Test fan-in scenarios with multiple threads sending to one receiver.

```rust
use std::sync::mpsc;

#[test]
fn test_multiple_producers() {
    let (tx, rx) = mpsc::channel();
    let mut handles = vec![];

    for i in 0..5 {
        let tx_clone = tx.clone();
        let handle = std::thread::spawn(move || {
            tx_clone.send(i).unwrap();
        });
        handles.push(handle);
    }

    drop(tx); // Drop original sender

    // Collect all messages
    let mut results: Vec<_> = rx.iter().collect();
    results.sort();

    assert_eq!(results, vec![0, 1, 2, 3, 4]);

    for handle in handles {
        handle.join().unwrap();
    }
}
```

### Pattern: Broadcast Channel for Testing

Use broadcast channels (tokio::sync::broadcast) to test scenarios with multiple subscribers.

```rust
use tokio::sync::broadcast;

#[test]
fn test_broadcast_multiple_receivers() {
    let (tx, mut rx1) = broadcast::channel(10);
    let mut rx2 = tx.subscribe();
    let mut rx3 = tx.subscribe();

    // Send messages
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();

    // All receivers should get all messages
    assert_eq!(rx1.try_recv().unwrap(), 1);
    assert_eq!(rx2.try_recv().unwrap(), 1);
    assert_eq!(rx3.try_recv().unwrap(), 1);
}
```

---

## Property-Based Testing

Property-based testing uses proptest to generate many test cases automatically.

### Pattern: Testing Invariants

Test that properties hold for arbitrary inputs.

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn shutdown_at_arbitrary_time(
        shutdown_delay_ms in 0u64..500,
        work_iterations in 10usize..100,
    ) {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        let handle = std::thread::spawn(move || {
            for i in 0..work_iterations {
                if token_clone.is_cancelled() {
                    return i;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            work_iterations
        });

        std::thread::sleep(std::time::Duration::from_millis(shutdown_delay_ms));
        token.cancel();

        let iterations_completed = handle.join().unwrap();
        prop_assert!(iterations_completed <= work_iterations);
    }
}
```

### Pattern: Testing State Transitions

Use proptest to test state machines with random transition sequences.

```rust
proptest! {
    #[test]
    fn state_machine_invariants(
        pause_at_window in 1usize..20,
        shutdown_delay_ms in 0u64..100,
    ) {
        // Test that shutdown can occur at any point in state machine
        // and system maintains invariants

        // Setup state machine
        let mut state = StateMachine::new();
        state.start();

        // Random state transitions
        for _ in 0..pause_at_window {
            state.advance();
        }

        // Shutdown at arbitrary time
        std::thread::sleep(std::time::Duration::from_millis(shutdown_delay_ms));
        state.shutdown();

        // Verify invariants maintained
        prop_assert!(state.is_valid());
    }
}
```

---

## Loom Testing for Concurrency

Loom deterministically explores all possible thread interleavings to find race conditions.

### Pattern: Basic Loom Test

```rust
#[cfg(loom)]
mod loom_tests {
    use loom::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use loom::thread;

    #[test]
    fn test_concurrent_flag_access() {
        loom::model(|| {
            let flag = Arc::new(AtomicBool::new(false));

            let f1 = flag.clone();
            let f2 = flag.clone();

            let t1 = thread::spawn(move || {
                f1.store(true, Ordering::SeqCst);
                f1.load(Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || {
                f2.load(Ordering::SeqCst)
            });

            let r1 = t1.join().unwrap();
            let _r2 = t2.join().unwrap();

            assert!(r1, "Thread that set flag should see true");
        });
    }
}
```

### Pattern: Testing Shutdown with Loom

```rust
#[cfg(loom)]
#[test]
fn test_shutdown_and_pause_interaction() {
    loom::model(|| {
        let shutdown = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));

        let s1 = shutdown.clone();
        let p1 = paused.clone();
        let worker = thread::spawn(move || {
            for _ in 0..3 {
                if p1.load(Ordering::SeqCst) {
                    if s1.load(Ordering::SeqCst) {
                        return true; // Shutdown while paused
                    }
                    thread::yield_now();
                    continue;
                }

                if s1.load(Ordering::SeqCst) {
                    return true; // Shutdown while running
                }

                thread::yield_now();
            }
            false
        });

        let s2 = shutdown.clone();
        let p2 = paused.clone();
        let controller = thread::spawn(move || {
            p2.store(true, Ordering::SeqCst);
            thread::yield_now();
            s2.store(true, Ordering::SeqCst);
        });

        controller.join().unwrap();
        worker.join().unwrap();
    });
}
```

### Running Loom Tests

```bash
# Loom tests must be run with special cfg flag
RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release

# Enable logging for debugging
LOOM_LOG=1 RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release

# Enable location tracking
LOOM_LOCATION=1 RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release
```

---

## Async Testing Patterns

### Pattern: Basic Tokio Test

```rust
#[tokio::test]
async fn test_async_worker() {
    let result = async_work().await;
    assert!(result.is_ok());
}

async fn async_work() -> Result<(), String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    Ok(())
}
```

### Pattern: Multi-Threaded Async Test

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_async_workers() {
    let handles = (0..10)
        .map(|i| tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            i * 2
        }))
        .collect::<Vec<_>>();

    let results = futures::future::join_all(handles).await;

    assert_eq!(results.len(), 10);
}
```

### Pattern: Testing with Paused Time

Pause tokio's time to test time-dependent code deterministically.

```rust
#[tokio::test(start_paused = true)]
async fn test_timeout_behavior() {
    let start = tokio::time::Instant::now();

    // Advance time manually
    tokio::time::advance(std::time::Duration::from_secs(10)).await;

    assert!(start.elapsed() >= std::time::Duration::from_secs(10));
}
```

### Pattern: Testing Async with Channels

```rust
#[tokio::test]
async fn test_async_channel_communication() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(10);

    tokio::spawn(async move {
        for i in 0..5 {
            tx.send(i).await.unwrap();
        }
    });

    let mut results = vec![];
    while let Some(value) = rx.recv().await {
        results.push(value);
        if results.len() == 5 {
            break;
        }
    }

    assert_eq!(results, vec![0, 1, 2, 3, 4]);
}
```

---

## Mock and Trait-Based Testing

### Pattern: Trait-Based Mocking

Define interfaces as traits and provide mock implementations for testing.

```rust
trait Device {
    fn read(&self) -> Vec<u8>;
    fn write(&self, data: &[u8]) -> Result<(), String>;
}

// Production implementation
struct RealDevice;
impl Device for RealDevice {
    fn read(&self) -> Vec<u8> {
        // Real hardware access
        vec![]
    }

    fn write(&self, data: &[u8]) -> Result<(), String> {
        // Real hardware access
        Ok(())
    }
}

// Mock implementation
struct MockDevice {
    read_data: Vec<u8>,
    written_data: Arc<Mutex<Vec<u8>>>,
}

impl Device for MockDevice {
    fn read(&self) -> Vec<u8> {
        self.read_data.clone()
    }

    fn write(&self, data: &[u8]) -> Result<(), String> {
        self.written_data.lock().unwrap().extend_from_slice(data);
        Ok(())
    }
}

// System under test accepts any Device
struct System<D: Device> {
    device: D,
}

impl<D: Device> System<D> {
    fn process(&self, input: &[u8]) -> Vec<u8> {
        self.device.write(input).unwrap();
        self.device.read()
    }
}

#[test]
fn test_system_with_mock_device() {
    let mock = MockDevice {
        read_data: vec![1, 2, 3],
        written_data: Arc::new(Mutex::new(Vec::new())),
    };

    let written_data_ref = mock.written_data.clone();
    let system = System { device: mock };

    let result = system.process(&[4, 5, 6]);

    assert_eq!(result, vec![1, 2, 3]);
    assert_eq!(*written_data_ref.lock().unwrap(), vec![4, 5, 6]);
}
```

### Pattern: Conditional Mocking with cfg

Use cfg attributes to switch between real and mock implementations.

```rust
#[cfg(not(test))]
type DeviceImpl = RealDevice;

#[cfg(test)]
type DeviceImpl = MockDevice;

struct Application {
    device: DeviceImpl,
}
```

### Pattern: Interior Mutability for Mocks

Use `Cell`, `RefCell`, or atomics to track mock interactions.

```rust
use std::cell::RefCell;

struct MockCounter {
    call_count: RefCell<usize>,
}

impl MockCounter {
    fn new() -> Self {
        Self {
            call_count: RefCell::new(0),
        }
    }

    fn increment(&self) {
        *self.call_count.borrow_mut() += 1;
    }

    fn count(&self) -> usize {
        *self.call_count.borrow()
    }
}

#[test]
fn test_mock_tracking() {
    let mock = MockCounter::new();

    mock.increment();
    mock.increment();
    mock.increment();

    assert_eq!(mock.count(), 3);
}
```

---

## Summary

Key principles for testing multi-threaded Rust code:

1. **Inject dependencies** through constructor parameters
2. **Use channels** instead of sleep for synchronization
3. **Use atomics** for cross-thread state verification
4. **Use barriers** to synchronize thread starts
5. **Use try_lock** to prevent test deadlocks
6. **Test shutdown** from all system states
7. **Use proptest** to test invariants with random inputs
8. **Use loom** to find race conditions deterministically
9. **Mock via traits** for testable designs
10. **Test async code** with tokio-test and paused time

These patterns enable writing fast, reliable, deterministic tests for concurrent Rust applications.
