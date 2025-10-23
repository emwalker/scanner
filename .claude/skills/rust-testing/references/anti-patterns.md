# Rust Testing Anti-Patterns to Avoid

This document catalogs common mistakes in Rust testing, particularly for multi-threaded applications, along with better alternatives.

## Table of Contents

1. [Sleep-Based Synchronization](#sleep-based-synchronization)
2. [Flaky Test Patterns](#flaky-test-patterns)
3. [Blocking Operations in Drop](#blocking-operations-in-drop)
4. [Testing Implementation Details](#testing-implementation-details)
5. [Over-Mocking](#over-mocking)
6. [Ignoring Shutdown Paths](#ignoring-shutdown-paths)
7. [Shared State Without Synchronization](#shared-state-without-synchronization)
8. [Thread Leaks in Tests](#thread-leaks-in-tests)

---

## Sleep-Based Synchronization

### Anti-Pattern: Using Sleep for Test Coordination

```rust
// ANTI-PATTERN: Sleep-based test synchronization
#[test]
fn test_worker_completes_work() {
    let result = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    std::thread::spawn(move || {
        // Do work
        let value = expensive_computation();
        *result_clone.lock().unwrap() = Some(value);
    });

    // Hope the work finishes in 100ms
    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_result = result.lock().unwrap();
    assert!(final_result.is_some()); // Flaky: might fail on slow machines
}
```

**Why it's bad:**
- Tests become flaky and timing-dependent
- Slow machines or high load can cause spurious failures
- Wastes time waiting even when work completes early
- Doesn't scale to many tests running in parallel

### Better Alternative: Channel-Based Synchronization

```rust
// BETTER: Use channel to signal completion
#[test]
fn test_worker_completes_work() {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let value = expensive_computation();
        tx.send(value).unwrap(); // Signal completion
    });

    // Wait for actual completion (with timeout for safety)
    let result = rx.recv_timeout(std::time::Duration::from_secs(5));
    assert!(result.is_ok());
}
```

### Better Alternative: Join Handle

```rust
// BETTER: Join the thread
#[test]
fn test_worker_completes_work() {
    let handle = std::thread::spawn(|| {
        expensive_computation()
    });

    let result = handle.join().unwrap();
    assert!(result > 0);
}
```

---

## Flaky Test Patterns

### Anti-Pattern: Race Conditions in Tests

```rust
// ANTI-PATTERN: Test has race condition
#[test]
fn test_concurrent_counter() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = counter.clone();
        let handle = std::thread::spawn(move || {
            // Race: non-atomic read-modify-write
            let value = counter_clone.load(Ordering::SeqCst);
            counter_clone.store(value + 1, Ordering::SeqCst);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Flaky: sometimes passes, sometimes fails
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}
```

**Why it's bad:**
- Test contains the same race condition you're trying to prevent
- Intermittent failures make CI/CD unreliable
- Hard to debug because failures aren't reproducible

### Better Alternative: Atomic Operations

```rust
// BETTER: Use atomic fetch_add
#[test]
fn test_concurrent_counter() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = counter.clone();
        let handle = std::thread::spawn(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), 10);
}
```

### Anti-Pattern: Relying on External State

```rust
// ANTI-PATTERN: Test depends on filesystem state
#[test]
fn test_reads_config() {
    // Assumes /tmp/config.txt exists with specific content
    let result = read_config("/tmp/config.txt");
    assert_eq!(result, "expected_value");
}
```

**Why it's bad:**
- Fails if file doesn't exist or has different content
- Other tests or processes might modify the file
- Not isolated - tests can interfere with each other

### Better Alternative: Create Test State

```rust
// BETTER: Create test state explicitly
#[test]
fn test_reads_config() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.txt");

    std::fs::write(&config_path, "expected_value").unwrap();

    let result = read_config(&config_path);
    assert_eq!(result, "expected_value");

    // temp_dir automatically cleaned up
}
```

---

## Blocking Operations in Drop

### Anti-Pattern: Blocking Lock in Drop

```rust
// ANTI-PATTERN: Blocking lock in Drop can deadlock
struct Resource {
    pool: Arc<Mutex<Vec<String>>>,
    id: String,
}

impl Drop for Resource {
    fn drop(&mut self) {
        // DANGER: This can deadlock during shutdown!
        let mut pool = self.pool.lock().unwrap();
        pool.push(self.id.clone());
    }
}
```

**Why it's bad:**
- During shutdown, another thread might hold the lock
- Drop can't be cancelled or skipped
- Causes the entire program to hang
- Difficult to debug because stack traces show Drop, not the lock holder

### Better Alternative: Try-Lock in Drop

```rust
// BETTER: Use try_lock in Drop
impl Drop for Resource {
    fn drop(&mut self) {
        // Non-blocking: gracefully handle lock contention
        if let Ok(mut pool) = self.pool.try_lock() {
            pool.push(self.id.clone());
        }
        // If lock fails, resource is lost but program doesn't hang
    }
}
```

### Better Alternative: Explicit Cleanup Method

```rust
// BETTER: Explicit cleanup with automatic Drop fallback
impl Resource {
    fn cleanup(&mut self) -> Result<(), String> {
        let mut pool = self.pool.try_lock()
            .map_err(|_| "Lock contention".to_string())?;
        pool.push(self.id.clone());
        Ok(())
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        // Try cleanup, but don't panic if it fails
        let _ = self.cleanup();
    }
}

#[test]
fn test_explicit_cleanup() {
    let pool = Arc::new(Mutex::new(Vec::new()));
    let mut resource = Resource {
        pool: pool.clone(),
        id: "test".to_string(),
    };

    // Explicit cleanup before drop
    resource.cleanup().unwrap();

    let final_pool = pool.lock().unwrap();
    assert!(final_pool.contains(&"test".to_string()));
}
```

---

## Testing Implementation Details

### Anti-Pattern: Testing Private Methods

```rust
// ANTI-PATTERN: Testing private implementation
#[test]
fn test_internal_helper() {
    let obj = MyObject::new();
    // Need to make private method pub for testing
    assert_eq!(obj.internal_helper(5), 10);
}
```

**Why it's bad:**
- Tests become coupled to implementation
- Refactoring breaks tests even when behavior is correct
- Makes code harder to change
- Exposes internal API that shouldn't be public

### Better Alternative: Test Public Behavior

```rust
// BETTER: Test public API and behavior
#[test]
fn test_public_functionality() {
    let obj = MyObject::new();
    obj.process(5);
    assert_eq!(obj.result(), 10);
    // Internal helper is tested indirectly through public API
}
```

### Anti-Pattern: Mocking Everything

```rust
// ANTI-PATTERN: Over-mocking hides real issues
#[test]
fn test_with_all_mocks() {
    let mock_db = MockDatabase::new();
    let mock_cache = MockCache::new();
    let mock_network = MockNetwork::new();
    let mock_filesystem = MockFilesystem::new();

    let system = System::new(mock_db, mock_cache, mock_network, mock_filesystem);

    // Test passes but we're only testing mock interactions
    system.process();
}
```

**Why it's bad:**
- Not testing real integration
- Real bugs slip through because mocks behave differently
- High maintenance burden (mocks must stay in sync)
- Tests become brittle

### Better Alternative: Integration Tests with Real Components

```rust
// BETTER: Integration test with real components
#[test]
fn test_with_real_components() {
    let temp_db = create_test_database();
    let temp_cache = create_in_memory_cache();

    // Use real components for integration tests
    let system = System::new(temp_db, temp_cache);

    let result = system.process();
    assert!(result.is_ok());

    // Verify side effects in real components
    assert_eq!(temp_db.count(), 1);
}
```

---

## Over-Mocking

### Anti-Pattern: Mocking Simple Types

```rust
// ANTI-PATTERN: Unnecessary mock for simple type
trait TimeProvider {
    fn now(&self) -> u64;
}

struct MockTime {
    fixed_time: u64,
}

impl TimeProvider for MockTime {
    fn now(&self) -> u64 {
        self.fixed_time
    }
}

#[test]
fn test_with_mock_time() {
    let mock_time = MockTime { fixed_time: 1000 };
    let system = System::new(Box::new(mock_time));
    // ...
}
```

**Why it's questionable:**
- Adds complexity for minimal benefit
- Makes refactoring harder
- Real time source might work just fine in tests

### Better Alternative: Direct Injection

```rust
// BETTER: Just pass the value directly
struct System {
    start_time: u64,
}

impl System {
    fn new(start_time: u64) -> Self {
        Self { start_time }
    }
}

#[test]
fn test_with_fixed_time() {
    let system = System::new(1000);
    // Simpler and more direct
}
```

### When to Mock

Mock when:
- External service is slow, expensive, or unreliable (network, databases)
- Behavior is non-deterministic (random numbers, real time)
- Resource is unavailable in test environment (hardware, external APIs)
- Testing error conditions that are hard to trigger naturally

Don't mock when:
- Component is fast and deterministic
- Component is core to the feature being tested
- Creating mocks is more complex than using real implementation

---

## Ignoring Shutdown Paths

### Anti-Pattern: Not Testing Shutdown

```rust
// ANTI-PATTERN: Only testing happy path
#[test]
fn test_worker_processes_all_items() {
    let worker = Worker::new();

    for i in 0..100 {
        worker.process(i);
    }

    assert_eq!(worker.completed(), 100);
    // Never tests what happens during shutdown!
}
```

**Why it's bad:**
- Shutdown bugs only manifest in production
- Resource leaks during cleanup
- Hangs or deadlocks during program exit
- Data loss or corruption

### Better Alternative: Test Shutdown Scenarios

```rust
// BETTER: Test shutdown at various points
#[test]
fn test_worker_graceful_shutdown() {
    let worker = Worker::new();
    let shutdown_token = CancellationToken::new();

    let token_clone = shutdown_token.clone();
    let handle = std::thread::spawn(move || {
        worker.run(token_clone)
    });

    // Trigger shutdown after some work
    std::thread::sleep(std::time::Duration::from_millis(50));
    shutdown_token.cancel();

    // Verify clean shutdown
    let result = handle.join();
    assert!(result.is_ok());
}

#[test]
fn test_worker_shutdown_while_paused() {
    // Test shutdown from different states
    let worker = Worker::new();
    worker.pause();

    let shutdown_token = CancellationToken::new();
    shutdown_token.cancel();

    // Should exit cleanly even when paused
    worker.run(shutdown_token);
}
```

---

## Shared State Without Synchronization

### Anti-Pattern: Shared Mutable State

```rust
// ANTI-PATTERN: Shared mutable state in tests
static mut GLOBAL_COUNTER: usize = 0;

#[test]
fn test_a() {
    unsafe {
        GLOBAL_COUNTER = 0;
        GLOBAL_COUNTER += 1;
        assert_eq!(GLOBAL_COUNTER, 1);
    }
}

#[test]
fn test_b() {
    unsafe {
        GLOBAL_COUNTER = 0;
        GLOBAL_COUNTER += 2;
        assert_eq!(GLOBAL_COUNTER, 2);
    }
}
// Tests interfere with each other when run in parallel!
```

**Why it's bad:**
- Tests run in parallel by default
- Race conditions between tests
- Test order dependency
- Impossible to debug

### Better Alternative: Isolated State

```rust
// BETTER: Each test gets its own state
#[test]
fn test_a() {
    let mut counter = 0;
    counter += 1;
    assert_eq!(counter, 1);
}

#[test]
fn test_b() {
    let mut counter = 0;
    counter += 2;
    assert_eq!(counter, 2);
}
```

### Better Alternative: Serial Test Execution

```rust
// BETTER: Use serial_test crate when global state unavoidable
use serial_test::serial;

#[test]
#[serial]
fn test_a_with_global() {
    // Tests with #[serial] run one at a time
    setup_global_state();
    // ...
    cleanup_global_state();
}

#[test]
#[serial]
fn test_b_with_global() {
    setup_global_state();
    // ...
    cleanup_global_state();
}
```

---

## Thread Leaks in Tests

### Anti-Pattern: Spawning Without Joining

```rust
// ANTI-PATTERN: Thread leak
#[test]
fn test_background_work() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    // Spawn thread but don't join
    std::thread::spawn(move || {
        for _ in 0..1000 {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Test ends before thread completes!
    std::thread::sleep(std::time::Duration::from_millis(10));
    // Assertion might pass or fail randomly
    assert!(counter.load(Ordering::SeqCst) > 0);
}
```

**Why it's bad:**
- Thread continues running after test ends
- Consumes resources
- Can interfere with other tests
- Non-deterministic test results

### Better Alternative: Always Join

```rust
// BETTER: Join all threads
#[test]
fn test_background_work() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let handle = std::thread::spawn(move || {
        for _ in 0..1000 {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Wait for thread to complete
    handle.join().unwrap();

    // Now assertion is deterministic
    assert_eq!(counter.load(Ordering::SeqCst), 1000);
}
```

---

## Summary: Key Anti-Patterns to Avoid

1. **Sleep-based synchronization** → Use channels, joins, or barriers
2. **Flaky tests with race conditions** → Use proper synchronization primitives
3. **Blocking lock in Drop** → Use try_lock or explicit cleanup
4. **Testing private methods** → Test public behavior instead
5. **Over-mocking** → Use real components when practical
6. **Ignoring shutdown paths** → Test shutdown from all states
7. **Shared mutable state** → Isolate test state or use serial execution
8. **Thread leaks** → Always join spawned threads

Following these guidelines will lead to more reliable, maintainable, and deterministic tests for concurrent Rust applications.
