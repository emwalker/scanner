# Concurrent and Parallel System Testing

This reference covers testing ECS systems that run concurrently, including deterministic testing with Loom, race condition detection, and shutdown safety patterns.

## The Challenge of Concurrent Testing

Standard timing-based tests are non-deterministic and fail to reliably expose race conditions:

```rust
// BAD: Non-deterministic test
#[test]
fn flaky_concurrent_test() {
    let data = Arc::new(Mutex::new(vec![]));

    let data1 = data.clone();
    let t1 = std::thread::spawn(move || {
        data1.lock().unwrap().push(1);
    });

    let data2 = data.clone();
    let t2 = std::thread::spawn(move || {
        data2.lock().unwrap().push(2);
    });

    t1.join().unwrap();
    t2.join().unwrap();

    // This assertion might pass or fail depending on thread scheduling
    assert_eq!(data.lock().unwrap()[0], 1);
}
```

Use deterministic testing approaches instead.

## Using Loom for Deterministic Concurrency Testing

Loom exhaustively tests all possible thread interleavings, making concurrent bugs reproducible.

### Basic Loom Setup

Add to `Cargo.toml`:
```toml
[dev-dependencies]
loom = "0.7"
```

Basic loom test structure:

```rust
#[cfg(loom)]
#[test]
fn test_concurrent_access() {
    loom::model(|| {
        // Test code here - loom will explore all interleavings
    });
}
```

### Testing Concurrent Component Access

```rust
use loom::sync::{Arc, Mutex};
use loom::thread;

#[cfg(loom)]
#[test]
fn test_concurrent_component_mutations() {
    loom::model(|| {
        let component = Arc::new(Mutex::new(Health { value: 100 }));

        let c1 = component.clone();
        let c2 = component.clone();

        let t1 = thread::spawn(move || {
            let mut health = c1.lock().unwrap();
            health.value = health.value.saturating_sub(10);
        });

        let t2 = thread::spawn(move || {
            let mut health = c2.lock().unwrap();
            health.value = health.value.saturating_sub(20);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Verify final state is consistent
        let final_health = component.lock().unwrap();
        // Order doesn't matter, but total damage should be 30
        assert!(final_health.value <= 70);
    });
}
```

### Testing Arc<RwLock<>> Patterns

Common pattern in ECS for shared world access:

```rust
use loom::sync::{Arc, RwLock};

#[derive(Clone)]
struct SharedWorld {
    entities: Arc<RwLock<Vec<Entity>>>,
}

#[cfg(loom)]
#[test]
fn test_concurrent_world_access() {
    loom::model(|| {
        let world = SharedWorld {
            entities: Arc::new(RwLock::new(vec![])),
        };

        let w1 = world.clone();
        let w2 = world.clone();

        // Reader thread
        let t1 = thread::spawn(move || {
            let entities = w1.entities.read().unwrap();
            entities.len()
        });

        // Writer thread
        let t2 = thread::spawn(move || {
            let mut entities = w2.entities.write().unwrap();
            entities.push(Entity::from_raw(1));
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}
```

### Testing try_lock for Shutdown Safety

Critical for preventing deadlocks during cleanup:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

struct Resource {
    data: Arc<Mutex<Vec<u32>>>,
    shutdown: Arc<AtomicBool>,
}

impl Drop for Resource {
    fn drop(&mut self) {
        // GOOD: Non-blocking drop
        if let Ok(mut data) = self.data.try_lock() {
            data.clear();
        }
        // If lock fails, just drop without blocking
    }
}

#[cfg(loom)]
#[test]
fn test_shutdown_no_deadlock() {
    loom::model(|| {
        let data = Arc::new(Mutex::new(vec![1, 2, 3]));
        let shutdown = Arc::new(AtomicBool::new(false));

        let resource = Resource {
            data: data.clone(),
            shutdown: shutdown.clone(),
        };

        let d = data.clone();
        let s = shutdown.clone();

        let worker = thread::spawn(move || {
            while !s.load(Ordering::Acquire) {
                if let Ok(mut data) = d.try_lock() {
                    data.push(4);
                }
                // Don't block if lock held
            }
        });

        // Signal shutdown
        shutdown.store(true, Ordering::Release);

        // Drop resource while worker might hold lock
        drop(resource);

        worker.join().unwrap();

        // Should complete without deadlock
    });
}
```

### Testing Lock Ordering

Prevent deadlocks from inconsistent lock acquisition order:

```rust
struct TwoResources {
    resource_a: Arc<Mutex<u32>>,
    resource_b: Arc<Mutex<u32>>,
}

fn correct_ordering(resources: &TwoResources) {
    let _a = resources.resource_a.lock().unwrap();
    let _b = resources.resource_b.lock().unwrap();
    // Always acquire A then B
}

fn incorrect_ordering(resources: &TwoResources) {
    let _b = resources.resource_b.lock().unwrap();
    let _a = resources.resource_a.lock().unwrap();
    // Acquires B then A - can deadlock!
}

#[cfg(loom)]
#[test]
#[should_panic]
fn test_inconsistent_lock_order_deadlocks() {
    loom::model(|| {
        let resources = TwoResources {
            resource_a: Arc::new(Mutex::new(0)),
            resource_b: Arc::new(Mutex::new(0)),
        };

        let r1 = resources.clone();
        let r2 = resources.clone();

        let t1 = thread::spawn(move || {
            correct_ordering(&r1);
        });

        let t2 = thread::spawn(move || {
            incorrect_ordering(&r2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Loom will find the deadlock
    });
}
```

## Testing with Managed Threads

Alternative to Loom for controlled execution:

```rust
use std::sync::{Arc, Mutex, Condvar};

struct ThreadController {
    active_thread: Mutex<Option<usize>>,
    condvar: Condvar,
}

impl ThreadController {
    fn new() -> Self {
        Self {
            active_thread: Mutex::new(None),
            condvar: Condvar::new(),
        }
    }

    fn yield_control(&self, thread_id: usize) {
        let mut active = self.active_thread.lock().unwrap();
        *active = None;
        self.condvar.notify_all();

        // Wait until we're active again
        while active.is_none() || active.unwrap() != thread_id {
            active = self.condvar.wait(active).unwrap();
        }
    }

    fn activate_thread(&self, thread_id: usize) {
        let mut active = self.active_thread.lock().unwrap();
        *active = Some(thread_id);
        self.condvar.notify_all();
    }
}

#[test]
fn test_deterministic_interleaving() {
    let controller = Arc::new(ThreadController::new());
    let data = Arc::new(Mutex::new(0));

    let c1 = controller.clone();
    let d1 = data.clone();
    let t1 = std::thread::spawn(move || {
        c1.yield_control(0);
        let mut val = d1.lock().unwrap();
        *val += 1;
        drop(val);
        c1.yield_control(0);
    });

    let c2 = controller.clone();
    let d2 = data.clone();
    let t2 = std::thread::spawn(move || {
        c2.yield_control(1);
        let mut val = d2.lock().unwrap();
        *val += 2;
        drop(val);
        c2.yield_control(1);
    });

    // Control execution order
    controller.activate_thread(0);  // t1 runs
    std::thread::sleep(std::time::Duration::from_millis(10));
    controller.activate_thread(1);  // t2 runs
    std::thread::sleep(std::time::Duration::from_millis(10));
    controller.activate_thread(0);  // t1 completes
    std::thread::sleep(std::time::Duration::from_millis(10));
    controller.activate_thread(1);  // t2 completes

    t1.join().unwrap();
    t2.join().unwrap();

    assert_eq!(*data.lock().unwrap(), 3);
}
```

## Testing Entity Pool Thread Safety

Common pattern: thread-safe entity pool for ECS:

```rust
struct EntityPool {
    available: Arc<Mutex<Vec<Entity>>>,
}

impl EntityPool {
    fn allocate(&self) -> Option<Entity> {
        let mut pool = self.available.lock().unwrap();
        pool.pop()
    }

    fn release(&self, entity: Entity) {
        let mut pool = self.available.lock().unwrap();
        pool.push(entity);
    }
}

#[cfg(loom)]
#[test]
fn test_pool_concurrent_allocation() {
    loom::model(|| {
        let pool = EntityPool {
            available: Arc::new(Mutex::new(vec![
                Entity::from_raw(1),
                Entity::from_raw(2),
            ])),
        };

        let p1 = pool.clone();
        let p2 = pool.clone();

        let t1 = thread::spawn(move || {
            p1.allocate()
        });

        let t2 = thread::spawn(move || {
            p2.allocate()
        });

        let e1 = t1.join().unwrap();
        let e2 = t2.join().unwrap();

        // Both should get entities
        assert!(e1.is_some());
        assert!(e2.is_some());

        // Should get different entities
        assert_ne!(e1.unwrap(), e2.unwrap());

        // Pool should be empty
        assert!(pool.allocate().is_none());
    });
}
```

## Testing AtomicBool Shutdown Flags

Use atomics for lock-free shutdown signaling:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

struct System {
    shutdown: Arc<AtomicBool>,
}

impl System {
    fn process(&self) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;  // Stop processing
        }

        // Do work
        true
    }
}

#[cfg(loom)]
#[test]
fn test_shutdown_flag_visibility() {
    loom::model(|| {
        let shutdown = Arc::new(AtomicBool::new(false));

        let system = System {
            shutdown: shutdown.clone(),
        };

        let s = system.clone();
        let worker = thread::spawn(move || {
            let mut iterations = 0;
            while s.process() {
                iterations += 1;
                if iterations > 100 {
                    break;  // Safety limit
                }
            }
            iterations
        });

        // Signal shutdown
        shutdown.store(true, Ordering::Release);

        let count = worker.join().unwrap();

        // Worker should have stopped due to shutdown flag
        assert!(count < 100);
    });
}
```

## Testing Channel-Based Communication

Test systems communicating via channels:

```rust
use std::sync::mpsc;

#[cfg(loom)]
#[test]
fn test_channel_communication() {
    loom::model(|| {
        let (tx, rx) = mpsc::channel();

        let producer = thread::spawn(move || {
            tx.send(42).unwrap();
        });

        let consumer = thread::spawn(move || {
            rx.recv().unwrap()
        });

        producer.join().unwrap();
        let value = consumer.join().unwrap();

        assert_eq!(value, 42);
    });
}
```

## Property-Based Testing with Concurrency

Combine proptest with concurrent execution:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_concurrent_invariant(operations: Vec<(bool, u32)>) {
        let data = Arc::new(Mutex::new(0u32));

        let threads: Vec<_> = operations.into_iter().map(|(is_add, value)| {
            let data = data.clone();
            std::thread::spawn(move || {
                let mut guard = data.lock().unwrap();
                if is_add {
                    *guard = guard.saturating_add(value);
                } else {
                    *guard = guard.saturating_sub(value);
                }
            })
        }).collect();

        for thread in threads {
            thread.join().unwrap();
        }

        // Invariant: value never overflows
        let final_value = *data.lock().unwrap();
        prop_assert!(final_value <= u32::MAX);
    }
}
```

## Testing Parallel System Schedules

Test Bevy's parallel system execution:

```rust
#[derive(Component)]
struct ComponentA(u32);

#[derive(Component)]
struct ComponentB(u32);

fn system_a(mut query: Query<&mut ComponentA>) {
    for mut comp in query.iter_mut() {
        comp.0 += 1;
    }
}

fn system_b(mut query: Query<&mut ComponentB>) {
    for mut comp in query.iter_mut() {
        comp.0 += 1;
    }
}

#[test]
fn test_parallel_execution_no_conflicts() {
    let mut app = App::new();

    // Spawn entities with both components
    for i in 0..1000 {
        app.world_mut().spawn((
            ComponentA(i),
            ComponentB(i),
        ));
    }

    // These systems access disjoint data and can run in parallel
    app.add_systems(Update, (system_a, system_b));

    // Run multiple times to increase chance of race condition
    for _ in 0..100 {
        app.update();
    }

    // Verify all updates applied correctly
    for (a, b) in app.world().query::<(&ComponentA, &ComponentB)>().iter(app.world()) {
        assert_eq!(a.0, b.0);  // Both incremented same number of times
    }
}
```

## Testing for Data Races with Miri

Use Miri to detect undefined behavior:

```bash
# Run tests with Miri
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test
```

Example test:

```rust
#[test]
fn test_no_data_race() {
    let data = Arc::new(AtomicU32::new(0));

    let d1 = data.clone();
    let t1 = std::thread::spawn(move || {
        for _ in 0..1000 {
            d1.fetch_add(1, Ordering::Relaxed);
        }
    });

    let d2 = data.clone();
    let t2 = std::thread::spawn(move || {
        for _ in 0..1000 {
            d2.fetch_add(1, Ordering::Relaxed);
        }
    });

    t1.join().unwrap();
    t2.join().unwrap();

    assert_eq!(data.load(Ordering::Relaxed), 2000);
}
```

## Best Practices

1. **Use Loom for Critical Sections**: Test all code paths with shared mutable state
2. **Test Shutdown Paths**: Ensure cleanup doesn't deadlock
3. **Use try_lock in Drop**: Never block in destructors
4. **Atomic Shutdown Flags**: Prefer atomics over mutexes for shutdown signaling
5. **Consistent Lock Ordering**: Always acquire locks in the same order
6. **Test High Contention**: Spawn many threads to expose races
7. **Property-Based Testing**: Verify invariants hold under concurrent access
8. **Run Tests Multiple Times**: `cargo test -- --test-threads=1` for reproducibility

## Common Patterns

Pattern for shutdown-safe Drop:
```rust
impl Drop for Resource {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.data.try_lock() {
            // Cleanup
        }
    }
}
```

Pattern for lock-free shutdown:
```rust
while !shutdown.load(Ordering::Acquire) {
    // Work
}
```

Pattern for Loom test:
```rust
#[cfg(loom)]
#[test]
fn test_concurrent_operation() {
    loom::model(|| {
        // Test code
    });
}
```

## When to Use Each Approach

- **Loom**: Small critical sections, exhaustive testing (expensive but thorough)
- **Managed Threads**: Deterministic interleaving testing, debugging specific scenarios
- **Property-Based + Concurrent**: Testing invariants under various interleavings
- **Miri**: Detecting undefined behavior and data races
- **Integration Tests**: Testing full system behavior with real parallelism
