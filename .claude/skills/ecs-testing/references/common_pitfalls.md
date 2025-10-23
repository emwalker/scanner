# Common Pitfalls and Anti-Patterns in ECS Testing

This reference covers common mistakes when testing ECS systems and how to avoid them.

## Anti-Pattern: Testing Implementation Details

### The Problem

Testing internal component structure or private implementation details makes tests fragile.

```rust
// BAD: Testing implementation details
#[test]
fn bad_test() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        Health { value: 100 },
        InternalProcessingState::Stage1,  // Internal detail
    )).id();

    app.add_systems(Update, process_system);
    app.update();

    // Testing internal state
    let state = app.world().entity(entity).get::<InternalProcessingState>().unwrap();
    assert!(matches!(state, InternalProcessingState::Stage2));
}
```

### The Solution

Test observable behavior and outputs instead:

```rust
// GOOD: Testing behavior
#[test]
fn good_test() {
    let mut app = App::new();

    let entity = app.world_mut().spawn(Health { value: 100 }).id();

    app.add_systems(Update, process_system);
    app.update();

    // Test the observable result
    assert!(app.world().entity(entity).contains::<Processed>());

    let health = app.world().entity(entity).get::<Health>().unwrap();
    assert_eq!(health.value, 100);  // Verify side effects
}
```

### Why It Matters

- Tests tied to implementation break during refactoring
- Internal details may change without affecting external behavior
- Tests should verify contracts, not implementation

## Anti-Pattern: Timing-Based Concurrency Tests

### The Problem

Using sleeps or delays to "ensure" ordering is non-deterministic:

```rust
// BAD: Flaky timing-based test
#[test]
fn flaky_concurrent_test() {
    let data = Arc::new(Mutex::new(vec![]));

    let d1 = data.clone();
    let t1 = std::thread::spawn(move || {
        d1.lock().unwrap().push(1);
    });

    std::thread::sleep(Duration::from_millis(10));  // Hope thread 1 finishes

    let d2 = data.clone();
    let t2 = std::thread::spawn(move || {
        d2.lock().unwrap().push(2);
    });

    t1.join().unwrap();
    t2.join().unwrap();

    // This might fail if timing is different
    assert_eq!(data.lock().unwrap()[0], 1);
}
```

### The Solution

Use explicit synchronization or deterministic testing tools:

```rust
// GOOD: Explicit synchronization
#[test]
fn deterministic_concurrent_test() {
    let data = Arc::new(Mutex::new(vec![]));
    let barrier = Arc::new(Barrier::new(2));

    let d1 = data.clone();
    let b1 = barrier.clone();
    let t1 = std::thread::spawn(move || {
        d1.lock().unwrap().push(1);
        b1.wait();  // Explicit sync point
    });

    let d2 = data.clone();
    let b2 = barrier.clone();
    let t2 = std::thread::spawn(move || {
        b2.wait();  // Wait for thread 1
        d2.lock().unwrap().push(2);
    });

    t1.join().unwrap();
    t2.join().unwrap();

    assert_eq!(*data.lock().unwrap(), vec![1, 2]);
}

// BETTER: Use Loom
#[cfg(loom)]
#[test]
fn loom_test() {
    loom::model(|| {
        // Loom explores all interleavings
    });
}
```

### Why It Matters

- Timing-based tests fail intermittently (flaky tests)
- Different machines have different timing
- CI environments may be slower than development machines

## Anti-Pattern: Overly Complex Test Setup

### The Problem

Recreating entire application state obscures test intent:

```rust
// BAD: Complex setup obscures what's being tested
#[test]
fn overcomplicated_test() {
    let mut app = App::new();

    // Massive setup
    app.insert_resource(GameConfig::default());
    app.insert_resource(AssetLoader::new());
    app.insert_resource(NetworkManager::new());
    app.insert_resource(AudioSystem::new());
    app.insert_resource(PhysicsEngine::new());

    // Spawn many entities
    for i in 0..100 {
        app.world_mut().spawn(create_complex_entity(i));
    }

    // Add many systems
    app.add_systems(Update, (
        physics_system,
        collision_system,
        animation_system,
        audio_system,
        network_system,
        ui_system,
    ));

    app.update();

    // What is this test actually testing?
    assert!(app.world().query::<&Transform>().iter(app.world()).count() > 0);
}
```

### The Solution

Minimal setup focused on test intent:

```rust
// GOOD: Minimal setup, clear intent
#[test]
fn focused_test() {
    let mut app = App::new();

    // Only what's needed for this test
    app.world_mut().spawn((
        Health { value: 0 },
        Dead,
    ));

    app.add_systems(Update, despawn_dead_entities);
    app.update();

    // Clear what we're testing: dead entities are despawned
    assert_eq!(app.world().entities().len(), 0);
}
```

### Why It Matters

- Hard to understand what the test is verifying
- Slow tests that do too much
- Difficult to maintain when setup code changes
- False failures from unrelated setup

## Anti-Pattern: Testing Multiple Concerns in One Test

### The Problem

One test verifying multiple unrelated behaviors:

```rust
// BAD: Tests multiple concerns
#[test]
fn test_everything() {
    let mut app = App::new();

    // Setup for health test
    let entity1 = app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 25 },
    )).id();

    // Setup for movement test
    let entity2 = app.world_mut().spawn((
        Position { x: 0.0, y: 0.0 },
        Velocity { x: 1.0, y: 0.0 },
    )).id();

    // Setup for spawning test
    app.insert_resource(SpawnTimer { remaining: 0.0 });

    app.add_systems(Update, (
        damage_system,
        movement_system,
        spawning_system,
    ));

    app.update();

    // Testing damage
    let health = app.world().entity(entity1).get::<Health>().unwrap();
    assert_eq!(health.value, 75);

    // Testing movement
    let pos = app.world().entity(entity2).get::<Position>().unwrap();
    assert_eq!(pos.x, 1.0);

    // Testing spawning
    let enemies = app.world().query::<&Enemy>().iter(app.world()).count();
    assert_eq!(enemies, 5);
}
```

### The Solution

One test per concern:

```rust
// GOOD: Separate tests for each concern
#[test]
fn test_damage_applies() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 25 },
    )).id();

    app.add_systems(Update, damage_system);
    app.update();

    let health = app.world().entity(entity).get::<Health>().unwrap();
    assert_eq!(health.value, 75);
}

#[test]
fn test_movement_updates_position() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        Position { x: 0.0, y: 0.0 },
        Velocity { x: 1.0, y: 0.0 },
    )).id();

    app.add_systems(Update, movement_system);
    app.update();

    let pos = app.world().entity(entity).get::<Position>().unwrap();
    assert_eq!(pos.x, 1.0);
}

#[test]
fn test_enemies_spawn_on_timer() {
    let mut app = App::new();

    app.insert_resource(SpawnTimer { remaining: 0.0 });
    app.add_systems(Update, spawning_system);
    app.update();

    let enemies = app.world().query::<&Enemy>().iter(app.world()).count();
    assert_eq!(enemies, 5);
}
```

### Why It Matters

- When test fails, unclear which behavior broke
- Changes to one system break tests for other systems
- Tests become interdependent

## Anti-Pattern: Ignoring Shutdown Cleanup

### The Problem

Not testing that resources are properly released:

```rust
// BAD: No cleanup testing
struct ResourcePool {
    resources: Arc<Mutex<Vec<Resource>>>,
}

impl Drop for ResourcePool {
    fn drop(&mut self) {
        // What if this locks forever?
        let mut resources = self.resources.lock().unwrap();
        resources.clear();
    }
}

#[test]
fn test_only_happy_path() {
    let pool = ResourcePool::new();
    let resource = pool.acquire();
    pool.release(resource);
    // What about Drop?
}
```

### The Solution

Test cleanup and shutdown paths:

```rust
// GOOD: Test shutdown behavior
#[test]
fn test_cleanup_no_deadlock() {
    let pool = ResourcePool::new();
    let resource = pool.acquire();

    // Simulate shutdown while resource held
    let resource_handle = resource.clone();
    let pool_handle = pool.clone();

    std::thread::spawn(move || {
        // Hold resource during shutdown
        let _r = resource_handle;
        std::thread::sleep(Duration::from_millis(100));
    });

    // Drop pool - should not deadlock
    drop(pool);
}

#[test]
fn test_drop_during_operation() {
    let pool = ResourcePool::new();

    {
        let _resource = pool.acquire();
        // Pool dropped while resource still acquired
    }
    // Should complete without hanging
}
```

### Why It Matters

- Cleanup bugs cause resource leaks
- Deadlocks during shutdown crash the application
- Shutdown paths are rarely exercised in normal testing

## Anti-Pattern: Not Testing Edge Cases

### The Problem

Only testing the happy path:

```rust
// BAD: Only tests normal case
#[test]
fn test_normal_case() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 25 },
    ));

    app.add_systems(Update, damage_system);
    app.update();

    let health = app.world().query::<&Health>().single(app.world());
    assert_eq!(health.value, 75);
}
```

### The Solution

Test boundaries and edge cases:

```rust
// GOOD: Tests edge cases
#[test]
fn test_zero_health() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 0 },
        Damage { value: 25 },
    ));

    app.add_systems(Update, damage_system);
    app.update();

    let health = app.world().query::<&Health>().single(app.world());
    assert_eq!(health.value, 0);  // Should stay at zero
}

#[test]
fn test_damage_exceeds_health() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 10 },
        Damage { value: 100 },
    ));

    app.add_systems(Update, damage_system);
    app.update();

    let health = app.world().query::<&Health>().single(app.world());
    assert_eq!(health.value, 0);  // Should not underflow
}

#[test]
fn test_no_entities() {
    let mut app = App::new();

    // No entities
    app.add_systems(Update, damage_system);
    app.update();  // Should handle gracefully
}

#[test]
fn test_many_entities() {
    let mut app = App::new();

    for _ in 0..10000 {
        app.world_mut().spawn((
            Health { value: 100 },
            Damage { value: 1 },
        ));
    }

    app.add_systems(Update, damage_system);
    app.update();
}
```

### Why It Matters

- Bugs often lurk in edge cases
- Boundary conditions expose off-by-one errors
- Empty collections cause crashes
- Large inputs expose performance issues

## Anti-Pattern: Mutation Without Verification

### The Problem

Not verifying that mutations actually occurred:

```rust
// BAD: Assumes system worked
#[test]
fn test_without_verification() {
    let mut app = App::new();

    app.world_mut().spawn(Health { value: 100 });
    app.add_systems(Update, damage_system);

    app.update();

    // No verification - did anything happen?
}
```

### The Solution

Always verify expected changes:

```rust
// GOOD: Verifies the mutation
#[test]
fn test_with_verification() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 25 },
    )).id();

    // Capture before state
    let before = app.world().entity(entity).get::<Health>().unwrap().value;

    app.add_systems(Update, damage_system);
    app.update();

    // Verify after state
    let after = app.world().entity(entity).get::<Health>().unwrap().value;
    assert_ne!(before, after);
    assert_eq!(after, 75);
}
```

### Why It Matters

- System might not run at all
- System might run but have no effect
- Silent failures go undetected

## Anti-Pattern: Shared Mutable State Between Tests

### The Problem

Tests sharing state through static variables:

```rust
// BAD: Shared mutable state
static mut GLOBAL_COUNTER: u32 = 0;

#[test]
fn test_one() {
    unsafe {
        GLOBAL_COUNTER += 1;
        assert_eq!(GLOBAL_COUNTER, 1);  // Fails if test_two runs first
    }
}

#[test]
fn test_two() {
    unsafe {
        GLOBAL_COUNTER += 1;
        assert_eq!(GLOBAL_COUNTER, 1);  // Fails if test_one runs first
    }
}
```

### The Solution

Isolate test state:

```rust
// GOOD: Isolated state
#[test]
fn test_one() {
    let mut counter = 0;
    counter += 1;
    assert_eq!(counter, 1);
}

#[test]
fn test_two() {
    let mut counter = 0;
    counter += 1;
    assert_eq!(counter, 1);
}
```

### Why It Matters

- Tests fail when run in parallel
- Test order affects results
- Debugging is difficult

## Anti-Pattern: Ignoring System Ordering

### The Problem

Assuming systems run in specific order without enforcing it:

```rust
// BAD: Assumes ordering without specifying
#[test]
fn test_assumes_order() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 10 },
        Damage { value: 15 },
    ));

    // These systems run in parallel by default!
    app.add_systems(Update, (
        damage_system,
        check_death_system,
    ));

    app.update();

    // This might fail - check_death might run before damage_system
    assert!(app.world().query::<&Dead>().iter(app.world()).count() > 0);
}
```

### The Solution

Explicitly specify ordering when it matters:

```rust
// GOOD: Explicit ordering
#[test]
fn test_with_explicit_order() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 10 },
        Damage { value: 15 },
    ));

    // Explicitly chain systems
    app.add_systems(Update, (
        damage_system,
        check_death_system,
    ).chain());

    app.update();

    assert!(app.world().query::<&Dead>().iter(app.world()).count() > 0);
}
```

### Why It Matters

- Parallel execution is non-deterministic
- Tests pass locally but fail in CI
- Bevy schedules systems in parallel when possible

## Anti-Pattern: Testing Through Side Effects Only

### The Problem

Only verifying indirect side effects:

```rust
// BAD: Only tests side effect
#[test]
fn test_side_effect_only() {
    let mut app = App::new();

    app.add_systems(Update, spawn_enemy_system);
    app.update();

    // Only checks that something was spawned
    assert!(app.world().entities().len() > 0);
    // But was it an enemy? Does it have the right components?
}
```

### The Solution

Verify direct results and side effects:

```rust
// GOOD: Verifies specific results
#[test]
fn test_specific_results() {
    let mut app = App::new();

    app.add_systems(Update, spawn_enemy_system);
    app.update();

    // Verify specific entity type spawned
    let enemy_count = app.world().query::<&Enemy>().iter(app.world()).count();
    assert_eq!(enemy_count, 1);

    // Verify it has required components
    let enemy = app.world().query::<(&Enemy, &Health)>().single(app.world());
    assert_eq!(enemy.1.value, 50);
}
```

### Why It Matters

- Side effects might come from wrong source
- Missing verification of data correctness
- False positives from partial correctness

## Best Practices Summary

1. **Test Behavior, Not Implementation**: Verify outputs, not internal state
2. **Deterministic Concurrency**: Use explicit sync or Loom, never sleeps
3. **Minimal Setup**: Only configure what the test needs
4. **One Concern Per Test**: Each test verifies one behavior
5. **Test Cleanup**: Verify shutdown and resource release
6. **Test Edge Cases**: Zero, empty, maximum, boundary values
7. **Verify Mutations**: Always check that changes occurred
8. **Isolate State**: No shared mutable state between tests
9. **Explicit Ordering**: Use .chain() when order matters
10. **Verify Directly**: Check specific results, not just side effects
