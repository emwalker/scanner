---
name: ecs-testing
description: This skill should be used when implementing tests for Entity-Component-System (ECS) architectures in Rust. Apply this skill when writing unit tests for systems, integration tests for multiple interacting systems, testing concurrent ECS operations, implementing property-based tests for ECS invariants, or refactoring ECS code to improve testability. The skill covers test fixtures, dependency injection patterns, deterministic concurrency testing, and common testing anti-patterns specific to ECS.
---

# ECS Testing

## Overview

Testing Entity-Component-System architectures in Rust requires specialized patterns and strategies. ECS separates data (components) from logic (systems), creating unique testing challenges: systems depend on world state, multiple systems interact through shared components, and parallel system execution introduces race conditions. This skill provides comprehensive patterns for testing ECS code at multiple levels, from isolated system tests to full integration scenarios.

## When to Use This Skill

Apply this skill when:

- Implementing new ECS systems or components that need test coverage
- Writing tests for existing untested ECS code
- Debugging race conditions or concurrency issues in parallel systems
- Refactoring ECS code and need to maintain test coverage
- Reviewing ECS code and evaluating testability
- Designing ECS architecture with testability in mind
- Testing shutdown behavior and cleanup in ECS systems
- Implementing property-based tests for ECS state invariants

## Core Testing Strategy

ECS testing follows a layered approach:

**Unit Tests**: Test individual systems in isolation with minimal world setup. Focus on component transformations, event handling, and resource mutations.

**Integration Tests**: Test multiple systems together to verify interactions, execution order, and shared state management.

**Concurrency Tests**: Test parallel system execution for race conditions, deadlocks, and deterministic behavior using tools like Loom.

**Property Tests**: Test ECS invariants across random inputs using property-based testing frameworks.

The key insight: structure code for testability by separating pure logic from world access, using dependency injection for external dependencies, and creating builder patterns for complex test fixtures.

## Unit Testing ECS Systems

### Basic System Testing Pattern

The fundamental pattern for testing a system:

1. Create a test `World` or `App`
2. Insert required resources and test entities
3. Run the system under test
4. Query the world state to verify results

Example structure (Bevy-style):

```rust
#[test]
fn test_damage_system() {
    let mut app = App::new();

    // Setup: spawn entities with components
    app.world_mut().spawn((Health { value: 100 }, Damage { value: 25 }));

    // Register and run the system
    app.add_systems(Update, apply_damage_system);
    app.update();

    // Verify: check resulting component state
    let mut query = app.world_mut().query::<&Health>();
    for health in query.iter(app.world()) {
        assert_eq!(health.value, 75);
    }
}
```

### Testing Systems with Resources

Systems often depend on shared resources. Test by inserting mock resources:

```rust
#[test]
fn test_system_with_time() {
    let mut app = App::new();

    // Insert mock resource with known state
    app.insert_resource(GameTime { elapsed: 1.0 });

    app.add_systems(Update, time_dependent_system);
    app.update();

    // Verify resource mutation
    let time = app.world().resource::<GameTime>();
    assert_eq!(time.elapsed, 2.0);
}
```

### Testing Event-Driven Systems

For systems that read or write events:

```rust
#[test]
fn test_event_system() {
    let mut app = App::new();
    app.add_event::<PlayerDied>();

    // Write test event
    app.world_mut().send_event(PlayerDied { id: 42 });

    app.add_systems(Update, handle_player_death);
    app.update();

    // Verify side effects occurred
    let mut query = app.world_mut().query::<&DeathMarker>();
    assert_eq!(query.iter(app.world()).count(), 1);
}
```

### Testing Command Execution

Systems using `Commands` for deferred operations require testing after commands execute:

```rust
#[test]
fn test_spawn_system() {
    let mut app = App::new();

    app.add_systems(Update, spawn_enemies);
    app.update();  // Commands execute at end of update

    let mut query = app.world_mut().query::<&Enemy>();
    assert_eq!(query.iter(app.world()).count(), 5);
}
```

See `references/unit_testing_patterns.md` for comprehensive examples including:
- Testing queries with filters
- Testing input-driven systems
- Testing entity despawning
- Testing optional components
- Handling panics in tests

## Integration Testing Multiple Systems

### Testing System Chains

Test multiple systems together to verify ordering and interactions:

```rust
#[test]
fn test_damage_and_death() {
    let mut app = App::new();

    app.world_mut().spawn((Health { value: 10 }, Damage { value: 15 }));

    // Chain systems explicitly
    app.add_systems(Update, (
        apply_damage_system,
        check_death_system,
        despawn_dead_system,
    ).chain());

    app.update();

    // Verify entity was despawned
    let count = app.world().query::<Entity>().iter(app.world()).count();
    assert_eq!(count, 0);
}
```

### Testing State Transitions

Use integration tests for complex state machines spanning multiple systems:

```rust
#[test]
fn test_state_machine_transition() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        State::Idle,
        Position::default(),
    )).id();

    app.add_systems(Update, (
        detect_movement,
        transition_to_moving,
        apply_movement,
    ).chain());

    // Trigger transition
    app.insert_resource(Input { moving: true });
    app.update();

    let state = app.world().entity(entity).get::<State>().unwrap();
    assert!(matches!(state, State::Moving));
}
```

See `references/integration_testing.md` for:
- Testing system dependencies
- Testing side effects across systems
- Testing resource contention
- Verifying execution order

## Testing Concurrent and Parallel Systems

### Challenges in Concurrent Testing

Parallel systems introduce non-determinism. Standard timing-based tests fail to reliably expose race conditions. Use deterministic testing approaches:

### Using Loom for Deterministic Concurrency Testing

Loom exhaustively tests thread interleavings. Mark test with `#[cfg(loom)]`:

```rust
#[cfg(loom)]
#[test]
fn test_concurrent_component_access() {
    loom::model(|| {
        let world = Arc::new(loom::sync::Mutex::new(World::new()));

        let world1 = world.clone();
        let world2 = world.clone();

        let t1 = loom::thread::spawn(move || {
            let mut w = world1.lock().unwrap();
            // System 1 operations
        });

        let t2 = loom::thread::spawn(move || {
            let mut w = world2.lock().unwrap();
            // System 2 operations
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}
```

### Testing Shutdown Safety

Critical for preventing deadlocks during teardown:

```rust
#[test]
fn test_shutdown_no_deadlock() {
    let mut app = App::new();
    let shutdown = Arc::new(AtomicBool::new(false));

    app.insert_resource(ShutdownFlag(shutdown.clone()));
    app.add_systems(Update, system_with_locks);

    // Trigger shutdown during execution
    shutdown.store(true, Ordering::SeqCst);

    // Should complete without hanging
    let result = std::panic::catch_unwind(|| {
        app.update();
    });

    assert!(result.is_ok());
}
```

### Testing for Race Conditions

Use property-based testing with concurrent execution:

```rust
#[test]
fn test_parallel_system_invariant() {
    proptest!(|(entities: Vec<(Health, Damage)>)| {
        let mut app = App::new();

        for (health, damage) in entities {
            app.world_mut().spawn((health, damage));
        }

        // Run systems in parallel (Bevy does this automatically)
        app.add_systems(Update, (system1, system2));
        app.update();

        // Verify invariant holds regardless of execution order
        for health in app.world().query::<&Health>().iter(app.world()) {
            prop_assert!(health.value >= 0);
        }
    });
}
```

See `references/concurrent_testing.md` for:
- Comprehensive Loom patterns
- Testing thread pools
- Managed thread testing
- Deadlock prevention patterns

## Property-Based Testing for ECS

Property-based testing validates invariants across randomly generated inputs. Excellent for:
- State machine transitions
- Component value ranges
- Entity lifecycle invariants
- System idempotence

### Testing ECS Invariants

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn health_never_negative(initial_health in 0u32..1000, damage in 0u32..500) {
        let mut app = App::new();
        app.world_mut().spawn(Health { value: initial_health });

        app.add_systems(Update, apply_damage_system);
        app.insert_resource(DamageEvent { amount: damage });
        app.update();

        for health in app.world().query::<&Health>().iter(app.world()) {
            prop_assert!(health.value >= 0);
        }
    }
}
```

### Generating Valid Entities

Create custom `Arbitrary` implementations for components:

```rust
impl Arbitrary for ValidEntity {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (0u32..100, 0u32..100)
            .prop_map(|(health, damage)| ValidEntity {
                health: Health { value: health },
                damage: Damage { value: damage },
            })
            .boxed()
    }
}
```

### Testing State Machine Transitions

Use proptest's state machine testing:

```rust
use proptest::state_machine::{ReferenceStateMachine, StateMachineTest};

#[derive(Clone, Debug)]
struct EcsModel {
    entities: Vec<EntityState>,
}

impl ReferenceStateMachine for EcsModel {
    type State = Self;
    type Transition = SystemTransition;

    fn init_state() -> BoxedStrategy<Self::State> {
        // Generate initial valid states
    }

    fn transitions(state: &Self::State) -> BoxedStrategy<Self::Transition> {
        // Generate valid transitions from current state
    }

    fn apply(state: Self::State, transition: &Self::Transition) -> Self::State {
        // Apply transition to model
    }

    fn preconditions(state: &Self::State, transition: &Self::Transition) -> bool {
        // Check if transition is valid
    }
}
```

See `references/property_based_testing.md` for:
- Advanced proptest patterns
- Shrinking strategies
- State machine testing examples
- Custom generators for ECS types

## Architectural Patterns for Testability

### Dependency Injection via Traits

Avoid coupling systems to concrete implementations. Use traits for external dependencies:

```rust
trait TimeProvider {
    fn elapsed(&self) -> f32;
}

fn movement_system(
    mut query: Query<&mut Position>,
    time: impl TimeProvider,
) {
    let dt = time.elapsed();
    // Use dt for movement
}

// In tests, provide mock implementation
struct MockTime { elapsed: f32 }
impl TimeProvider for MockTime {
    fn elapsed(&self) -> f32 { self.elapsed }
}
```

The "Deps-Pattern" uses generic parameters for zero-cost abstraction:

```rust
fn system<T: HealthProvider + DamageProvider>(deps: &T) {
    let health = deps.health();
    let damage = deps.damage();
    // Logic here
}
```

Tests provide mock implementations without heap allocation or dynamic dispatch.

### Builder Pattern for Test Fixtures

For complex entity setup, use builders:

```rust
struct EntityBuilder {
    health: u32,
    position: Option<Position>,
    velocity: Option<Velocity>,
}

impl EntityBuilder {
    fn new(health: u32) -> Self {
        Self { health, position: None, velocity: None }
    }

    fn with_position(mut self, pos: Position) -> Self {
        self.position = Some(pos);
        self
    }

    fn with_velocity(mut self, vel: Velocity) -> Self {
        self.velocity = Some(vel);
        self
    }

    fn spawn(self, world: &mut World) -> Entity {
        let mut entity = world.spawn(Health { value: self.health });
        if let Some(pos) = self.position {
            entity.insert(pos);
        }
        if let Some(vel) = self.velocity {
            entity.insert(vel);
        }
        entity.id()
    }
}

// In tests:
let entity = EntityBuilder::new(100)
    .with_position(Position::default())
    .spawn(&mut app.world_mut());
```

### World Setup Helpers

Create reusable test utilities:

```rust
fn setup_test_world() -> World {
    let mut world = World::new();
    world.insert_resource(GameTime::default());
    world.insert_resource(Config::test_config());
    world
}

fn spawn_test_player(world: &mut World) -> Entity {
    world.spawn((
        Player,
        Health { value: 100 },
        Position::default(),
    )).id()
}
```

See `references/testability_architecture.md` for:
- Complete dependency injection patterns
- Test helper organization
- Fixture builder examples
- World setup best practices

## Common Pitfalls and Anti-Patterns

### Anti-Pattern: Testing Implementation Details

**Don't**: Test internal component structure
```rust
// Bad: tests internal implementation
assert_eq!(entity.get::<InternalState>(), Some(&InternalState::Processing));
```

**Do**: Test observable behavior
```rust
// Good: tests behavior
assert!(query.iter().any(|e| e.has_component::<OutputReady>()));
```

### Anti-Pattern: Timing-Based Concurrency Tests

**Don't**: Use sleeps to "ensure" ordering
```rust
// Bad: flaky test
system1.run();
std::thread::sleep(Duration::from_millis(100));
assert!(state_changed);
```

**Do**: Use deterministic synchronization
```rust
// Good: explicit synchronization
system1.run();
barrier.wait();
assert!(state_changed);
```

### Anti-Pattern: Overly Complex Test Setup

**Don't**: Recreate entire game state in every test
```rust
// Bad: test setup obscures intent
let world = create_full_game_world();
world.add_all_systems();
world.insert_all_resources();
// What is this test actually testing?
```

**Do**: Minimal setup focused on test intent
```rust
// Good: clear test purpose
let mut app = App::new();
app.world_mut().spawn(Health { value: 0 });
app.add_systems(Update, remove_dead_entities);
// Testing: dead entities are removed
```

See `references/common_pitfalls.md` for comprehensive anti-patterns guide.

## Testing Workflow Checklist

When implementing tests for ECS systems:

1. **Identify test level**: Unit (single system), integration (multiple systems), or property-based?
2. **Set up minimal world**: Only required components, resources, and entities
3. **Use builders for complex fixtures**: Create reusable builders for common setups
4. **Test behavior, not implementation**: Verify outputs and side effects, not internal state
5. **For concurrent systems**: Use Loom or managed threads for deterministic testing
6. **For state machines**: Consider property-based testing with state machine framework
7. **Verify cleanup**: Test shutdown behavior and resource cleanup
8. **Check invariants**: Ensure ECS invariants hold after system execution

## References

This skill includes detailed reference documentation:

- `references/unit_testing_patterns.md` - Comprehensive examples of testing individual systems with various query patterns, event handling, and command execution
- `references/integration_testing.md` - Multi-system testing patterns, system chains, state transitions, and testing system dependencies
- `references/concurrent_testing.md` - Loom patterns, deterministic concurrency testing, race condition detection, and shutdown safety testing
- `references/property_based_testing.md` - Using proptest and quickcheck with ECS, state machine testing, custom generators, and shrinking strategies
- `references/testability_architecture.md` - Dependency injection patterns, the "Deps-Pattern", builder patterns for fixtures, and structuring code for testability
- `references/common_pitfalls.md` - Anti-patterns to avoid, common mistakes, and solutions for typical ECS testing challenges

Load these references when working on specific testing scenarios for detailed examples and patterns.
