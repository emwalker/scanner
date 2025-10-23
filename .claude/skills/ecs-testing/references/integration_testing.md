# Integration Testing for ECS Systems

This reference covers testing multiple interacting systems together to verify complex behaviors, execution order, and shared state management.

## Testing System Chains

Explicitly chain systems to test execution order:

```rust
#[derive(Component)]
struct Health { value: u32 }

#[derive(Component)]
struct Damage { value: u32 }

#[derive(Component)]
struct Dead;

fn apply_damage(mut query: Query<(&mut Health, &Damage)>) {
    for (mut health, damage) in query.iter_mut() {
        health.value = health.value.saturating_sub(damage.value);
    }
}

fn mark_dead(mut commands: Commands, query: Query<(Entity, &Health), Without<Dead>>) {
    for (entity, health) in query.iter() {
        if health.value == 0 {
            commands.entity(entity).insert(Dead);
        }
    }
}

fn despawn_dead(mut commands: Commands, query: Query<Entity, With<Dead>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_death_despawn_pipeline() {
        let mut app = App::new();

        // Entity with lethal damage
        let entity = app.world_mut().spawn((
            Health { value: 50 },
            Damage { value: 60 },
        )).id();

        // Chain systems in order
        app.add_systems(Update, (
            apply_damage,
            mark_dead,
            despawn_dead,
        ).chain());

        app.update();

        // Entity should be completely removed
        assert!(app.world().get_entity(entity).is_err());
    }

    #[test]
    fn test_system_ordering_matters() {
        let mut app = App::new();

        let entity = app.world_mut().spawn((
            Health { value: 50 },
            Damage { value: 60 },
        )).id();

        // Wrong order: despawn before marking dead
        app.add_systems(Update, (
            apply_damage,
            despawn_dead,  // Runs before entity is marked dead
            mark_dead,
        ).chain());

        app.update();

        // Entity still exists because despawn ran before Dead marker added
        assert!(app.world().get_entity(entity).is_ok());
        assert!(app.world().entity(entity).contains::<Dead>());
    }
}
```

## Testing System Dependencies

Test systems that depend on outputs from other systems:

```rust
#[derive(Resource)]
struct Score { value: u32 }

#[derive(Event)]
struct EnemyDied { points: u32 }

fn detect_deaths(
    query: Query<&Health>,
    mut events: EventWriter<EnemyDied>,
) {
    for health in query.iter() {
        if health.value == 0 {
            events.send(EnemyDied { points: 100 });
        }
    }
}

fn update_score(
    mut score: ResMut<Score>,
    mut events: EventReader<EnemyDied>,
) {
    for event in events.read() {
        score.value += event.points;
    }
}

#[test]
fn test_event_pipeline() {
    let mut app = App::new();
    app.add_event::<EnemyDied>();
    app.insert_resource(Score { value: 0 });

    // Spawn dead enemies
    app.world_mut().spawn(Health { value: 0 });
    app.world_mut().spawn(Health { value: 0 });
    app.world_mut().spawn(Health { value: 50 });  // Alive

    app.add_systems(Update, (
        detect_deaths,
        update_score,
    ).chain());

    app.update();

    let score = app.world().resource::<Score>();
    assert_eq!(score.value, 200);  // 2 enemies * 100 points
}
```

## Testing State Transitions

Test complex state machines spanning multiple systems:

```rust
#[derive(Component, PartialEq, Debug, Clone, Copy)]
enum State {
    Idle,
    Moving,
    Attacking,
}

#[derive(Component)]
struct Velocity { x: f32, y: f32 }

#[derive(Component)]
struct AttackTimer { remaining: f32 }

#[derive(Resource)]
struct Input {
    move_x: f32,
    move_y: f32,
    attack: bool,
}

fn update_state_from_input(
    mut query: Query<(&mut State, Option<&AttackTimer>)>,
    input: Res<Input>,
) {
    for (mut state, attack_timer) in query.iter_mut() {
        // Can't change state while attacking
        if attack_timer.is_some() {
            continue;
        }

        *state = if input.attack {
            State::Attacking
        } else if input.move_x != 0.0 || input.move_y != 0.0 {
            State::Moving
        } else {
            State::Idle
        };
    }
}

fn apply_state_effects(
    mut commands: Commands,
    mut query: Query<(Entity, &State, &mut Velocity)>,
    input: Res<Input>,
) {
    for (entity, state, mut velocity) in query.iter_mut() {
        match state {
            State::Idle => {
                velocity.x = 0.0;
                velocity.y = 0.0;
            }
            State::Moving => {
                velocity.x = input.move_x;
                velocity.y = input.move_y;
            }
            State::Attacking => {
                velocity.x = 0.0;
                velocity.y = 0.0;
                commands.entity(entity).insert(AttackTimer { remaining: 1.0 });
            }
        }
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[test]
    fn test_idle_to_moving_transition() {
        let mut app = App::new();

        let entity = app.world_mut().spawn((
            State::Idle,
            Velocity { x: 0.0, y: 0.0 },
        )).id();

        app.insert_resource(Input {
            move_x: 1.0,
            move_y: 0.0,
            attack: false,
        });

        app.add_systems(Update, (
            update_state_from_input,
            apply_state_effects,
        ).chain());

        app.update();

        let state = app.world().entity(entity).get::<State>().unwrap();
        assert_eq!(*state, State::Moving);

        let velocity = app.world().entity(entity).get::<Velocity>().unwrap();
        assert_eq!(velocity.x, 1.0);
        assert_eq!(velocity.y, 0.0);
    }

    #[test]
    fn test_attack_prevents_state_change() {
        let mut app = App::new();

        let entity = app.world_mut().spawn((
            State::Attacking,
            Velocity { x: 0.0, y: 0.0 },
            AttackTimer { remaining: 0.5 },
        )).id();

        // Try to move while attacking
        app.insert_resource(Input {
            move_x: 1.0,
            move_y: 0.0,
            attack: false,
        });

        app.add_systems(Update, update_state_from_input);
        app.update();

        // State should remain Attacking
        let state = app.world().entity(entity).get::<State>().unwrap();
        assert_eq!(*state, State::Attacking);
    }

    #[test]
    fn test_full_state_cycle() {
        let mut app = App::new();

        let entity = app.world_mut().spawn((
            State::Idle,
            Velocity { x: 0.0, y: 0.0 },
        )).id();

        app.add_systems(Update, (
            update_state_from_input,
            apply_state_effects,
        ).chain());

        // Start moving
        app.insert_resource(Input {
            move_x: 1.0,
            move_y: 0.0,
            attack: false,
        });
        app.update();
        assert_eq!(*app.world().entity(entity).get::<State>().unwrap(), State::Moving);

        // Start attacking
        app.insert_resource(Input {
            move_x: 0.0,
            move_y: 0.0,
            attack: true,
        });
        app.update();
        assert_eq!(*app.world().entity(entity).get::<State>().unwrap(), State::Attacking);
        assert!(app.world().entity(entity).contains::<AttackTimer>());

        // Return to idle (after removing timer)
        app.world_mut().entity_mut(entity).remove::<AttackTimer>();
        app.insert_resource(Input {
            move_x: 0.0,
            move_y: 0.0,
            attack: false,
        });
        app.update();
        assert_eq!(*app.world().entity(entity).get::<State>().unwrap(), State::Idle);
    }
}
```

## Testing Resource Contention

Test systems that compete for shared resources:

```rust
#[derive(Resource)]
struct EntityPool {
    available: Vec<Entity>,
    in_use: Vec<Entity>,
}

fn allocate_entity_system(
    mut pool: ResMut<EntityPool>,
    query: Query<Entity, With<NeedsEntity>>,
) {
    for entity in query.iter() {
        if let Some(pooled) = pool.available.pop() {
            pool.in_use.push(pooled);
        }
    }
}

fn release_entity_system(
    mut pool: ResMut<EntityPool>,
    query: Query<Entity, With<ReleaseEntity>>,
) {
    for entity in query.iter() {
        if let Some(pos) = pool.in_use.iter().position(|&e| e == entity) {
            let released = pool.in_use.remove(pos);
            pool.available.push(released);
        }
    }
}

#[test]
fn test_pool_allocation_and_release() {
    let mut app = App::new();

    // Create pool with entities
    let pooled1 = app.world_mut().spawn_empty().id();
    let pooled2 = app.world_mut().spawn_empty().id();

    app.insert_resource(EntityPool {
        available: vec![pooled1, pooled2],
        in_use: vec![],
    });

    // Spawn requestor
    let requestor = app.world_mut().spawn(NeedsEntity).id();

    app.add_systems(Update, (
        allocate_entity_system,
        release_entity_system,
    ).chain());

    app.update();

    // Verify allocation
    let pool = app.world().resource::<EntityPool>();
    assert_eq!(pool.available.len(), 1);
    assert_eq!(pool.in_use.len(), 1);

    // Mark for release
    app.world_mut().entity_mut(requestor).insert(ReleaseEntity);
    app.update();

    // Verify release
    let pool = app.world().resource::<EntityPool>();
    assert_eq!(pool.available.len(), 2);
    assert_eq!(pool.in_use.len(), 0);
}
```

## Testing Multi-Stage Pipelines

Test complex processing pipelines across multiple update cycles:

```rust
#[derive(Component)]
struct QueuedForProcessing;

#[derive(Component)]
struct Processing { progress: f32 }

#[derive(Component)]
struct Completed;

fn start_processing(
    mut commands: Commands,
    query: Query<Entity, With<QueuedForProcessing>>,
) {
    for entity in query.iter() {
        commands.entity(entity)
            .remove::<QueuedForProcessing>()
            .insert(Processing { progress: 0.0 });
    }
}

fn update_processing(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Processing)>,
) {
    for (entity, mut processing) in query.iter_mut() {
        processing.progress += 0.1;

        if processing.progress >= 1.0 {
            commands.entity(entity)
                .remove::<Processing>()
                .insert(Completed);
        }
    }
}

#[test]
fn test_processing_pipeline() {
    let mut app = App::new();

    let entity = app.world_mut().spawn(QueuedForProcessing).id();

    app.add_systems(Update, (
        start_processing,
        update_processing,
    ).chain());

    // First update: queue -> processing
    app.update();
    assert!(!app.world().entity(entity).contains::<QueuedForProcessing>());
    assert!(app.world().entity(entity).contains::<Processing>());

    // Updates 2-10: processing
    for _ in 0..9 {
        app.update();
        assert!(app.world().entity(entity).contains::<Processing>());
    }

    // Update 11: processing -> completed
    app.update();
    assert!(!app.world().entity(entity).contains::<Processing>());
    assert!(app.world().entity(entity).contains::<Completed>());
}

#[test]
fn test_multiple_entities_pipeline() {
    let mut app = App::new();

    // Spawn entities with staggered start
    let entity1 = app.world_mut().spawn(QueuedForProcessing).id();
    let entity2 = app.world_mut().spawn(QueuedForProcessing).id();

    app.add_systems(Update, (start_processing, update_processing).chain());

    // Both start processing
    app.update();
    assert!(app.world().entity(entity1).contains::<Processing>());
    assert!(app.world().entity(entity2).contains::<Processing>());

    // Add third entity mid-processing
    let entity3 = app.world_mut().spawn(QueuedForProcessing).id();

    // Continue processing
    for _ in 0..9 {
        app.update();
    }

    // entity1 and entity2 complete, entity3 still processing
    assert!(app.world().entity(entity1).contains::<Completed>());
    assert!(app.world().entity(entity2).contains::<Completed>());
    assert!(app.world().entity(entity3).contains::<Processing>());
}
```

## Testing Side Effects Across Systems

Test that one system's side effects correctly trigger another system:

```rust
#[derive(Event)]
struct SpawnRequest { count: u32 }

#[derive(Component)]
struct Enemy;

#[derive(Resource)]
struct SpawnedCount { value: u32 }

fn handle_spawn_requests(
    mut commands: Commands,
    mut events: EventReader<SpawnRequest>,
) {
    for request in events.read() {
        for _ in 0..request.count {
            commands.spawn(Enemy);
        }
    }
}

fn count_enemies(
    mut count: ResMut<SpawnedCount>,
    query: Query<&Enemy>,
) {
    count.value = query.iter().count() as u32;
}

#[test]
fn test_spawn_and_count_integration() {
    let mut app = App::new();
    app.add_event::<SpawnRequest>();
    app.insert_resource(SpawnedCount { value: 0 });

    // Send spawn request
    app.world_mut().send_event(SpawnRequest { count: 5 });

    app.add_systems(Update, (
        handle_spawn_requests,
        count_enemies,
    ).chain());

    app.update();

    // Verify both systems worked together
    let count = app.world().resource::<SpawnedCount>();
    assert_eq!(count.value, 5);

    let enemy_count = app.world().query::<&Enemy>().iter(app.world()).count();
    assert_eq!(enemy_count, 5);
}
```

## Testing Parallel System Execution

Test that systems can run in parallel without conflicts:

```rust
#[derive(Component)]
struct PositionA { x: f32 }

#[derive(Component)]
struct PositionB { y: f32 }

fn system_a(mut query: Query<&mut PositionA>) {
    for mut pos in query.iter_mut() {
        pos.x += 1.0;
    }
}

fn system_b(mut query: Query<&mut PositionB>) {
    for mut pos in query.iter_mut() {
        pos.y += 1.0;
    }
}

#[test]
fn test_parallel_systems_no_conflict() {
    let mut app = App::new();

    // Spawn entities with both components
    for _ in 0..10 {
        app.world_mut().spawn((
            PositionA { x: 0.0 },
            PositionB { y: 0.0 },
        ));
    }

    // These systems access disjoint component sets, can run parallel
    app.add_systems(Update, (system_a, system_b));

    app.update();

    // Verify both ran
    for (pos_a, pos_b) in app.world().query::<(&PositionA, &PositionB)>().iter(app.world()) {
        assert_eq!(pos_a.x, 1.0);
        assert_eq!(pos_b.y, 1.0);
    }
}
```

## Best Practices

1. **Test System Order**: Explicitly use `.chain()` and test that order matters
2. **Test Intermediate States**: Verify state between system executions
3. **Test Full Pipelines**: Run complete cycles from input to output
4. **Test Side Effects**: Verify one system's output correctly triggers another
5. **Test Multiple Updates**: Some behaviors only emerge over multiple frames
6. **Test Parallel Safety**: Ensure systems accessing different data can coexist
7. **Isolate Failures**: When integration test fails, write unit tests to identify which system

## Common Patterns

Pattern for testing multi-frame behavior:
```rust
for frame in 0..expected_frames {
    app.update();
    // Verify intermediate state at each frame
}
```

Pattern for testing event propagation:
```rust
app.add_systems(Update, (producer_system, consumer_system).chain());
app.update();
// Verify consumer processed producer's events
```

Pattern for testing state machine:
```rust
// Initial state
assert_eq!(get_state(&app), State::A);
// Trigger transition
trigger_event(&mut app);
app.update();
// Verify new state
assert_eq!(get_state(&app), State::B);
```
