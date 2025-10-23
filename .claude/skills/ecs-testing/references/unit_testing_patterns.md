# Unit Testing Patterns for ECS Systems

This reference provides comprehensive examples of testing individual ECS systems in isolation.

## Basic Query Testing

Test systems that read and modify components through queries:

```rust
use bevy::prelude::*;

#[derive(Component)]
struct Health { value: u32 }

#[derive(Component)]
struct Damage { value: u32 }

fn apply_damage_system(mut query: Query<(&mut Health, &Damage)>) {
    for (mut health, damage) in query.iter_mut() {
        health.value = health.value.saturating_sub(damage.value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_reduces_health() {
        let mut app = App::new();

        // Spawn test entity
        app.world_mut().spawn((
            Health { value: 100 },
            Damage { value: 25 },
        ));

        // Run system
        app.add_systems(Update, apply_damage_system);
        app.update();

        // Verify result
        let mut query = app.world_mut().query::<&Health>();
        let health = query.single(app.world());
        assert_eq!(health.value, 75);
    }

    #[test]
    fn test_damage_cannot_go_negative() {
        let mut app = App::new();

        app.world_mut().spawn((
            Health { value: 10 },
            Damage { value: 50 },
        ));

        app.add_systems(Update, apply_damage_system);
        app.update();

        let mut query = app.world_mut().query::<&Health>();
        let health = query.single(app.world());
        assert_eq!(health.value, 0);  // saturating_sub prevents underflow
    }
}
```

## Testing Queries with Filters

Use query filters to test conditional logic:

```rust
#[derive(Component)]
struct Dead;

fn mark_dead_entities(
    mut commands: Commands,
    query: Query<(Entity, &Health), Without<Dead>>,
) {
    for (entity, health) in query.iter() {
        if health.value == 0 {
            commands.entity(entity).insert(Dead);
        }
    }
}

#[test]
fn test_zero_health_entities_marked_dead() {
    let mut app = App::new();

    // Entity that should be marked dead
    let dead_entity = app.world_mut().spawn(Health { value: 0 }).id();

    // Entity that should remain alive
    let alive_entity = app.world_mut().spawn(Health { value: 50 }).id();

    app.add_systems(Update, mark_dead_entities);
    app.update();

    // Verify dead entity has Dead component
    assert!(app.world().entity(dead_entity).contains::<Dead>());

    // Verify alive entity does not
    assert!(!app.world().entity(alive_entity).contains::<Dead>());
}

#[test]
fn test_already_dead_entities_not_reprocessed() {
    let mut app = App::new();

    let entity = app.world_mut().spawn((
        Health { value: 0 },
        Dead,
    )).id();

    // Create a system that would panic if called
    fn should_not_run(query: Query<&Health, Without<Dead>>) {
        for _ in query.iter() {
            panic!("Should not process already dead entities");
        }
    }

    app.add_systems(Update, should_not_run);
    app.update();  // Should not panic
}
```

## Testing with Resources

Test systems that read or mutate global resources:

```rust
#[derive(Resource)]
struct GameTime {
    elapsed: f32,
    delta: f32,
}

fn update_time_system(mut time: ResMut<GameTime>) {
    time.elapsed += time.delta;
}

#[test]
fn test_time_accumulation() {
    let mut app = App::new();

    app.insert_resource(GameTime {
        elapsed: 0.0,
        delta: 0.016,  // ~60 FPS
    });

    app.add_systems(Update, update_time_system);

    // Run multiple updates
    for _ in 0..60 {
        app.update();
    }

    let time = app.world().resource::<GameTime>();
    assert!((time.elapsed - 0.96).abs() < 0.001);  // ~1 second
}

#[test]
fn test_system_without_required_resource() {
    let mut app = App::new();

    // Note: Not inserting GameTime resource
    app.add_systems(Update, update_time_system);

    // This will panic - resource required but missing
    let result = std::panic::catch_unwind(|| {
        app.update();
    });

    assert!(result.is_err());
}
```

## Testing Event Handling

Test systems that read or write events:

```rust
#[derive(Event)]
struct DamageEvent {
    target: Entity,
    amount: u32,
}

fn process_damage_events(
    mut events: EventReader<DamageEvent>,
    mut query: Query<&mut Health>,
) {
    for event in events.read() {
        if let Ok(mut health) = query.get_mut(event.target) {
            health.value = health.value.saturating_sub(event.amount);
        }
    }
}

#[test]
fn test_damage_events_applied() {
    let mut app = App::new();
    app.add_event::<DamageEvent>();

    let entity = app.world_mut().spawn(Health { value: 100 }).id();

    // Send damage events
    app.world_mut().send_event(DamageEvent {
        target: entity,
        amount: 25,
    });
    app.world_mut().send_event(DamageEvent {
        target: entity,
        amount: 30,
    });

    app.add_systems(Update, process_damage_events);
    app.update();

    let health = app.world().entity(entity).get::<Health>().unwrap();
    assert_eq!(health.value, 45);  // 100 - 25 - 30
}

#[test]
fn test_events_for_missing_entities_ignored() {
    let mut app = App::new();
    app.add_event::<DamageEvent>();

    // Create entity but don't spawn it in world
    let fake_entity = Entity::from_raw(9999);

    app.world_mut().send_event(DamageEvent {
        target: fake_entity,
        amount: 25,
    });

    app.add_systems(Update, process_damage_events);

    // Should not panic, just ignore event
    app.update();
}
```

## Testing Event Writers

Test systems that generate events:

```rust
#[derive(Event)]
struct EnemyDied {
    entity: Entity,
    score: u32,
}

fn check_for_deaths(
    query: Query<(Entity, &Health)>,
    mut events: EventWriter<EnemyDied>,
) {
    for (entity, health) in query.iter() {
        if health.value == 0 {
            events.send(EnemyDied {
                entity,
                score: 100,
            });
        }
    }
}

#[test]
fn test_death_events_generated() {
    let mut app = App::new();
    app.add_event::<EnemyDied>();

    let dead1 = app.world_mut().spawn(Health { value: 0 }).id();
    let dead2 = app.world_mut().spawn(Health { value: 0 }).id();
    let alive = app.world_mut().spawn(Health { value: 50 }).id();

    app.add_systems(Update, check_for_deaths);
    app.update();

    // Read events
    let events = app.world().resource::<Events<EnemyDied>>();
    let mut reader = events.get_reader();
    let event_list: Vec<_> = reader.read(events).collect();

    assert_eq!(event_list.len(), 2);
    assert!(event_list.iter().any(|e| e.entity == dead1));
    assert!(event_list.iter().any(|e| e.entity == dead2));
    assert!(!event_list.iter().any(|e| e.entity == alive));
}
```

## Testing Commands

Test systems using Commands for deferred operations:

```rust
fn despawn_dead_system(
    mut commands: Commands,
    query: Query<(Entity, &Health)>,
) {
    for (entity, health) in query.iter() {
        if health.value == 0 {
            commands.entity(entity).despawn();
        }
    }
}

#[test]
fn test_dead_entities_despawned() {
    let mut app = App::new();

    let dead = app.world_mut().spawn(Health { value: 0 }).id();
    let alive = app.world_mut().spawn(Health { value: 50 }).id();

    app.add_systems(Update, despawn_dead_system);
    app.update();  // Commands execute at end of update

    // Dead entity should be gone
    assert!(app.world().get_entity(dead).is_err());

    // Alive entity should remain
    assert!(app.world().get_entity(alive).is_ok());
}

#[test]
fn test_spawning_via_commands() {
    fn spawn_system(mut commands: Commands) {
        for _ in 0..5 {
            commands.spawn(Health { value: 100 });
        }
    }

    let mut app = App::new();
    app.add_systems(Update, spawn_system);

    // Before update: no entities
    assert_eq!(app.world().entities().len(), 0);

    app.update();

    // After update: commands executed
    let count = app.world().query::<&Health>().iter(app.world()).count();
    assert_eq!(count, 5);
}
```

## Testing Input Handling

Test systems that respond to input:

```rust
use bevy::input::ButtonInput;
use bevy::input::keyboard::KeyCode;

#[derive(Component)]
struct Player;

fn spawn_bullet_on_space(
    input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    query: Query<&Transform, With<Player>>,
) {
    if input.just_pressed(KeyCode::Space) {
        for transform in query.iter() {
            commands.spawn((
                Bullet,
                Transform::from_translation(transform.translation),
            ));
        }
    }
}

#[test]
fn test_bullet_spawned_on_space() {
    let mut app = App::new();

    app.world_mut().spawn((
        Player,
        Transform::default(),
    ));

    // Setup input
    let mut input = ButtonInput::<KeyCode>::default();
    input.press(KeyCode::Space);
    app.insert_resource(input);

    app.add_systems(Update, spawn_bullet_on_space);
    app.update();

    // Verify bullet spawned
    let count = app.world().query::<&Bullet>().iter(app.world()).count();
    assert_eq!(count, 1);
}

#[test]
fn test_no_bullet_without_input() {
    let mut app = App::new();

    app.world_mut().spawn((Player, Transform::default()));
    app.insert_resource(ButtonInput::<KeyCode>::default());

    app.add_systems(Update, spawn_bullet_on_space);
    app.update();

    let count = app.world().query::<&Bullet>().iter(app.world()).count();
    assert_eq!(count, 0);
}

#[test]
fn test_just_pressed_only_triggers_once() {
    let mut app = App::new();

    app.world_mut().spawn((Player, Transform::default()));

    let mut input = ButtonInput::<KeyCode>::default();
    input.press(KeyCode::Space);
    app.insert_resource(input);

    app.add_systems(Update, spawn_bullet_on_space);

    // First update: should spawn
    app.update();
    assert_eq!(app.world().query::<&Bullet>().iter(app.world()).count(), 1);

    // Second update: just_pressed is false now, no new bullet
    app.update();
    assert_eq!(app.world().query::<&Bullet>().iter(app.world()).count(), 1);
}
```

## Testing Optional Components

Test systems handling entities with varying component sets:

```rust
#[derive(Component)]
struct Armor { value: u32 }

fn apply_damage_with_armor(
    mut query: Query<(&mut Health, &Damage, Option<&Armor>)>,
) {
    for (mut health, damage, armor) in query.iter_mut() {
        let damage_taken = match armor {
            Some(armor) => damage.value.saturating_sub(armor.value),
            None => damage.value,
        };
        health.value = health.value.saturating_sub(damage_taken);
    }
}

#[test]
fn test_armor_reduces_damage() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 50 },
        Armor { value: 20 },
    ));

    app.add_systems(Update, apply_damage_with_armor);
    app.update();

    let mut query = app.world_mut().query::<&Health>();
    let health = query.single(app.world());
    assert_eq!(health.value, 70);  // 100 - (50 - 20)
}

#[test]
fn test_no_armor_full_damage() {
    let mut app = App::new();

    app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 50 },
        // No Armor component
    ));

    app.add_systems(Update, apply_damage_with_armor);
    app.update();

    let mut query = app.world_mut().query::<&Health>();
    let health = query.single(app.world());
    assert_eq!(health.value, 50);  // Full damage
}
```

## Testing Panics and Error Conditions

Test that systems properly handle invalid states:

```rust
fn divide_health_system(mut query: Query<&mut Health>, divisor: Res<Divisor>) {
    assert!(divisor.value > 0, "Divisor must be positive");

    for mut health in query.iter_mut() {
        health.value /= divisor.value;
    }
}

#[test]
#[should_panic(expected = "Divisor must be positive")]
fn test_zero_divisor_panics() {
    let mut app = App::new();

    app.world_mut().spawn(Health { value: 100 });
    app.insert_resource(Divisor { value: 0 });

    app.add_systems(Update, divide_health_system);
    app.update();  // Should panic
}

#[test]
fn test_valid_division() {
    let mut app = App::new();

    app.world_mut().spawn(Health { value: 100 });
    app.insert_resource(Divisor { value: 2 });

    app.add_systems(Update, divide_health_system);
    app.update();

    let mut query = app.world_mut().query::<&Health>();
    let health = query.single(app.world());
    assert_eq!(health.value, 50);
}
```

## Testing Multiple Entities

Test systems that process varying numbers of entities:

```rust
#[test]
fn test_system_with_no_entities() {
    let mut app = App::new();

    // No entities spawned
    app.add_systems(Update, apply_damage_system);

    // Should handle gracefully
    app.update();
}

#[test]
fn test_system_with_many_entities() {
    let mut app = App::new();

    // Spawn many entities
    for i in 0..1000 {
        app.world_mut().spawn((
            Health { value: 100 },
            Damage { value: i % 10 },
        ));
    }

    app.add_systems(Update, apply_damage_system);
    app.update();

    // Verify all processed
    let query = app.world().query::<&Health>();
    assert_eq!(query.iter(app.world()).count(), 1000);
}
```

## Best Practices

1. **Minimal Setup**: Only spawn entities and resources required for the test
2. **Clear Intent**: Test name and setup should make the test purpose obvious
3. **Single Assertion**: Each test should verify one behavior
4. **Isolation**: Tests should not depend on each other or shared state
5. **Use Single/Get**: Use `.single()` when expecting exactly one entity, `.get()` for optional
6. **Test Edge Cases**: Zero entities, missing components, boundary values
7. **Verify Side Effects**: Check that commands executed, events sent, resources mutated

## Common Patterns

Pattern for testing entity spawning:
```rust
let count_before = app.world().entities().len();
app.update();
let count_after = app.world().entities().len();
assert_eq!(count_after - count_before, expected_new_entities);
```

Pattern for testing component insertion:
```rust
let entity = app.world_mut().spawn(ComponentA).id();
app.update();
assert!(app.world().entity(entity).contains::<ComponentB>());
```

Pattern for testing resource mutation:
```rust
let before = app.world().resource::<MyResource>().clone();
app.update();
let after = app.world().resource::<MyResource>();
assert_ne!(*after, before);
```
