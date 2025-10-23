# Property-Based Testing for ECS

Property-based testing validates that code maintains certain invariants across randomly generated inputs. This is particularly powerful for ECS systems where state spaces are large and edge cases numerous.

## Core Concepts

Rather than testing specific examples, property-based testing:
1. Generates many random inputs
2. Verifies properties (invariants) hold for all inputs
3. Automatically shrinks failing cases to minimal examples

## Using Proptest

Add to `Cargo.toml`:
```toml
[dev-dependencies]
proptest = "1.4"
```

### Basic Property Test

```rust
use proptest::prelude::*;

#[derive(Component)]
struct Health { value: u32 }

#[derive(Component)]
struct Damage { value: u32 }

fn apply_damage_system(mut query: Query<(&mut Health, &Damage)>) {
    for (mut health, damage) in query.iter_mut() {
        health.value = health.value.saturating_sub(damage.value);
    }
}

proptest! {
    #[test]
    fn health_never_negative(
        initial_health in 0u32..1000,
        damage in 0u32..2000
    ) {
        let mut app = App::new();

        app.world_mut().spawn((
            Health { value: initial_health },
            Damage { value: damage },
        ));

        app.add_systems(Update, apply_damage_system);
        app.update();

        // Property: health can never be negative
        for health in app.world().query::<&Health>().iter(app.world()) {
            prop_assert!(health.value <= initial_health);
            prop_assert!(health.value == initial_health.saturating_sub(damage));
        }
    }
}
```

### Testing with Multiple Entities

```rust
proptest! {
    #[test]
    fn damage_applies_to_all_entities(
        entities: Vec<(u32, u32)>  // Vec of (health, damage) pairs
    ) {
        prop_assume!(!entities.is_empty());
        prop_assume!(entities.len() <= 100);

        let mut app = App::new();

        // Spawn entities
        for (health, damage) in &entities {
            app.world_mut().spawn((
                Health { value: *health },
                Damage { value: *damage },
            ));
        }

        app.add_systems(Update, apply_damage_system);
        app.update();

        // Verify property for each entity
        let results: Vec<_> = app.world()
            .query::<&Health>()
            .iter(app.world())
            .map(|h| h.value)
            .collect();

        prop_assert_eq!(results.len(), entities.len());

        for (i, (initial_health, damage)) in entities.iter().enumerate() {
            let expected = initial_health.saturating_sub(*damage);
            prop_assert_eq!(results[i], expected);
        }
    }
}
```

## Custom Arbitrary Implementations

Create generators for domain-specific types:

```rust
use proptest::prelude::*;

#[derive(Debug, Clone)]
struct ValidEntity {
    health: u32,
    damage: u32,
    position: (f32, f32),
}

impl Arbitrary for ValidEntity {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (
            1u32..=1000,  // Health: 1-1000
            0u32..=100,   // Damage: 0-100
            (-100.0f32..100.0, -100.0f32..100.0),  // Position
        )
            .prop_map(|(health, damage, position)| ValidEntity {
                health,
                damage,
                position,
            })
            .boxed()
    }
}

proptest! {
    #[test]
    fn test_with_valid_entities(entities: Vec<ValidEntity>) {
        prop_assume!(entities.len() <= 50);

        let mut app = App::new();

        for entity in entities {
            app.world_mut().spawn((
                Health { value: entity.health },
                Damage { value: entity.damage },
                Position { x: entity.position.0, y: entity.position.1 },
            ));
        }

        // Test system behavior
        app.add_systems(Update, apply_damage_system);
        app.update();

        // Verify invariants
        for (health, position) in app.world().query::<(&Health, &Position)>().iter(app.world()) {
            prop_assert!(health.value >= 0);
            prop_assert!(position.x.abs() <= 100.0);
            prop_assert!(position.y.abs() <= 100.0);
        }
    }
}
```

## Testing State Machines

Use proptest's state machine testing for complex transitions:

```rust
use proptest::state_machine::{ReferenceStateMachine, StateMachineTest};
use proptest::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum EntityState {
    Idle,
    Moving,
    Attacking,
    Dead,
}

#[derive(Debug, Clone)]
enum Transition {
    StartMoving,
    StopMoving,
    Attack,
    TakeDamage(u32),
}

#[derive(Clone, Debug)]
struct EntityModel {
    state: EntityState,
    health: u32,
}

impl ReferenceStateMachine for EntityModel {
    type State = Self;
    type Transition = Transition;

    fn init_state() -> BoxedStrategy<Self::State> {
        (prop_oneof![
            Just(EntityState::Idle),
            Just(EntityState::Moving),
        ], 1u32..=100)
            .prop_map(|(state, health)| EntityModel { state, health })
            .boxed()
    }

    fn transitions(state: &Self::State) -> BoxedStrategy<Self::Transition> {
        if state.health == 0 {
            // Dead state has no transitions
            return Just(Transition::TakeDamage(0)).boxed();
        }

        match state.state {
            EntityState::Idle => prop_oneof![
                Just(Transition::StartMoving),
                Just(Transition::Attack),
                (1u32..=50).prop_map(Transition::TakeDamage),
            ].boxed(),
            EntityState::Moving => prop_oneof![
                Just(Transition::StopMoving),
                Just(Transition::Attack),
                (1u32..=50).prop_map(Transition::TakeDamage),
            ].boxed(),
            EntityState::Attacking => prop_oneof![
                Just(Transition::StartMoving),
                (1u32..=50).prop_map(Transition::TakeDamage),
            ].boxed(),
            EntityState::Dead => Just(Transition::TakeDamage(0)).boxed(),
        }
    }

    fn apply(mut state: Self::State, transition: &Self::Transition) -> Self::State {
        match transition {
            Transition::StartMoving if state.state != EntityState::Dead => {
                state.state = EntityState::Moving;
            }
            Transition::StopMoving if state.state == EntityState::Moving => {
                state.state = EntityState::Idle;
            }
            Transition::Attack if state.state != EntityState::Dead => {
                state.state = EntityState::Attacking;
            }
            Transition::TakeDamage(damage) => {
                state.health = state.health.saturating_sub(*damage);
                if state.health == 0 {
                    state.state = EntityState::Dead;
                }
            }
            _ => {}
        }
        state
    }

    fn preconditions(state: &Self::State, transition: &Self::Transition) -> bool {
        match transition {
            Transition::TakeDamage(_) => true,
            _ => state.state != EntityState::Dead,
        }
    }
}

proptest! {
    #[test]
    fn test_state_machine_invariants(
        transitions in proptest::collection::vec(any::<Transition>(), 0..20)
    ) {
        let mut model = EntityModel {
            state: EntityState::Idle,
            health: 100,
        };

        let mut app = App::new();
        let entity = app.world_mut().spawn((
            State::Idle,
            Health { value: 100 },
        )).id();

        for transition in transitions {
            if !EntityModel::preconditions(&model, &transition) {
                continue;
            }

            // Apply to model
            model = EntityModel::apply(model, &transition);

            // Apply to ECS
            apply_transition_to_ecs(&mut app, entity, &transition);
            app.update();

            // Verify ECS matches model
            let ecs_state = app.world().entity(entity).get::<State>().unwrap();
            let ecs_health = app.world().entity(entity).get::<Health>().unwrap();

            prop_assert_eq!(model.health, ecs_health.value);
            prop_assert!(states_match(&model.state, ecs_state));
        }
    }
}

fn apply_transition_to_ecs(app: &mut App, entity: Entity, transition: &Transition) {
    // Helper to apply transition to ECS entity
    match transition {
        Transition::StartMoving => {
            app.world_mut().entity_mut(entity).insert(State::Moving);
        }
        Transition::StopMoving => {
            app.world_mut().entity_mut(entity).insert(State::Idle);
        }
        Transition::Attack => {
            app.world_mut().entity_mut(entity).insert(State::Attacking);
        }
        Transition::TakeDamage(damage) => {
            let mut health = app.world_mut().entity_mut(entity)
                .get_mut::<Health>()
                .unwrap();
            health.value = health.value.saturating_sub(*damage);
        }
    }
}
```

## Testing Invariants

Define and test properties that must always hold:

```rust
proptest! {
    #[test]
    fn entity_count_invariant(
        spawn_count in 0usize..100,
        despawn_count in 0usize..100
    ) {
        let mut app = App::new();

        // Spawn entities
        for _ in 0..spawn_count {
            app.world_mut().spawn(Health { value: 100 });
        }

        // Despawn some (but not more than exist)
        let to_despawn = std::cmp::min(despawn_count, spawn_count);
        let entities: Vec<_> = app.world()
            .query::<Entity>()
            .iter(app.world())
            .take(to_despawn)
            .collect();

        for entity in entities {
            app.world_mut().entity_mut(entity).despawn();
        }

        // Invariant: entity count matches expected
        let actual_count = app.world().query::<&Health>().iter(app.world()).count();
        prop_assert_eq!(actual_count, spawn_count - to_despawn);
    }
}
```

## Testing System Idempotence

Verify running a system multiple times produces same result:

```rust
fn normalize_health_system(mut query: Query<&mut Health>) {
    for mut health in query.iter_mut() {
        if health.value > 100 {
            health.value = 100;
        }
    }
}

proptest! {
    #[test]
    fn normalize_is_idempotent(health_values: Vec<u32>) {
        prop_assume!(!health_values.is_empty());
        prop_assume!(health_values.len() <= 50);

        let mut app = App::new();

        for health in &health_values {
            app.world_mut().spawn(Health { value: *health });
        }

        app.add_systems(Update, normalize_health_system);

        // Run once
        app.update();
        let first_run: Vec<_> = app.world()
            .query::<&Health>()
            .iter(app.world())
            .map(|h| h.value)
            .collect();

        // Run again
        app.update();
        let second_run: Vec<_> = app.world()
            .query::<&Health>()
            .iter(app.world())
            .map(|h| h.value)
            .collect();

        // Property: second run should be identical
        prop_assert_eq!(first_run, second_run);

        // Property: all values should be <= 100
        for value in first_run {
            prop_assert!(value <= 100);
        }
    }
}
```

## Testing Commutative Operations

Verify order independence:

```rust
proptest! {
    #[test]
    fn damage_order_independent(
        damages: Vec<u32>
    ) {
        prop_assume!(!damages.is_empty());
        prop_assume!(damages.len() <= 10);

        let initial_health = 1000u32;

        // Test with original order
        let mut app1 = App::new();
        let entity1 = app1.world_mut().spawn(Health { value: initial_health }).id();

        for damage in &damages {
            let mut health = app1.world_mut().entity_mut(entity1)
                .get_mut::<Health>()
                .unwrap();
            health.value = health.value.saturating_sub(*damage);
        }

        let result1 = app1.world().entity(entity1).get::<Health>().unwrap().value;

        // Test with reversed order
        let mut app2 = App::new();
        let entity2 = app2.world_mut().spawn(Health { value: initial_health }).id();

        for damage in damages.iter().rev() {
            let mut health = app2.world_mut().entity_mut(entity2)
                .get_mut::<Health>()
                .unwrap();
            health.value = health.value.saturating_sub(*damage);
        }

        let result2 = app2.world().entity(entity2).get::<Health>().unwrap().value;

        // Property: order shouldn't matter for independent damage
        prop_assert_eq!(result1, result2);
    }
}
```

## Generating Complex Scenarios

Use strategies to create realistic test scenarios:

```rust
#[derive(Debug, Clone)]
struct GameScenario {
    players: Vec<Entity>,
    enemies: Vec<Entity>,
    events: Vec<GameEvent>,
}

#[derive(Debug, Clone)]
enum GameEvent {
    PlayerAttack { player: usize, enemy: usize },
    EnemyAttack { enemy: usize, player: usize },
    Heal { player: usize, amount: u32 },
}

fn game_scenario_strategy() -> impl Strategy<Value = GameScenario> {
    (1usize..=4, 1usize..=10).prop_flat_map(|(num_players, num_enemies)| {
        let events = proptest::collection::vec(
            prop_oneof![
                (0..num_players, 0..num_enemies).prop_map(|(p, e)| {
                    GameEvent::PlayerAttack { player: p, enemy: e }
                }),
                (0..num_enemies, 0..num_players).prop_map(|(e, p)| {
                    GameEvent::EnemyAttack { enemy: e, player: p }
                }),
                (0..num_players, 1u32..50).prop_map(|(p, amount)| {
                    GameEvent::Heal { player: p, amount }
                }),
            ],
            0..20,
        );

        events.prop_map(move |events| GameScenario {
            players: (0..num_players).map(|_| Entity::from_raw(0)).collect(),
            enemies: (0..num_enemies).map(|_| Entity::from_raw(0)).collect(),
            events,
        })
    })
}

proptest! {
    #[test]
    fn test_game_scenario(scenario in game_scenario_strategy()) {
        let mut app = App::new();

        // Setup scenario
        let players: Vec<_> = (0..scenario.players.len())
            .map(|_| app.world_mut().spawn((Player, Health { value: 100 })).id())
            .collect();

        let enemies: Vec<_> = (0..scenario.enemies.len())
            .map(|_| app.world_mut().spawn((Enemy, Health { value: 50 })).id())
            .collect();

        // Execute events
        for event in &scenario.events {
            match event {
                GameEvent::PlayerAttack { player, enemy } => {
                    if *player < players.len() && *enemy < enemies.len() {
                        // Apply attack logic
                    }
                }
                GameEvent::EnemyAttack { enemy, player } => {
                    if *enemy < enemies.len() && *player < players.len() {
                        // Apply attack logic
                    }
                }
                GameEvent::Heal { player, amount } => {
                    if *player < players.len() {
                        let mut health = app.world_mut()
                            .entity_mut(players[*player])
                            .get_mut::<Health>()
                            .unwrap();
                        health.value = (health.value + amount).min(100);
                    }
                }
            }
        }

        // Verify invariants
        for player in players {
            if let Ok(health) = app.world().entity(player).get::<Health>() {
                prop_assert!(health.value <= 100);
            }
        }
    }
}
```

## Shrinking to Minimal Examples

Proptest automatically shrinks failing cases:

```rust
proptest! {
    #[test]
    fn test_with_shrinking(values: Vec<u32>) {
        // This will fail and shrink to minimal case
        prop_assume!(!values.is_empty());

        let sum: u32 = values.iter().sum();

        // This will fail for large sums, shrinking to minimal failing case
        prop_assert!(sum < 1000);
    }
}
```

## Using QuickCheck (Alternative)

QuickCheck is simpler but less flexible:

```rust
use quickcheck::{quickcheck, TestResult};

#[quickcheck]
fn health_bounded(initial: u32, damage: u32) -> TestResult {
    if initial > 10000 || damage > 10000 {
        return TestResult::discard();
    }

    let mut app = App::new();
    app.world_mut().spawn((
        Health { value: initial },
        Damage { value: damage },
    ));

    app.add_systems(Update, apply_damage_system);
    app.update();

    let health = app.world().query::<&Health>().single(app.world());
    TestResult::from_bool(health.value <= initial)
}
```

## Best Practices

1. **Use prop_assume!**: Filter out invalid inputs early
2. **Limit Collection Sizes**: Cap vectors at reasonable sizes (< 100)
3. **Test Invariants**: Focus on properties that must always hold
4. **Custom Generators**: Create domain-specific Arbitrary implementations
5. **State Machine Testing**: Model complex transitions explicitly
6. **Shrinking**: Let proptest find minimal failing cases
7. **Idempotence**: Test that repeated application doesn't change results
8. **Commutativity**: Verify order independence where applicable

## Common Patterns

Pattern for bounded values:
```rust
proptest! {
    #[test]
    fn test(value in 0u32..1000) {
        // value guaranteed in range
    }
}
```

Pattern for filtering:
```rust
proptest! {
    #[test]
    fn test(value: u32) {
        prop_assume!(value > 0);
        prop_assume!(value < 1000);
        // value satisfies conditions
    }
}
```

Pattern for multiple properties:
```rust
proptest! {
    #[test]
    fn test(input: Input) {
        let result = system(input);
        prop_assert!(property1(&result));
        prop_assert!(property2(&result));
        prop_assert!(property3(&result));
    }
}
```
