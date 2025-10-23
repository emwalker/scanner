# Architectural Patterns for Testability

This reference covers code structure and design patterns that make ECS systems easy to test, including dependency injection, test fixtures, and separation of concerns.

## The Deps-Pattern: Zero-Cost Dependency Injection

The Deps-Pattern uses trait bounds and generics for dependency injection without runtime overhead.

### Basic Pattern

```rust
// Define trait for dependency
trait TimeProvider {
    fn elapsed(&self) -> f32;
    fn delta(&self) -> f32;
}

// System uses generic parameter
fn movement_system<T: TimeProvider>(
    deps: &T,
    mut query: Query<(&mut Position, &Velocity)>,
) {
    let dt = deps.delta();

    for (mut pos, vel) in query.iter_mut() {
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
    }
}

// Production implementation
struct RealTime {
    elapsed: f32,
    delta: f32,
}

impl TimeProvider for RealTime {
    fn elapsed(&self) -> f32 { self.elapsed }
    fn delta(&self) -> f32 { self.delta }
}

// Test implementation
struct MockTime {
    elapsed: f32,
    delta: f32,
}

impl TimeProvider for MockTime {
    fn elapsed(&self) -> f32 { self.elapsed }
    fn delta(&self) -> f32 { self.delta }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_with_mock_time() {
        let mock_time = MockTime {
            elapsed: 1.0,
            delta: 0.016,  // 60 FPS
        };

        let mut app = App::new();

        app.world_mut().spawn((
            Position { x: 0.0, y: 0.0 },
            Velocity { x: 10.0, y: 5.0 },
        ));

        movement_system(&mock_time, app.world_mut().query::<(&mut Position, &Velocity)>());

        let position = app.world().query::<&Position>().single(app.world());
        assert_eq!(position.x, 0.16);  // 10.0 * 0.016
        assert_eq!(position.y, 0.08);  // 5.0 * 0.016
    }
}
```

### Multiple Dependencies

Combine multiple traits for complex dependencies:

```rust
trait HealthProvider {
    fn max_health(&self) -> u32;
}

trait DamageProvider {
    fn base_damage(&self) -> u32;
    fn multiplier(&self) -> f32;
}

trait ConfigProvider: HealthProvider + DamageProvider {
    // Composite trait
}

fn combat_system<T: ConfigProvider>(
    deps: &T,
    mut query: Query<(&mut Health, &Damage)>,
) {
    let max_health = deps.max_health();
    let base_damage = deps.base_damage();
    let multiplier = deps.multiplier();

    for (mut health, damage) in query.iter_mut() {
        let actual_damage = (damage.value as f32 * multiplier) as u32 + base_damage;
        health.value = health.value.saturating_sub(actual_damage).min(max_health);
    }
}

// Test implementation
struct MockConfig;

impl HealthProvider for MockConfig {
    fn max_health(&self) -> u32 { 100 }
}

impl DamageProvider for MockConfig {
    fn base_damage(&self) -> u32 { 5 }
    fn multiplier(&self) -> f32 { 1.5 }
}

impl ConfigProvider for MockConfig {}
```

### Resource-Based Dependency Injection

For Bevy, wrap dependencies in resources:

```rust
trait RandomGenerator {
    fn next_u32(&mut self) -> u32;
    fn next_f32(&mut self) -> f32;
}

#[derive(Resource)]
struct RngResource<T: RandomGenerator> {
    generator: T,
}

fn spawn_random_enemies<T: RandomGenerator>(
    mut commands: Commands,
    mut rng: ResMut<RngResource<T>>,
) {
    let count = rng.generator.next_u32() % 10;

    for _ in 0..count {
        let health = rng.generator.next_u32() % 100;
        commands.spawn((
            Enemy,
            Health { value: health },
        ));
    }
}

// Mock generator for tests
struct PredictableRng {
    values: Vec<u32>,
    index: usize,
}

impl RandomGenerator for PredictableRng {
    fn next_u32(&mut self) -> u32 {
        let value = self.values[self.index];
        self.index = (self.index + 1) % self.values.len();
        value
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

#[test]
fn test_predictable_spawning() {
    let mut app = App::new();

    app.insert_resource(RngResource {
        generator: PredictableRng {
            values: vec![3, 50, 75, 25],
            index: 0,
        },
    });

    app.add_systems(Update, spawn_random_enemies);
    app.update();

    // Should spawn 3 enemies (first value: 3 % 10)
    let count = app.world().query::<&Enemy>().iter(app.world()).count();
    assert_eq!(count, 3);

    // With known health values: 50, 75, 25
    let healths: Vec<_> = app.world()
        .query::<&Health>()
        .iter(app.world())
        .map(|h| h.value)
        .collect();
    assert_eq!(healths, vec![50, 75, 25]);
}
```

## Builder Pattern for Test Fixtures

Create fluent builders for complex entity setup:

### Basic Builder

```rust
struct EntityBuilder {
    health: u32,
    position: Option<Position>,
    velocity: Option<Velocity>,
    tags: Vec<Tag>,
}

impl EntityBuilder {
    fn new(health: u32) -> Self {
        Self {
            health,
            position: None,
            velocity: None,
            tags: vec![],
        }
    }

    fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = Some(Position { x, y });
        self
    }

    fn with_velocity(mut self, x: f32, y: f32) -> Self {
        self.velocity = Some(Velocity { x, y });
        self
    }

    fn with_tag(mut self, tag: Tag) -> Self {
        self.tags.push(tag);
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

        for tag in self.tags {
            entity.insert(tag);
        }

        entity.id()
    }
}

#[test]
fn test_with_builder() {
    let mut app = App::new();

    let entity = EntityBuilder::new(100)
        .with_position(10.0, 20.0)
        .with_velocity(1.0, 0.0)
        .with_tag(Tag::Player)
        .spawn(&mut app.world_mut());

    assert!(app.world().entity(entity).contains::<Health>());
    assert!(app.world().entity(entity).contains::<Position>());
    assert!(app.world().entity(entity).contains::<Velocity>());
}
```

### Builder with Defaults

```rust
impl Default for EntityBuilder {
    fn default() -> Self {
        Self {
            health: 100,
            position: Some(Position { x: 0.0, y: 0.0 }),
            velocity: None,
            tags: vec![],
        }
    }
}

#[test]
fn test_with_defaults() {
    let mut app = App::new();

    // Uses defaults, only override what matters for test
    let entity = EntityBuilder::default()
        .with_velocity(5.0, 0.0)
        .spawn(&mut app.world_mut());

    let pos = app.world().entity(entity).get::<Position>().unwrap();
    assert_eq!(pos.x, 0.0);
    assert_eq!(pos.y, 0.0);
}
```

### Scenario Builders

Build complete test scenarios:

```rust
struct ScenarioBuilder {
    players: Vec<EntityBuilder>,
    enemies: Vec<EntityBuilder>,
    resources: Vec<Box<dyn Resource>>,
}

impl ScenarioBuilder {
    fn new() -> Self {
        Self {
            players: vec![],
            enemies: vec![],
            resources: vec![],
        }
    }

    fn with_player(mut self, builder: EntityBuilder) -> Self {
        self.players.push(builder);
        self
    }

    fn with_enemy(mut self, builder: EntityBuilder) -> Self {
        self.enemies.push(builder);
        self
    }

    fn with_resource<R: Resource>(mut self, resource: R) -> Self {
        self.resources.push(Box::new(resource));
        self
    }

    fn build(self, app: &mut App) -> Scenario {
        let player_entities: Vec<_> = self.players
            .into_iter()
            .map(|builder| {
                let entity = builder.spawn(&mut app.world_mut());
                app.world_mut().entity_mut(entity).insert(Player);
                entity
            })
            .collect();

        let enemy_entities: Vec<_> = self.enemies
            .into_iter()
            .map(|builder| {
                let entity = builder.spawn(&mut app.world_mut());
                app.world_mut().entity_mut(entity).insert(Enemy);
                entity
            })
            .collect();

        Scenario {
            players: player_entities,
            enemies: enemy_entities,
        }
    }
}

struct Scenario {
    players: Vec<Entity>,
    enemies: Vec<Entity>,
}

#[test]
fn test_combat_scenario() {
    let mut app = App::new();

    let scenario = ScenarioBuilder::new()
        .with_player(EntityBuilder::new(100).with_position(0.0, 0.0))
        .with_enemy(EntityBuilder::new(50).with_position(10.0, 0.0))
        .with_enemy(EntityBuilder::new(50).with_position(20.0, 0.0))
        .build(&mut app);

    assert_eq!(scenario.players.len(), 1);
    assert_eq!(scenario.enemies.len(), 2);
}
```

## World Setup Helpers

Create reusable world initialization:

```rust
fn setup_test_world() -> World {
    let mut world = World::new();

    // Register standard resources
    world.insert_resource(GameTime::default());
    world.insert_resource(Config::test_config());
    world.insert_resource(AssetLoader::mock());

    world
}

fn setup_combat_world() -> World {
    let mut world = setup_test_world();

    // Add combat-specific resources
    world.insert_resource(DamageMultiplier { value: 1.0 });
    world.insert_resource(CombatLog::new());

    world
}

#[test]
fn test_with_combat_world() {
    let world = setup_combat_world();

    // World already configured for combat testing
    assert!(world.contains_resource::<DamageMultiplier>());
    assert!(world.contains_resource::<CombatLog>());
}
```

## Separating Logic from World Access

Extract pure functions for easier testing:

```rust
// Pure function - easy to test
fn calculate_damage(base: u32, multiplier: f32, armor: u32) -> u32 {
    let raw_damage = (base as f32 * multiplier) as u32;
    raw_damage.saturating_sub(armor)
}

// System - thin wrapper around pure logic
fn apply_damage_system(
    mut query: Query<(&mut Health, &Damage, Option<&Armor>)>,
    multiplier: Res<DamageMultiplier>,
) {
    for (mut health, damage, armor) in query.iter_mut() {
        let armor_value = armor.map(|a| a.value).unwrap_or(0);
        let actual_damage = calculate_damage(
            damage.value,
            multiplier.value,
            armor_value,
        );
        health.value = health.value.saturating_sub(actual_damage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_calculation() {
        // Test pure function without ECS
        assert_eq!(calculate_damage(100, 1.5, 20), 130);
        assert_eq!(calculate_damage(100, 1.0, 150), 0);
        assert_eq!(calculate_damage(50, 2.0, 50), 50);
    }

    #[test]
    fn test_system_integration() {
        // Integration test with full ECS
        let mut app = App::new();
        app.insert_resource(DamageMultiplier { value: 1.5 });

        app.world_mut().spawn((
            Health { value: 200 },
            Damage { value: 100 },
            Armor { value: 20 },
        ));

        app.add_systems(Update, apply_damage_system);
        app.update();

        let health = app.world().query::<&Health>().single(app.world());
        assert_eq!(health.value, 70);  // 200 - 130
    }
}
```

## Test Utilities Module

Organize test helpers:

```rust
#[cfg(test)]
mod test_utils {
    use super::*;

    pub fn spawn_player(world: &mut World) -> Entity {
        world.spawn((
            Player,
            Health { value: 100 },
            Position::default(),
        )).id()
    }

    pub fn spawn_enemy(world: &mut World, health: u32) -> Entity {
        world.spawn((
            Enemy,
            Health { value: health },
            Position::default(),
        )).id()
    }

    pub fn assert_health(world: &World, entity: Entity, expected: u32) {
        let health = world.entity(entity).get::<Health>().unwrap();
        assert_eq!(health.value, expected);
    }

    pub fn assert_has_component<T: Component>(world: &World, entity: Entity) {
        assert!(world.entity(entity).contains::<T>());
    }

    pub fn get_entities_with<T: Component>(world: &World) -> Vec<Entity> {
        world.query::<Entity>()
            .iter(world)
            .filter(|&e| world.entity(e).contains::<T>())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn test_with_utils() {
        let mut app = App::new();

        let player = spawn_player(&mut app.world_mut());
        let enemy = spawn_enemy(&mut app.world_mut(), 50);

        assert_health(&app.world(), player, 100);
        assert_health(&app.world(), enemy, 50);

        assert_has_component::<Player>(&app.world(), player);
        assert_has_component::<Enemy>(&app.world(), enemy);
    }
}
```

## Trait-Based System Interfaces

Define interfaces for systems:

```rust
trait DamageSystem {
    fn apply_damage(&mut self, entity: Entity, amount: u32);
}

trait SpawnSystem {
    fn spawn_enemy(&mut self, position: Position) -> Entity;
}

// Implementation for real world
impl DamageSystem for World {
    fn apply_damage(&mut self, entity: Entity, amount: u32) {
        if let Some(mut health) = self.entity_mut(entity).get_mut::<Health>() {
            health.value = health.value.saturating_sub(amount);
        }
    }
}

// Mock implementation for tests
struct MockWorld {
    damage_calls: Vec<(Entity, u32)>,
}

impl DamageSystem for MockWorld {
    fn apply_damage(&mut self, entity: Entity, amount: u32) {
        self.damage_calls.push((entity, amount));
    }
}

#[test]
fn test_with_mock_world() {
    let mut mock = MockWorld {
        damage_calls: vec![],
    };

    let entity = Entity::from_raw(1);

    mock.apply_damage(entity, 50);
    mock.apply_damage(entity, 25);

    assert_eq!(mock.damage_calls.len(), 2);
    assert_eq!(mock.damage_calls[0], (entity, 50));
    assert_eq!(mock.damage_calls[1], (entity, 25));
}
```

## Configuration-Based Testing

Use configurations to control system behavior:

```rust
#[derive(Resource, Clone)]
struct TestConfig {
    enable_damage: bool,
    enable_healing: bool,
    damage_multiplier: f32,
}

impl TestConfig {
    fn minimal() -> Self {
        Self {
            enable_damage: true,
            enable_healing: false,
            damage_multiplier: 1.0,
        }
    }

    fn full() -> Self {
        Self {
            enable_damage: true,
            enable_healing: true,
            damage_multiplier: 1.0,
        }
    }
}

fn combat_system(
    config: Res<TestConfig>,
    mut query: Query<(&mut Health, &Damage)>,
) {
    if !config.enable_damage {
        return;
    }

    for (mut health, damage) in query.iter_mut() {
        let actual_damage = (damage.value as f32 * config.damage_multiplier) as u32;
        health.value = health.value.saturating_sub(actual_damage);
    }
}

#[test]
fn test_damage_disabled() {
    let mut app = App::new();

    let mut config = TestConfig::minimal();
    config.enable_damage = false;
    app.insert_resource(config);

    app.world_mut().spawn((
        Health { value: 100 },
        Damage { value: 50 },
    ));

    app.add_systems(Update, combat_system);
    app.update();

    // Damage should not apply
    let health = app.world().query::<&Health>().single(app.world());
    assert_eq!(health.value, 100);
}
```

## Best Practices

1. **Dependency Injection**: Use trait bounds for external dependencies
2. **Pure Functions**: Extract business logic from ECS systems
3. **Builder Pattern**: Create fluent builders for complex entity setup
4. **Test Helpers**: Centralize common setup code
5. **Minimal Setup**: Only configure what the test needs
6. **Mock Resources**: Create test doubles for external dependencies
7. **Configuration**: Use config to control system behavior in tests
8. **Separation of Concerns**: Keep ECS coordination separate from business logic

## Common Patterns

Pattern for testable system:
```rust
// Pure logic
fn calculate(input: Input) -> Output { ... }

// System wrapper
fn system(res: Res<Resource>, query: Query<...>) {
    let output = calculate(input);
    // Apply output to ECS
}
```

Pattern for builder chain:
```rust
let entity = Builder::new()
    .with_component_a()
    .with_component_b()
    .spawn(world);
```

Pattern for test helper:
```rust
fn assert_invariant(world: &World) {
    // Verify world state
}
```
