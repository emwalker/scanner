# Entity Component System (ECS) Patterns in Rust

Comprehensive guide to ECS architecture patterns for game engines and data-oriented systems in Rust.

## Archetype/Table Storage Pattern

Store entities with identical component sets in contiguous memory tables.

**Description**: Group entities by their component signature (archetype). Each archetype has dense arrays for each component type. Iterating over entities with specific components becomes a linear scan through cache-friendly memory. When components are added/removed, entities move between archetypes.

```rust
// Conceptual example (simplified from Bevy)
struct Archetype {
    entity_ids: Vec<Entity>,
    component_a: Vec<ComponentA>,
    component_b: Vec<ComponentB>,
}

struct World {
    archetypes: HashMap<ComponentSet, Archetype>,
}

// Query iterates over matching archetypes
fn query_system(world: &World) {
    for archetype in world.archetypes.values() {
        if archetype.has::<ComponentA>() && archetype.has::<ComponentB>() {
            for i in 0..archetype.len() {
                let a = &archetype.component_a[i];
                let b = &archetype.component_b[i];
                // Process components
            }
        }
    }
}
```

**When to use**:
- Large numbers of entities with homogeneous component sets
- Systems that iterate over many entities with the same components
- Performance-critical game loops requiring cache-friendly iteration
- Queries frequently access multiple components together
- Component composition changes infrequently relative to iteration

**When NOT to use**:
- Entities frequently add/remove components (archetype moves are expensive)
- Heterogeneous entity populations with many unique component combinations
- Small entity counts where archetype overhead dominates
- Need random access to specific entities more than batch iteration
- Memory fragmentation from many small archetypes is problematic

## Sparse Set Storage Pattern

Store component data in dense arrays with sparse index lookup.

**Description**: Each component type has a sparse array mapping entity IDs to dense array indices, and a dense array holding actual component data. Provides O(1) component addition/removal and O(1) lookup, with excellent iteration performance. No archetype moves needed.

```rust
struct SparseSet<T> {
    sparse: Vec<Option<usize>>,  // Entity ID -> dense index
    dense: Vec<T>,                // Packed component data
    entities: Vec<Entity>,        // Dense index -> Entity ID
}

impl<T> SparseSet<T> {
    fn insert(&mut self, entity: Entity, component: T) {
        let dense_index = self.dense.len();
        self.dense.push(component);
        self.entities.push(entity);
        self.sparse[entity.id()] = Some(dense_index);
    }

    fn remove(&mut self, entity: Entity) -> Option<T> {
        let dense_index = self.sparse[entity.id()]?;
        self.sparse[entity.id()] = None;

        // Swap-remove to maintain density
        let last = self.dense.len() - 1;
        self.dense.swap(dense_index, last);
        self.entities.swap(dense_index, last);
        self.sparse[self.entities[dense_index].id()] = Some(dense_index);

        Some(self.dense.pop().unwrap())
    }
}
```

**When to use**:
- Frequent component addition/removal during gameplay
- Need O(1) random access to specific entity components
- Heterogeneous entity populations with diverse component sets
- Components are often added/removed individually
- Iteration performance matters but archetype moves are too expensive

**When NOT to use**:
- Memory overhead of sparse array is prohibitive (use archetype pattern)
- Entity IDs are not dense integers (sparse array becomes wasteful)
- Queries almost always access multiple components together (archetypes more cache-friendly)
- Component sets are stable (archetype overhead is one-time cost)

## Query Pattern with Filters

Declaratively specify component requirements and filters for system iteration.

**Description**: Systems declare component dependencies using query syntax. Filters like `With`, `Without`, `Changed`, `Added`, `Removed` provide fine-grained control over which entities are processed.

```rust
// Bevy-style query syntax
fn movement_system(
    mut query: Query<(&mut Position, &Velocity), With<Player>>,
) {
    for (mut pos, vel) in query.iter_mut() {
        pos.x += vel.x;
        pos.y += vel.y;
    }
}

fn spawn_effects_system(
    mut commands: Commands,
    query: Query<Entity, Added<Damaged>>,
) {
    for entity in query.iter() {
        commands.spawn(DamageEffect { target: entity });
    }
}

fn despawn_dead_system(
    mut commands: Commands,
    query: Query<Entity, (With<Health>, Without<Alive>)>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}
```

**When to use**:
- Declarative system dependencies improve readability
- Need fine-grained filtering beyond component presence
- Change detection reduces unnecessary work
- Systems should only run on relevant entity subsets
- Framework handles query optimization and parallelization

**When NOT to use**:
- Simple iteration over all entities with component
- Custom filtering logic too complex for filter combinators
- Need manual control over iteration order
- Query overhead dominates for tiny entity sets

## Resource (Singleton) Pattern

Global state accessible to all systems without entity/component overhead.

**Description**: Resources are singleton values stored separately from entities. Systems can request read or write access to resources. Common for configuration, global state, or shared services.

```rust
#[derive(Resource)]
struct GameTime {
    elapsed: f32,
    delta: f32,
}

#[derive(Resource)]
struct Score(u32);

fn update_time(mut time: ResMut<GameTime>) {
    time.elapsed += time.delta;
}

fn check_win_condition(
    score: Res<Score>,
    mut game_state: ResMut<GameState>,
) {
    if score.0 >= 1000 {
        *game_state = GameState::Victory;
    }
}
```

**When to use**:
- Global configuration or settings
- Singleton services (audio manager, input handler, asset loader)
- Data that doesn't belong to any specific entity
- Shared state needed by many systems
- System communication via shared state

**When NOT to use**:
- Data logically belongs to an entity (use components instead)
- Need multiple instances (use entities with marker components)
- Resource becomes a dumping ground for unrelated state
- Excessive resources make system dependencies unclear

## Command Buffer Pattern

Queue entity/component mutations for deferred execution.

**Description**: Systems enqueue commands that modify world state (spawn, despawn, add/remove components). Commands execute after all systems complete, avoiding mid-iteration modifications and supporting parallelism.

```rust
fn spawn_enemies(
    mut commands: Commands,
    time: Res<GameTime>,
    spawner: Res<EnemySpawner>,
) {
    if time.elapsed % spawner.interval < time.delta {
        commands.spawn((
            Enemy,
            Position { x: 100.0, y: 100.0 },
            Health(50),
        ));
    }
}

fn damage_system(
    mut commands: Commands,
    query: Query<(Entity, &Health)>,
) {
    for (entity, health) in query.iter() {
        if health.0 == 0 {
            commands.entity(entity).despawn();
        }
    }
}
```

**When to use**:
- Systems run in parallel and can't mutate world directly
- Avoid iterator invalidation from mid-iteration modifications
- Batch modifications for performance
- Deferred operations simplify system logic
- Framework handles command execution and ordering

**When NOT to use**:
- Need immediate effect of mutation in same system
- Single-threaded execution where direct mutation is safe
- Command overhead is significant for tiny workloads
- Debugging deferred execution is problematic

## Entity Hierarchy Pattern

Model parent-child relationships with explicit hierarchy components.

**Description**: Entities have `Parent` and `Children` components creating tree structures. Transformations, visibility, and other properties propagate through hierarchies. Useful for scene graphs, UI layouts, skeletal animations.

```rust
#[derive(Component)]
struct Parent(Entity);

#[derive(Component)]
struct Children(Vec<Entity>);

fn setup_hierarchy(mut commands: Commands) {
    let parent = commands.spawn(Transform::default()).id();

    let child = commands.spawn((
        Transform::default(),
        Parent(parent),
    )).id();

    commands.entity(parent).insert(Children(vec![child]));
}

fn propagate_transforms(
    parent_query: Query<(&Transform, &Children)>,
    mut child_query: Query<(&mut Transform, &Parent)>,
) {
    for (parent_transform, children) in parent_query.iter() {
        for &child_entity in children.0.iter() {
            if let Ok((mut child_transform, _)) = child_query.get_mut(child_entity) {
                // Propagate parent transform to child
                *child_transform = parent_transform.combine(&*child_transform);
            }
        }
    }
}
```

**When to use**:
- Scene graphs with nested transforms
- UI layouts with parent-child widget relationships
- Skeletal animation rigs with bone hierarchies
- Logical grouping where operations affect descendants
- Cascade deletions (despawning parent despawns children)

**When NOT to use**:
- Flat entity structure suffices
- Hierarchy changes frequently (maintenance overhead)
- Deep hierarchies hurt performance (propagation cost)
- Relationships are many-to-many not tree-structured

## System Scheduling Pattern

Explicit system ordering and parallelization with dependency tracking.

**Description**: Systems declare execution order constraints (before/after other systems) and resource access patterns. Scheduler automatically parallelizes systems with non-conflicting resource access.

```rust
fn schedule_systems(app: &mut App) {
    app.add_systems(Update, (
        input_system,
        movement_system.after(input_system),
        collision_system.after(movement_system),
        (
            render_system,
            audio_system,
        ).after(collision_system), // Run in parallel
    ));
}

// Automatic parallelization based on queries
fn physics_system(mut query: Query<(&mut Position, &Velocity)>) {
    // Can run in parallel with audio_system (no conflicting access)
}

fn audio_system(mut query: Query<(&Transform, &AudioSource)>) {
    // Can run in parallel with physics_system
}
```

**When to use**:
- Complex system graphs with dependencies
- Maximize parallelism on multi-core systems
- Declarative ordering is clearer than manual management
- Systems have well-defined resource access patterns
- Framework handles synchronization and scheduling

**When NOT to use**:
- Simple linear system execution suffices
- Single-threaded execution (no parallelism benefits)
- Dynamic system ordering based on runtime state
- Debugging parallel execution is problematic
- Scheduling overhead dominates for trivial systems

## SystemParam Pattern

Custom parameter types for reusable system injection patterns.

**Description**: Implement `SystemParam` trait to create custom types that can be injected into system parameters. Encapsulates common query/resource patterns for reusability.

```rust
#[derive(SystemParam)]
struct PlayerQuery<'w, 's> {
    position: Query<'w, 's, &'static mut Position, With<Player>>,
    health: Query<'w, 's, &'static Health, With<Player>>,
}

fn player_system(mut player: PlayerQuery) {
    for mut pos in player.position.iter_mut() {
        // Access player position
    }
}

#[derive(SystemParam)]
struct GameServices<'w> {
    time: Res<'w, GameTime>,
    score: Res<'w, Score>,
    settings: Res<'w, GameSettings>,
}

fn gameplay_system(services: GameServices) {
    // All game services in one parameter
}
```

**When to use**:
- Common query/resource patterns used across many systems
- Encapsulate complex access patterns
- Reduce boilerplate in system signatures
- Create domain-specific abstractions
- Type-safe system parameter composition

**When NOT to use**:
- Pattern only used in single system
- Simple query/resource access doesn't need abstraction
- Complexity of SystemParam trait is excessive
- Hides important system dependencies

## Event Pattern (EventWriter/EventReader)

Typed message passing between systems via event queues.

**Description**: Systems write events to typed queues using `EventWriter`. Other systems read events using `EventReader`. Events persist for two frames allowing all systems to process them, then automatically clear.

```rust
#[derive(Event)]
struct CollisionEvent {
    entity_a: Entity,
    entity_b: Entity,
}

fn collision_detection(
    mut events: EventWriter<CollisionEvent>,
    query: Query<(Entity, &Position, &Collider)>,
) {
    for (entity_a, pos_a, collider_a) in query.iter() {
        for (entity_b, pos_b, collider_b) in query.iter() {
            if collides(pos_a, collider_a, pos_b, collider_b) {
                events.send(CollisionEvent { entity_a, entity_b });
            }
        }
    }
}

fn damage_system(
    mut events: EventReader<CollisionEvent>,
    mut health_query: Query<&mut Health>,
) {
    for event in events.read() {
        if let Ok(mut health) = health_query.get_mut(event.entity_a) {
            health.0 -= 10;
        }
    }
}
```

**When to use**:
- Decouple event producers from consumers
- One-to-many event broadcasting
- Events represent discrete occurrences not continuous state
- Systems process events independently
- Events need automatic cleanup after processing

**When NOT to use**:
- Continuous state better modeled as components
- Direct system communication is clearer
- Events only have single consumer (use channels or commands)
- Need guaranteed delivery or persistence (events auto-clear)
- Event types proliferate excessively

## Observer/Trigger Pattern

Entity-targeted events with lifecycle hooks.

**Description**: Attach observer callbacks to entities that trigger on component lifecycle events (add/remove) or custom events. Events can target specific entities and propagate through hierarchies.

```rust
fn setup_observers(mut commands: Commands) {
    commands.spawn(Health(100))
        .observe(|trigger: Trigger<DamageEvent>| {
            println!("Entity damaged: {:?}", trigger.event());
        });
}

#[derive(Event)]
struct DamageEvent {
    amount: i32,
}

fn apply_damage(
    mut commands: Commands,
    query: Query<Entity, With<Health>>,
) {
    for entity in query.iter() {
        commands.trigger_targets(
            DamageEvent { amount: 10 },
            entity,
        );
    }
}
```

**When to use**:
- Entity-specific event handling
- Component lifecycle hooks (on add, on remove)
- Event propagation through entity hierarchies
- Localized reactive behavior per-entity
- Alternative to polling for state changes

**When NOT to use**:
- Global events better suited for EventWriter/EventReader
- Performance-critical paths (observer dispatch overhead)
- Observers create coupling between entity and logic
- Simple component-based state machine suffices

## Plugin Pattern

Modular application composition with reusable feature bundles.

**Description**: Plugins group related systems, resources, and components into reusable modules. Apps compose functionality by adding plugins, promoting separation of concerns and code reuse.

```rust
struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update, (
                apply_velocity,
                apply_gravity,
                collision_detection,
            ))
            .insert_resource(PhysicsConfig::default());
    }
}

struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update, audio_system)
            .insert_resource(AudioManager::new());
    }
}

fn main() {
    App::new()
        .add_plugins((
            PhysicsPlugin,
            AudioPlugin,
        ))
        .run();
}
```

**When to use**:
- Modular application architecture
- Reusable feature bundles across projects
- Organize related systems and resources
- Plugin-based extensibility
- Separation of concerns (physics, rendering, audio)

**When NOT to use**:
- Single monolithic application with no reuse
- Plugin boundaries are unclear or constantly changing
- Overhead of plugin trait is unnecessary
- Dependencies between plugins become complex

## State Machine Pattern

Mode-based system execution with state transitions.

**Description**: Define application states (Menu, Playing, Paused) and configure systems to run only in specific states. State transitions trigger enter/exit systems for setup and cleanup.

```rust
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum GameState {
    #[default]
    Menu,
    Playing,
    Paused,
}

fn setup(app: &mut App) {
    app
        .init_state::<GameState>()
        .add_systems(OnEnter(GameState::Playing), setup_game)
        .add_systems(Update, gameplay_systems.run_if(in_state(GameState::Playing)))
        .add_systems(OnExit(GameState::Playing), cleanup_game);
}

fn pause_system(
    input: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if input.just_pressed(KeyCode::Escape) {
        match state.get() {
            GameState::Playing => next_state.set(GameState::Paused),
            GameState::Paused => next_state.set(GameState::Playing),
            _ => {}
        }
    }
}
```

**When to use**:
- Application has distinct modes (menu, gameplay, editor)
- Systems only relevant in certain states
- State transitions need setup/cleanup logic
- Clear state machine semantics improve code organization
- Conditional system execution based on mode

**When NOT to use**:
- State space is continuous not discrete
- All systems run regardless of application mode
- States proliferate excessively creating complexity
- State transitions are complex with many edge cases

## Change Detection Pattern

Track component modifications to avoid redundant processing.

**Description**: Queries can filter for `Changed<T>` or `Added<T>` components, allowing systems to process only entities where components were modified since last run. Reduces wasted computation.

```rust
fn render_system(
    query: Query<(&Transform, &Sprite), Changed<Transform>>,
) {
    for (transform, sprite) in query.iter() {
        // Only re-render entities whose transform changed
        update_render_data(transform, sprite);
    }
}

fn propagate_dirty_flags(
    parents: Query<Entity, Changed<Transform>>,
    children: Query<&Children>,
    mut dirty: Query<&mut Dirty>,
) {
    for parent in parents.iter() {
        if let Ok(children) = children.get(parent) {
            for &child in children.0.iter() {
                if let Ok(mut dirty) = dirty.get_mut(child) {
                    dirty.0 = true;
                }
            }
        }
    }
}
```

**When to use**:
- Expensive operations should only run when data changes
- Propagate changes through hierarchies
- Invalidation patterns (dirty flags, cache invalidation)
- Most entities don't change every frame
- Processing all entities every frame is wasteful

**When NOT to use**:
- Systems process all entities regardless of changes
- Change detection overhead exceeds processing savings
- Components change every frame for all entities
- Debugging change detection false positives is difficult

## ParamSet Pattern

Resolve conflicting queries in same system.

**Description**: When a system needs both mutable and immutable access to overlapping components, use `ParamSet` to enforce exclusive access. Only one query in the set can be accessed at a time.

```rust
fn conflicting_system(
    mut queries: ParamSet<(
        Query<&mut Transform>,
        Query<(&Transform, &Velocity)>,
    )>,
) {
    // Process first query
    for mut transform in queries.p0().iter_mut() {
        // Mutable access to Transform
    }

    // Now access second query (first is done)
    for (transform, velocity) in queries.p1().iter() {
        // Immutable access to Transform
    }
}
```

**When to use**:
- System needs multiple conflicting access patterns
- Sequential processing of different entity subsets
- Avoiding system splits for minor conflicts
- Queries access overlapping components with different mutability

**When NOT to use**:
- Queries don't conflict (use separate parameters)
- System can be split into multiple systems
- ParamSet forces sequential execution (hurts parallelism)
- Makes data dependencies less clear

## Local State Pattern

Per-system persistent state without global resources.

**Description**: Systems can have `Local<T>` parameters that persist between invocations but are isolated to that system. Useful for counters, caches, or system-specific state.

```rust
fn spawn_enemies(
    mut commands: Commands,
    mut timer: Local<f32>,
    time: Res<GameTime>,
) {
    *timer += time.delta;

    if *timer >= 5.0 {
        commands.spawn(Enemy);
        *timer = 0.0;
    }
}

fn fps_counter(
    mut frame_times: Local<Vec<f32>>,
    time: Res<GameTime>,
) {
    frame_times.push(time.delta);
    if frame_times.len() > 60 {
        frame_times.remove(0);
    }

    let avg = frame_times.iter().sum::<f32>() / frame_times.len() as f32;
    println!("FPS: {}", 1.0 / avg);
}
```

**When to use**:
- System-specific state not shared with other systems
- Avoids polluting global resources
- Simple counters, timers, or caches
- State lifetime tied to system lifetime
- Encapsulate implementation details

**When NOT to use**:
- State needs to be shared between systems (use resources)
- State should be queryable or inspectable
- System becomes stateful when it should be pure
- Local state makes testing harder

## Pattern Interactions

Real-world ECS architectures combine multiple patterns:

### Game Engine (Bevy)
- **Archetype storage** for entity/component management
- **System scheduling** for parallelization and ordering
- **Query pattern** with filters for entity iteration
- **Resource pattern** for global services
- **Command buffers** for deferred world mutations
- **Event pattern** for inter-system communication
- **Plugin pattern** for modular features
- **State machine** for game modes
- **Hierarchy pattern** for scene graphs

### Physics Simulation (Rapier + Bevy)
- **Sparse set storage** for frequently changing physics components
- **Change detection** to minimize physics recalculations
- **Resource pattern** for physics world singleton
- **Event pattern** for collision notifications
- **System scheduling** to order physics steps correctly
- **Command buffers** to apply forces and impulses

### UI Framework
- **Hierarchy pattern** for widget parent-child relationships
- **Observer pattern** for input event handling
- **State machine** for widget states (hover, pressed, disabled)
- **Change detection** for incremental layout updates
- **Query pattern** with filters for widget subsets
- **Local state** for per-widget caches

### Multiplayer Networked Game
- **Event pattern** for network messages
- **Command buffers** for client prediction and server reconciliation
- **Change detection** for replication dirty tracking
- **Resource pattern** for network connection
- **System scheduling** for client/server/shared logic separation
- **State machine** for connection states (connecting, connected, disconnected)

## Performance Considerations

**Archetype vs Sparse Set**: Archetypes excel at iteration (2-10x faster) but pay cost on component add/remove. Sparse sets have O(1) modifications but slower iteration and higher memory overhead.

**Query overhead**: Complex queries with many filters can be slower than simple iteration. Profile before optimizing. Cache query results when entities change infrequently.

**Change detection granularity**: Systems that write components trigger change detection even if value unchanged. Use `bypass_change_detection()` for reads that don't need tracking.

**Command buffer batching**: Commands execute in batches after system sets. Large batches of spawn/despawn can cause frame spikes. Consider spreading work across frames.

**Hierarchy traversal**: Deep hierarchies hurt propagation systems. Flatten hierarchies when possible. Cache global transforms to avoid recomputation.

**Event memory**: Events accumulate for 2 frames then clear. Floods of events can cause memory spikes. Consider custom event cleanup or bounded event queues.

**Parallel system efficiency**: Systems with conflicting resource access can't run in parallel. Minimize mutable resource access. Consider splitting systems or using change detection to reduce conflicts.

## References

- [Bevy ECS Documentation](https://docs.rs/bevy_ecs/latest/bevy_ecs/)
- [Bevy Examples - ECS](https://bevyengine.org/examples/ecs-entity-component-system/)
- [Understanding Bevy's Archetypes](https://deterministic.space/bevy-ecs-archetype-internals.html)
- [Specs Book - System Scheduling](https://specs.amethyst.rs/docs/tutorials/06_system_data.html)
- [hecs Documentation](https://docs.rs/hecs/latest/hecs/)
- [Catherine West - RustConf 2018: Using Rust For Game Development](https://www.youtube.com/watch?v=aKLntZcp27M)
- [Sander Mertens - ECS FAQ](https://github.com/SanderMertens/ecs-faq)
- [flecs Documentation - Observers](https://www.flecs.dev/flecs/md_docs_2Observers.html)
- [Entity Component System Design Patterns](https://github.com/doyoubi/ecs-pattern)
- [Understanding Component-Based and Data-Driven Design Patterns](https://www.dataorienteddesign.com/dodbook/)
