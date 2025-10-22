---
name: ecs-design
description: This skill should be used when designing or implementing Entity-Component-System (ECS) architecture in Rust for the scanner application. Apply this skill when creating or modifying entities, adding new components, implementing systems, performing dual-write migrations, or when the user mentions "ECS", "components", "entities", "systems", or discusses stateful interactive features. The skill provides patterns for EntityWorld storage, Arc<Mutex<>> thread-safety, component composition, and system design.
---

# ECS Architecture for Scanner

This skill provides guidance for designing and implementing Entity-Component-System architecture in the scanner application. Use the bundled template generator script and reference materials for extended patterns.

## Implementation Overview

- **Storage**: HashMap-based EntityWorld (dozens of entities, not thousands)
- **Entities**: TunerEntity, ScanEntity, StationEntity, AudioEntity
- **Thread-safety**: Arc<Mutex<EntityWorld<T>>>
- **Systems**: System trait + Scheduler for sequential execution
- **Components**: Priority, Constraint, Device, Allocation, Status, Tuning, Playback, etc.

## Quick Start: Creating New ECS Elements

Use the bundled template generator to create new components, entities, or systems:

```bash
scripts/generate_ecs_template.sh component MyComponent
scripts/generate_ecs_template.sh entity MyEntity
scripts/generate_ecs_template.sh system MySystem
```

The generator creates properly structured templates with TODOs, test scaffolding, and common patterns already in place.

## Component Design Rules

### Core Principles

1. **Pure data only**: State transitions OK, business logic NO
2. **Single responsibility**: One aspect of entity state (allocation, status, info)
3. **No data duplication**: Reference other components, don't copy their data

### Critical Anti-Patterns

For extended anti-patterns and advanced patterns (God components, cross-component dependencies, event components, query builders, etc.), refer to `references/advanced_patterns.md`.

❌ **Business logic in components**
```rust
// Bad - allocation logic belongs in pool/systems
impl AllocationComponent {
    pub fn find_best_tuner(&self, pool: &Pool) -> Option<Tuner> { /* ... */ }
}
```

❌ **Singleton components for global state**
```rust
// Bad - use Resource or Arc<Config>, not components
struct GameConfig { /* ... */ }
```

❌ **Data duplication**
```rust
// Bad - Transform data duplicated
struct TankComponent {
    position: Vec3,  // Already in Transform!
    rotation: Quat,  // Already in Transform!
}

// Good - read from Transform component instead
```

❌ **Overly specific components** - prevents reuse
```rust
// Bad - too specific
struct TankTransformComponent { /* ... */ }

// Good - reusable across entity types
struct TransformComponent { /* ... */ }
```

❌ **Flagging components** - all state in one massive struct
```rust
// Anti-pattern (though sometimes pragmatic)
struct PlayerStateFlags {
    is_jumping: bool,
    is_running: bool,
    is_attacking: bool,
    // ...30 more flags
}
```

### Component Granularity

**Too small**: Dependent state scattered, complex queries
**Too large**: Reduced reusability, harder maintenance
**Right size**: Single aspect of entity state

Scanner example:
```rust
// ✅ Good - single responsibilities
pub struct StationEntity {
    id: StationId,
    pub info: StationInfoComponent,      // frequency, signal_strength, audio_quality
    pub discovery: StationDiscoveryComponent,  // when/where discovered
    pub history: StationHistoryComponent,      // play history
}

// ❌ Bad - everything in one component
pub struct StationComponent {
    // info fields
    // discovery fields
    // history fields
}
```

### Component Methods

Keep simple - only state queries and transitions:

```rust
// ✅ Good
impl AllocationComponent {
    pub fn new() -> Self { /* ... */ }
    pub fn allocate(&mut self, allocated_to: String) { /* ... */ }
    pub fn is_available(&self) -> bool { /* ... */ }
}

// ❌ Bad - complex logic
impl AllocationComponent {
    pub fn allocate_best_tuner(&mut self, pool: &Pool) -> Option<Tuner> { /* ... */ }
}
```

### Component Checklist

- [ ] Pure data, no business logic
- [ ] Single responsibility
- [ ] No data duplication from other components
- [ ] Derives Clone + Debug (or Debug only if contains non-Clone types)
- [ ] Has state transition tests
- [ ] Added to components/mod.rs exports

## Entity Design

### Must Implement Entity Trait

```rust
impl Entity for MyEntity {
    type Id = MyId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}
```

Id type must implement: Hash + Eq + Clone + Debug

### Entity Structure

```rust
pub struct MyEntity {
    id: MyId,                    // Private - use id() method
    pub component_a: ComponentA,  // Public components
    pub component_b: ComponentB,
}
```

### Convenience Methods

Simple queries OK, complex queries belong in systems (future):

```rust
impl TunerEntity {
    // ✅ Good - simple query
    pub fn is_available(&self) -> bool {
        self.device.connected && self.allocation.is_available()
    }

    // ❌ Bad - complex business logic
    pub fn allocate_best_for_task(&mut self, pool: &Pool, task: Task) -> Result<()> {
        // This belongs in a system or pool method
    }
}
```

### Entity Checklist

- [ ] Implements Entity trait
- [ ] Composes existing components
- [ ] Constructor initializes all components
- [ ] Convenience methods are simple queries only
- [ ] Has creation + query tests
- [ ] Added to entities/mod.rs exports

## World Usage

### Thread-Safe Access

```rust
// Field declaration
pub(crate) tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>

// Lock to access
let entities = self.tuner_entities.lock().unwrap();
let count = entities.iter().filter(|e| e.is_available()).count();

// Non-blocking access (shutdown-safe)
if let Ok(entities) = self.tuner_entities.try_lock() {
    // ...
}
```

### Queries

```rust
// Find entities matching criteria
let available: Vec<_> = world
    .iter()
    .filter(|e| e.is_available())
    .collect();

// Count entities
let count = world
    .iter()
    .filter(|e| e.allocation.is_allocated())
    .count();

// Find best match
let best = world
    .iter()
    .filter(|e| e.device.connected)
    .filter(|e| e.allocation.is_available())
    .max_by_key(|e| e.priorities.scanning);
```

## Systems

### System Trait

Systems implement behavior that operates on entities and components:

```rust
pub trait System: Send {
    fn name(&self) -> &'static str;
    fn run(&mut self, context: &mut SystemContext) -> Result<()>;
}
```

### SystemContext

Provides access to entity worlds during system execution:

```rust
let mut context = SystemContext::new()
    .with_tuner_entities(tuner_entities)
    .with_scan_entities(scan_entities)
    .with_audio_entities(audio_entities)
    .with_station_entities(station_entities);
```

### System Examples

**DeviceDiscoverySystem**: Monitors and logs tuner state
**TunerAllocationSystem**: Allocates tuners based on priority/constraints
**AudioManagementSystem**: Cleans up stopped or expired audio sessions
**ScanCoordinationSystem**: Coordinates scan lifecycle and progress

### System Pattern

```rust
pub struct MySystem {
    // System-specific state (e.g., pending requests)
}

impl System for MySystem {
    fn name(&self) -> &'static str {
        "MySystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let entities = match &context.tuner_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let entities = entities.lock().unwrap();
        for entity in entities.iter() {
            // Process entity
        }

        Ok(())
    }
}
```

### Scheduler

Execute systems in sequence:

```rust
let mut scheduler = Scheduler::new();
scheduler.add_system(Box::new(DeviceDiscoverySystem::new()));
scheduler.add_system(Box::new(TunerAllocationSystem::new()));
scheduler.add_system(Box::new(AudioManagementSystem::new()));

scheduler.run(&mut context)?;
```

### System Checklist

- [ ] Implements System trait with unique name
- [ ] Stateless where possible (request queues OK)
- [ ] Handles missing entity worlds gracefully
- [ ] Uses clone() on Arc<Mutex<>> to avoid borrow issues
- [ ] Has tests for empty context and various entity states
- [ ] Implements Default if it has a simple new()

## Migration Strategy (Proven Pattern)

When migrating existing code to use ECS:

1. **Add EntityWorld alongside existing structures** (dual-write)
2. **Synchronize both on writes** (create entities when adding devices, update both on allocation)
3. **Migrate reads to query entities** (change HashMap lookups to entity queries)
4. **Remove old structures** (delete HashMap, make EntityWorld sole source of truth)

This keeps tests passing at each step - zero-downtime migration.

Example from Phase 2 (Pool migration):
```rust
// Step 1: Add EntityWorld
pub(crate) tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>

// Step 2: Dual-write (synchronize both)
let entity = TunerEntity::new(...);
self.tuner_entities.lock().unwrap().insert(entity);
self.available_tuners.insert(tuner_id, tuner);  // Old HashMap

// Step 3: Migrate reads
// Old: self.available_tuners.values().filter(...)
// New: self.tuner_entities.lock().unwrap().iter().filter(...)

// Step 4: Remove old HashMap fields
```

## When NOT to Use ECS

- **Simple CRUD applications**: < 10 entities with stable relationships
- **Hierarchical data**: Better suited for tree structures
- **Team unfamiliar + small project**: Learning curve may not be worth it
- **OOP is clearer**: When object-oriented design maps better to domain

## File Organization

```
src/ecs/
├── mod.rs              # Public API and re-exports
├── entity.rs           # Entity trait definition
├── world.rs            # EntityWorld storage
├── system.rs           # System trait and SystemContext
├── schedule.rs         # Scheduler for system execution
├── components/
│   ├── mod.rs          # Component re-exports
│   ├── allocation.rs   # AllocationComponent
│   ├── constraint.rs   # ConstraintComponent
│   ├── device.rs       # DeviceComponent
│   ├── priority.rs     # PriorityComponent
│   ├── status.rs       # StatusComponent
│   ├── audio/          # Audio components
│   ├── scan/           # Scan components
│   └── station/        # Station components
├── entities/
│   ├── mod.rs          # Entity re-exports
│   ├── tuner.rs        # TunerEntity
│   ├── scan.rs         # ScanEntity
│   ├── station.rs      # StationEntity
│   └── audio.rs        # AudioEntity
└── systems/
    ├── mod.rs          # System re-exports
    ├── device/
    │   └── discovery.rs     # DeviceDiscoverySystem
    ├── tuner/
    │   └── allocation.rs    # TunerAllocationSystem
    ├── audio/
    │   └── management.rs    # AudioManagementSystem
    └── scan/
        └── coordination.rs  # ScanCoordinationSystem
```

## Common Patterns

### Marker Components (Zero-Sized Types)

```rust
// No memory overhead, useful for filtering
#[derive(Debug, Clone)]
pub struct Player;

#[derive(Debug, Clone)]
pub struct Enemy;

// Query entities by marker
let players: Vec<_> = world.iter()
    .filter(|e| has_player_marker(e))
    .collect();
```

### Component Composition

```rust
// Prefer composition over monolithic components
pub struct TunerEntity {
    id: TunerId,
    pub device: DeviceComponent,      // What hardware?
    pub allocation: AllocationComponent,  // Is it allocated?
    pub status: StatusComponent,      // What's it doing?
}
```

### State Transitions via Enums

```rust
// Use enums for mutually exclusive states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationState {
    Available,
    Allocated,
}

// Prevents impossible states (can't be both available AND allocated)
```

## Testing

### Test Component State Transitions

```rust
#[test]
fn test_allocation_lifecycle() {
    let mut component = AllocationComponent::new();
    assert!(component.is_available());

    component.allocate("scan_1".to_string());
    assert!(component.is_allocated());

    component.deallocate();
    assert!(component.is_available());
}
```

### Test Entity Creation

```rust
#[test]
fn test_entity_creation() {
    let entity = TunerEntity::new(device_id, 0, capabilities, backend);

    assert!(entity.is_available());
    assert!(entity.device.connected);
    assert_eq!(entity.status.activity, TunerActivity::Idle);
}
```

### Test Queries on World

```rust
#[test]
fn test_query_available_tuners() {
    let mut world = EntityWorld::new();

    let mut entity1 = create_test_entity("dev1", 0);
    entity1.allocation.allocate("scan_1".to_string());

    let entity2 = create_test_entity("dev1", 1);

    world.insert(entity1);
    world.insert(entity2);

    let available = world.iter()
        .filter(|e| e.is_available())
        .count();

    assert_eq!(available, 1);
}
```
