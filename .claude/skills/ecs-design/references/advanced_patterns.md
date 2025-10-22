# Advanced ECS Patterns and Anti-Patterns

This reference provides extended examples and patterns for Entity-Component-System design in the scanner application.

## Extended Anti-Patterns

### Anti-Pattern: God Components

**Problem:** Components that know too much or do too much.

```rust
// ❌ Bad - God component with everything
pub struct EntityState {
    // Device info
    device_id: String,
    device_type: DeviceType,
    connected: bool,

    // Allocation
    allocated_to: Option<String>,
    allocation_time: Option<Instant>,

    // Status
    activity: Activity,
    last_update: Instant,

    // Priority
    scanning_priority: u32,
    playback_priority: u32,

    // Constraints
    max_allocations: usize,
    allowed_bands: Vec<Band>,

    // ...and 20 more fields
}
```

**Solution:** Break into focused components.

```rust
// ✅ Good - Focused components
pub struct DeviceComponent {
    device_id: String,
    device_type: DeviceType,
    connected: bool,
}

pub struct AllocationComponent {
    allocated_to: Option<String>,
    allocation_time: Option<Instant>,
}

pub struct StatusComponent {
    activity: Activity,
    last_update: Instant,
}
```

### Anti-Pattern: Cross-Component Data Dependencies

**Problem:** Component A stores data from Component B, creating synchronization issues.

```rust
// ❌ Bad - Duplicated position data
pub struct Transform {
    position: Vec3,
    rotation: Quat,
}

pub struct Physics {
    velocity: Vec3,
    position: Vec3,  // ❌ Duplicated from Transform!
}

// Now we need to sync physics.position with transform.position
```

**Solution:** Reference the other component or use a single source of truth.

```rust
// ✅ Good - Single source of truth
pub struct Transform {
    position: Vec3,
    rotation: Quat,
}

pub struct Physics {
    velocity: Vec3,
    // Position is read from Transform component, not stored here
}

// In system:
fn update_physics(entity: &mut Entity) {
    let new_position = entity.transform.position + entity.physics.velocity * dt;
    entity.transform.position = new_position;
}
```

### Anti-Pattern: Components With External Side Effects

**Problem:** Components that perform I/O, logging, or other side effects in their methods.

```rust
// ❌ Bad - Component performs I/O
impl AllocationComponent {
    pub fn allocate(&mut self, id: String) {
        self.allocated_to = Some(id.clone());

        // ❌ Side effect in component method
        log::info!("Allocated to {}", id);

        // ❌ External call in component method
        metrics::record_allocation(&id);
    }
}
```

**Solution:** Keep components pure, move side effects to systems.

```rust
// ✅ Good - Pure component
impl AllocationComponent {
    pub fn allocate(&mut self, id: String) {
        self.allocated_to = Some(id);
    }

    pub fn allocated_to(&self) -> Option<&str> {
        self.allocated_to.as_deref()
    }
}

// Side effects in system
impl System for AllocationSystem {
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        for entity in entities.iter() {
            if should_allocate(entity) {
                entity.allocation.allocate(scan_id.clone());

                // Side effects here, not in component
                log::info!("Allocated {} to {}", entity.id(), scan_id);
                metrics::record_allocation(&scan_id);
            }
        }
        Ok(())
    }
}
```

### Anti-Pattern: Mutable Component Queries Without Clear Ownership

**Problem:** Multiple systems modifying the same component without coordination.

```rust
// ❌ Bad - Race condition between systems
// System 1
impl System for System1 {
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        for entity in entities.iter() {
            entity.status.set_activity(Activity::Scanning);
        }
        Ok(())
    }
}

// System 2 (runs in parallel)
impl System for System2 {
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        for entity in entities.iter() {
            entity.status.set_activity(Activity::Idle);
        }
        Ok(())
    }
}
// Who wins? Undefined!
```

**Solution:** Use sequential execution or explicit ownership rules.

```rust
// ✅ Good - Sequential execution via scheduler
let mut scheduler = Scheduler::new();
scheduler.add_system(Box::new(System1::new()));
scheduler.add_system(Box::new(System2::new()));
// System1 runs completely before System2

// OR: Clear ownership rules
// Rule: Only TunerAllocationSystem modifies allocation
// Rule: Only AudioManagementSystem modifies playback
```

## Advanced Patterns

### Pattern: Event Components (Marker + Data)

Use marker components with associated data for event-driven behavior.

```rust
// Marker component
#[derive(Debug, Clone)]
pub struct NewlyDiscovered;

// Event data component
#[derive(Debug, Clone)]
pub struct DiscoveryEvent {
    discovered_at: Instant,
    discovered_by: String,
}

// Entity with event
pub struct StationEntity {
    id: StationId,
    pub info: StationInfoComponent,
    pub discovery: Option<DiscoveryEvent>,
    pub newly_discovered: Option<NewlyDiscovered>,
}

// System processes and clears event
impl System for NotificationSystem {
    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        for entity in entities.iter() {
            if entity.newly_discovered.is_some() {
                notify_user(entity);
                entity.newly_discovered = None;  // Clear event
            }
        }
        Ok(())
    }
}
```

### Pattern: Hierarchical Components

For entities with parent-child relationships.

```rust
#[derive(Debug, Clone)]
pub struct Parent {
    parent_id: Option<EntityId>,
}

#[derive(Debug, Clone)]
pub struct Children {
    child_ids: Vec<EntityId>,
}

// Query pattern
fn find_children(parent: &Entity, world: &EntityWorld) -> Vec<&Entity> {
    parent.children.child_ids.iter()
        .filter_map(|id| world.get(id))
        .collect()
}
```

### Pattern: Component State Machines

Use enums for clear state transitions.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunerState {
    Idle,
    Scanning { scan_id: String },
    Playing { station_id: String },
    Error { reason: String },
}

#[derive(Debug, Clone)]
pub struct StatusComponent {
    state: TunerState,
    last_transition: Instant,
}

impl StatusComponent {
    pub fn transition_to_scanning(&mut self, scan_id: String) {
        // Only allow from Idle
        if matches!(self.state, TunerState::Idle) {
            self.state = TunerState::Scanning { scan_id };
            self.last_transition = Instant::now();
        }
    }

    pub fn transition_to_playing(&mut self, station_id: String) {
        // Allow from Idle or Scanning
        if matches!(self.state, TunerState::Idle | TunerState::Scanning { .. }) {
            self.state = TunerState::Playing { station_id };
            self.last_transition = Instant::now();
        }
    }

    pub fn transition_to_idle(&mut self) {
        // Allow from any state
        self.state = TunerState::Idle;
        self.last_transition = Instant::now();
    }
}
```

### Pattern: Resource Pooling with ECS

Combine entity-based resource tracking with pool management.

```rust
pub struct Pool {
    tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>,
}

impl Pool {
    pub fn allocate_tuner(&self, scan_id: String) -> Option<TunerId> {
        let mut entities = self.tuner_entities.lock().unwrap();

        // Find best available tuner
        let tuner = entities.iter_mut()
            .filter(|e| e.is_available())
            .max_by_key(|e| e.priorities.scanning)?;

        // Allocate it
        tuner.allocation.allocate(scan_id);

        Some(tuner.id().clone())
    }

    pub fn deallocate_tuner(&self, tuner_id: &TunerId) {
        let mut entities = self.tuner_entities.lock().unwrap();

        if let Some(tuner) = entities.get_mut(tuner_id) {
            tuner.allocation.deallocate();
            tuner.status.transition_to_idle();
        }
    }
}
```

### Pattern: Query Builders

For complex entity queries.

```rust
pub struct TunerQuery<'a> {
    entities: &'a EntityWorld<TunerEntity>,
    filters: Vec<Box<dyn Fn(&TunerEntity) -> bool>>,
}

impl<'a> TunerQuery<'a> {
    pub fn new(entities: &'a EntityWorld<TunerEntity>) -> Self {
        Self {
            entities,
            filters: vec![],
        }
    }

    pub fn available(mut self) -> Self {
        self.filters.push(Box::new(|e| e.is_available()));
        self
    }

    pub fn connected(mut self) -> Self {
        self.filters.push(Box::new(|e| e.device.connected));
        self
    }

    pub fn min_scanning_priority(mut self, priority: u32) -> Self {
        self.filters.push(Box::new(move |e| e.priorities.scanning >= priority));
        self
    }

    pub fn execute(&self) -> Vec<&TunerEntity> {
        self.entities.iter()
            .filter(|e| self.filters.iter().all(|f| f(e)))
            .collect()
    }
}

// Usage
let available_tuners = TunerQuery::new(&entities)
    .available()
    .connected()
    .min_scanning_priority(5)
    .execute();
```

## Testing Strategies

### Testing Component Invariants

```rust
#[test]
fn test_allocation_invariants() {
    let mut component = AllocationComponent::new();

    // Invariant: Can't deallocate when not allocated
    assert!(component.is_available());
    component.deallocate();  // Should be no-op
    assert!(component.is_available());

    // Invariant: Can't double-allocate
    component.allocate("scan1".to_string());
    component.allocate("scan2".to_string());  // Should be no-op
    assert_eq!(component.allocated_to(), Some("scan1"));
}
```

### Testing System Isolation

```rust
#[test]
fn test_system_handles_missing_entities() {
    let mut system = MySystem::new();
    let mut context = SystemContext::new();
    // No entities provided

    // System should handle gracefully
    let result = system.run(&mut context);
    assert!(result.is_ok());
}

#[test]
fn test_system_handles_empty_world() {
    let mut system = MySystem::new();
    let entities = Arc::new(Mutex::new(EntityWorld::new()));
    let mut context = SystemContext::new()
        .with_tuner_entities(entities);

    // Empty world
    let result = system.run(&mut context);
    assert!(result.is_ok());
}
```

### Testing Entity Lifecycle

```rust
#[test]
fn test_entity_full_lifecycle() {
    let mut world = EntityWorld::new();

    // Create
    let entity = TunerEntity::new(device_id, 0, caps, backend);
    let id = entity.id().clone();
    world.insert(entity);

    // Query
    assert!(world.contains(&id));
    let entity = world.get(&id).unwrap();
    assert!(entity.is_available());

    // Modify
    let entity = world.get_mut(&id).unwrap();
    entity.allocation.allocate("scan1".to_string());
    assert!(!entity.is_available());

    // Remove
    world.remove(&id);
    assert!(!world.contains(&id));
}
```

## Performance Considerations

### Batch Operations

```rust
// ❌ Slow - Lock per operation
for scan_id in scan_ids {
    let entities = self.tuner_entities.lock().unwrap();
    process_entity(&entities, scan_id);
}  // Lock released

// ✅ Fast - Single lock for all operations
let entities = self.tuner_entities.lock().unwrap();
for scan_id in scan_ids {
    process_entity(&entities, scan_id);
}
```

### Minimize Lock Contention

```rust
// ❌ Bad - Long critical section
let mut entities = self.tuner_entities.lock().unwrap();
for entity in entities.iter_mut() {
    let result = expensive_computation(entity);  // Holding lock during computation
    entity.update(result);
}

// ✅ Good - Short critical sections
let entity_data: Vec<_> = {
    let entities = self.tuner_entities.lock().unwrap();
    entities.iter().map(|e| e.clone()).collect()
};

let results: Vec<_> = entity_data.iter()
    .map(expensive_computation)
    .collect();

{
    let mut entities = self.tuner_entities.lock().unwrap();
    for (id, result) in results {
        if let Some(entity) = entities.get_mut(&id) {
            entity.update(result);
        }
    }
}
```

### Use try_lock for Shutdown Safety

```rust
// ✅ Good - Non-blocking during shutdown
impl Drop for Resource {
    fn drop(&mut self) {
        if let Ok(mut entities) = self.entities.try_lock() {
            // Cleanup
        }
        // If lock fails, skip cleanup - we're shutting down anyway
    }
}
```
