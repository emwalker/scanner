# Observer Pattern in Rust

Comprehensive guide to implementing notification and state change patterns in Rust.

## Closure-Based Observer Pattern

Store callbacks as boxed closures in a vector, the simplest approach for Rust.

**Description**: Maintain a collection of `Box<dyn Fn(T) + Send + Sync>` callbacks that are invoked when state changes. Callbacks receive data directly (e.g., the new state) rather than querying it themselves.

```rust
pub type StateChangeCallback = Arc<Mutex<Vec<Box<dyn Fn(PoolStatus) + Send + Sync>>>>;

pub fn add_callback(&self, callback: Box<dyn Fn(PoolStatus) + Send + Sync>) {
    self.callbacks.lock().unwrap().push(callback);
}

pub fn notify(&self, status: PoolStatus) {
    if let Ok(callbacks) = self.callbacks.lock() {
        for callback in callbacks.iter() {
            callback(status.clone());
        }
    }
}
```

**When to use**:
- Single or few subscribers that live for the application lifetime
- Simple notification requirements without complex subscription management
- Thread-safe notification is needed (Arc + Mutex)
- Callbacks don't need to unsubscribe individually

**When NOT to use**:
- Many dynamic subscribers that come and go (no unsubscribe by ID)
- Subscribers need to be removed individually
- Complex filtering or routing of notifications is required
- You need backpressure or flow control

## Weak Reference Observer Pattern

Use `Rc<Weak<dyn Observer>>` or `Arc<Weak<dyn Observer>>` to prevent circular dependencies.

**Description**: Observable holds weak references to observers, allowing observers to be dropped without explicit unsubscription. When notifying, upgrade weak references to strong references temporarily.

```rust
pub trait Observer {
    fn update(&self, data: &Data);
}

pub struct Subject {
    observers: Vec<Weak<dyn Observer>>,
}

impl Subject {
    pub fn attach(&mut self, observer: Weak<dyn Observer>) {
        self.observers.push(observer);
    }

    pub fn notify(&mut self, data: &Data) {
        self.observers.retain(|weak| {
            if let Some(observer) = weak.upgrade() {
                observer.update(data);
                true
            } else {
                false // Remove dead weak references
            }
        });
    }
}
```

**When to use**:
- Observers have shorter lifetimes than the observable
- Want automatic cleanup when observers are dropped
- Need to prevent circular reference memory leaks
- Single-threaded context (use `Arc<Weak<>>` for multi-threaded)

**When NOT to use**:
- All observers live as long as the observable (unnecessary complexity)
- Need guaranteed delivery (observers might be dropped)
- Performance-critical paths (weak reference upgrades have overhead)
- Multi-threaded without using Arc instead of Rc

## Message Passing with Channels

Use Rust's channel primitives for decoupled, composable event notification.

**Description**: Observable sends events to channels; observers receive from channels. Supports multiple channel types: mpsc (single consumer), broadcast (multiple consumers, all see all messages), watch (multiple consumers, only see latest value).

```rust
// Broadcast channel - all receivers get all messages
let (tx, _) = tokio::sync::broadcast::channel(100);

// Subscribe
let mut rx1 = tx.subscribe();
let mut rx2 = tx.subscribe();

// Publish
tx.send(event).unwrap();

// Receive in separate tasks
tokio::spawn(async move {
    while let Ok(event) = rx1.recv().await {
        // handle event
    }
});
```

**When to use**:
- Asynchronous, event-driven architectures
- Need backpressure and flow control
- Observers run in separate tasks/threads
- Want to compose streams with combinators
- "Share by communicating, don't communicate by sharing" philosophy

**When NOT to use**:
- Synchronous, immediate notification required
- Simple use case where callbacks suffice
- Don't want channel infrastructure overhead
- Need observer removal by identity

## Actor Model

Self-contained tasks that communicate via message passing.

**Description**: Each actor is a spawned async task with exclusive ownership of its state. Communication happens through typed message channels. Actors process messages sequentially, eliminating data races.

```rust
struct MyActor {
    receiver: mpsc::Receiver<Message>,
    state: MyState,
}

impl MyActor {
    async fn run(mut self) {
        while let Some(msg) = self.receiver.recv().await {
            self.handle_message(msg);
        }
    }

    fn handle_message(&mut self, msg: Message) {
        match msg {
            Message::DoSomething(data) => { /* ... */ }
            Message::GetState(respond_to) => {
                respond_to.send(self.state.clone()).unwrap();
            }
        }
    }
}

#[derive(Clone)]
struct MyActorHandle {
    sender: mpsc::Sender<Message>,
}

impl MyActorHandle {
    fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel(32);
        let actor = MyActor { receiver, state: MyState::new() };
        tokio::spawn(actor.run());
        Self { sender }
    }
}
```

**When to use**:
- Complex state management that needs isolation
- Natural task boundaries in your application
- Need to process events sequentially
- Want to eliminate shared mutable state
- Building distributed or concurrent systems

**When NOT to use**:
- Simple notification without state management
- Need synchronous callbacks
- Want multiple observers to react to same event simultaneously
- Memory overhead of spawning tasks is prohibitive

## Event Bus / Mediator Pattern

Central hub that routes events between decoupled components.

**Description**: Components publish events to a bus without knowing who will receive them. Other components subscribe to event types they're interested in. Combines Observer, Mediator, and sometimes Singleton patterns.

```rust
pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self { sender }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    pub fn publish(&self, event: Event) {
        let _ = self.sender.send(event);
    }
}

#[derive(Clone)]
pub enum Event {
    UserLoggedIn(UserId),
    DataUpdated(Data),
    // ...
}
```

**When to use**:
- Many-to-many communication between modules
- Components should be unaware of each other
- Event routing and filtering is needed
- Plugin or extension architectures
- Microservices-style communication patterns

**When NOT to use**:
- Simple one-to-many notification (observer pattern simpler)
- Direct component communication is clearer
- Events need guaranteed delivery and ordering
- Type-safe compile-time checking is priority (events are often enums or Any types)

## Reactive Streams (Futures/Stream Combinators)

Compose asynchronous event sequences with functional combinators.

**Description**: Events flow through a `Stream` that can be transformed, filtered, combined with other streams. Built on Rust's async ecosystem (futures, tokio::stream).

```rust
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

let (tx, _) = broadcast::channel(100);
let stream = BroadcastStream::new(tx.subscribe());

let processed = stream
    .filter(|event| event.is_important())
    .map(|event| event.transform())
    .take(10);

tokio::spawn(async move {
    tokio::pin!(processed);
    while let Some(result) = processed.next().await {
        // handle result
    }
});
```

**When to use**:
- Need to transform or filter event streams
- Combine multiple event sources
- Apply backpressure and rate limiting
- Build reactive dataflow pipelines
- Already using async/await ecosystem

**When NOT to use**:
- Synchronous, immediate callbacks needed
- Simple notification without transformation
- Learning curve of Stream combinators is prohibitive
- Performance-critical tight loops (stream overhead)

## Entity Component System (ECS) Observers

Specialized pattern for game engines and simulation systems.

**Description**: Observers react to entity lifecycle events (component add/remove) and custom events. Events can target specific entities and propagate through entity hierarchies. This is one of many patterns used in ECS architectures.

```rust
// Bevy ECS example
fn setup(mut commands: Commands) {
    commands.spawn(MyComponent)
        .observe(|trigger: Trigger<MyEvent>| {
            println!("Event received: {:?}", trigger.event());
        });
}

// Trigger events
commands.trigger(MyEvent { data: 42 });
commands.trigger_targets(MyEvent { data: 42 }, entity);
```

**When to use**:
- Building game engines or simulations
- Component-based architectures
- Need entity lifecycle hooks
- Event propagation through hierarchies (parent/child)
- Already using an ECS framework (bevy, specs, hecs)

**When NOT to use**:
- Not building a game or simulation
- Simple application state management
- Don't have entity/component architecture
- Overhead of ECS runtime is unjustified

For comprehensive coverage of ECS patterns including archetype storage, system scheduling, queries with filters, resources, command buffers, hierarchies, and more, see [docs/patterns/ecs.md](ecs.md)

## Trait-Based Observer with Dynamic Dispatch

Define Observer trait, store `Arc<Mutex<dyn Observer>>` in a collection.

**Description**: Traditional OOP-style observer pattern using trait objects. Observable maintains collection of boxed trait objects.

```rust
pub trait Observer: Send + Sync {
    fn update(&self, data: &Data);
}

pub struct Subject {
    observers: Vec<Arc<Mutex<dyn Observer>>>,
}

impl Subject {
    pub fn attach(&mut self, observer: Arc<Mutex<dyn Observer>>) {
        self.observers.push(observer);
    }

    pub fn notify(&self, data: &Data) {
        for observer in &self.observers {
            if let Ok(obs) = observer.try_lock() {
                obs.update(data);
            }
        }
    }
}
```

**When to use**:
- Coming from OOP languages, familiar pattern
- Observers need to maintain their own state
- Multiple observer types with different behaviors
- Compile-time polymorphism via traits

**When NOT to use**:
- Closure-based approach would be simpler
- Ownership and lifetime complexity becomes unwieldy
- Don't need different observer types (use closures)
- Performance-critical (dynamic dispatch overhead)

## Signal/Slot Pattern (Qt-Style)

Typed sender/receiver pairs with compile-time checking.

**Description**: Signals are emitted by components, slots are connected to signals. Rust implementations typically use channel-based approaches rather than Qt's macro-based system.

```rust
struct Button {
    clicked: tokio::sync::watch::Sender<()>,
}

impl Button {
    fn new() -> Self {
        let (tx, _) = watch::channel(());
        Self { clicked: tx }
    }

    fn click(&self) {
        self.clicked.send(()).unwrap();
    }

    fn on_clicked(&self) -> watch::Receiver<()> {
        self.clicked.subscribe()
    }
}
```

**When to use**:
- GUI programming
- Type-safe event connections
- Coming from Qt/C++ background
- Need compile-time verification of signal/slot types

**When NOT to use**:
- Not building a GUI
- Channel-based approach is more idiomatic in Rust
- Don't need Qt-style signal naming conventions
- Async message passing is more appropriate

## Future/Waker Notification Pattern

Low-level async notification using wakers for custom futures.

**Description**: Implement custom `Future` that stores a `Waker`. When the operation completes, call `waker.wake()` to notify the executor to poll the future again.

```rust
use std::task::{Context, Poll, Waker};
use std::future::Future;
use std::pin::Pin;

struct MyFuture {
    waker: Option<Waker>,
    data: Option<Data>,
}

impl Future for MyFuture {
    type Output = Data;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Data> {
        if let Some(data) = self.data.take() {
            Poll::Ready(data)
        } else {
            self.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

// Later, when data arrives:
if let Some(waker) = self.waker.take() {
    waker.wake();
}
```

**When to use**:
- Building custom async primitives
- Wrapping callback-based APIs as futures
- Need lowest-level async control
- Integrating with non-standard async sources

**When NOT to use**:
- Higher-level abstractions (channels, streams) suffice
- Not implementing custom Future types
- Complexity of waker management is unnecessary
- Existing async combinators solve the problem

## Hook/Lifecycle Pattern

Callbacks for component initialization, update, and cleanup.

**Description**: Framework calls hooks at specific lifecycle points. Components register callbacks to run when created, updated, or destroyed.

```rust
pub trait ComponentHooks {
    fn on_mount(&mut self) {}
    fn on_update(&mut self, old_props: &Props) {}
    fn on_unmount(&mut self) {}
}

// Or closure-based:
pub struct Component {
    on_mount: Vec<Box<dyn FnOnce() + Send>>,
    on_update: Vec<Box<dyn Fn(&Props) + Send>>,
}
```

**When to use**:
- Component-based frameworks
- Need setup/teardown for resources
- Separating component logic from lifecycle management
- React/Vue-style component model in Rust

**When NOT to use**:
- Simple RAII (Drop trait) handles cleanup
- No component lifecycle concept
- Hooks add unnecessary complexity
- Direct function calls are clearer

## Event Sourcing / CQRS Observer

Persist state changes as events, observers replay events for state reconstruction.

**Description**: Instead of storing current state, store sequence of events that led to that state. Observers consume event stream to build projections/views.

```rust
#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    UserCreated { id: UserId, name: String },
    UserUpdated { id: UserId, name: String },
}

pub struct EventStore {
    events: Vec<Event>,
    subscribers: Vec<mpsc::Sender<Event>>,
}

impl EventStore {
    pub fn append(&mut self, event: Event) {
        self.events.push(event.clone());
        for sub in &self.subscribers {
            let _ = sub.try_send(event.clone());
        }
    }

    pub fn subscribe(&mut self) -> mpsc::Receiver<Event> {
        let (tx, rx) = mpsc::channel(100);
        self.subscribers.push(tx);
        rx
    }
}
```

**When to use**:
- Need audit trail of all changes
- Multiple views of same data (CQRS)
- Time-travel debugging or event replay
- Domain-driven design with aggregates
- Building event-sourced systems

**When NOT to use**:
- Simple CRUD applications
- Current state is all that matters
- Storage overhead of events is prohibitive
- Complexity doesn't justify benefits
- Don't need event replay or audit trail

## Type-Erased Observer with Any/Downcast

Store heterogeneous observers with runtime type checking.

**Description**: Use `Box<dyn Any>` to store observers of different types, downcast when notifying specific observer types.

```rust
use std::any::Any;

pub struct Subject {
    observers: Vec<Box<dyn Any + Send + Sync>>,
}

impl Subject {
    pub fn attach<T: 'static + Send + Sync>(&mut self, observer: T) {
        self.observers.push(Box::new(observer));
    }

    pub fn notify<T: 'static>(&self, f: impl Fn(&T)) {
        for observer in &self.observers {
            if let Some(obs) = observer.downcast_ref::<T>() {
                f(obs);
            }
        }
    }
}
```

**When to use**:
- Need to store heterogeneous observer types
- Runtime flexibility trumps compile-time safety
- Building plugin systems with unknown types
- Interfacing with dynamic languages

**When NOT to use**:
- Type-safe alternatives exist (enums, trait objects)
- Runtime type checking is error-prone
- Downcast failures are hard to debug
- Performance of type checking matters

## Pattern Interactions

Real-world systems often combine multiple patterns:

### GUI Application (iced/egui)
- **Elm Architecture** (Model-View-Update) for application structure
- **Message passing** for user interactions → updates
- **Reactive streams** for async operations
- **Hooks** for component lifecycle

### Game Engine (Bevy)
- **ECS** for entity management
- **ECS Observers** for entity lifecycle events
- **Event bus** for global events (input, window resize)
- **Channels** for async asset loading

### Web Backend (actix-web)
- **Actor model** for request handlers
- **Event sourcing** for domain events
- **CQRS** for read/write separation
- **Channels** for background task communication

### System Monitor
- **Closure-based observers** for metric callbacks
- **Broadcast channels** for alert distribution
- **Weak references** for temporary subscribers
- **Actor model** for isolated monitoring tasks

## Performance Considerations

**Dynamic dispatch overhead**: Trait objects incur vtable lookup cost (~3x slower than static dispatch in tight loops, negligible in I/O-bound code)

**Channel buffering**: Bounded channels provide backpressure but can deadlock; unbounded channels prevent deadlocks but can exhaust memory

**Lock contention**: `Arc<Mutex<Vec<Callback>>>` serializes all observer notifications; consider per-observer channels for parallelism

**Clone costs**: Broadcasting `PoolStatus` clones it for each subscriber; use `Arc` for expensive types or consider reference-based notification

**Waker allocation**: Custom futures should reuse wakers when possible using `futures::task::AtomicWaker`

## References

- [Rust Async Book - Task Wakeups with Waker](https://rust-lang.github.io/async-book/02_execution/03_wakeups.html)
- [Tokio Tutorial - Channels](https://tokio.rs/tokio/tutorial/channels)
- [Actors with Tokio - Alice Ryhl](https://ryhl.io/blog/actors-with-tokio/)
- [The Hardest Pattern in Rust: Mediator](https://fadeevab.com/mediator-pattern-in-rust/)
- [Implementing an Event Bus using Rust](https://blog.digital-horror.com/blog/event-bus-in-tokio/)
- [Bevy ECS Observers](https://bevy.org/examples/ecs-entity-component-system/observers/)
- [Observer Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/observer/rust/example)
- [Stack Overflow - Observer Pattern in Rust](https://stackoverflow.com/questions/37572734/how-can-i-implement-the-observer-pattern-in-rust)
- [CQRS and Event Sourcing using Rust](https://doc.rust-cqrs.org/)
- [Pharos - Observable Library](https://lib.rs/crates/pharos)
- [rxRust - Reactive Extensions](https://github.com/rxRust/rxRust)
