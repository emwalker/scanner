# Elm Architecture Fundamentals

The Elm Architecture (TEA) is a structured pattern for building interactive applications. It consists of three core components that work together to manage state, handle user interactions, and render the interface.

## Core Components

### Model

The Model represents the complete state of the application at any given moment. It contains all the data needed to render the current view and execute updates.

**Characteristics:**
- Immutable representation of application state
- Contains only data, no behavior
- Completely describes what the user sees
- Pure data structures (structs, enums)

**Rust Example:**
```rust
#[derive(Clone)]
pub struct Model {
    pub scan_results: Vec<Station>,
    pub selected_station: Option<StationId>,
    pub view_mode: ViewMode,
    pub error: Option<String>,
}

#[derive(Clone)]
pub enum ViewMode {
    List,
    Details,
    Settings,
}
```

### View

The View is a pure function that transforms the current Model into a visual representation. For TUI applications, this means rendering terminal output using a framework like ratatui.

**Characteristics:**
- Pure function: same Model always produces same output
- No side effects (no I/O, no state mutations)
- Deterministic and testable
- Responds to user interactions by producing Messages

**Rust Example:**
```rust
pub fn view(model: &Model, area: Rect) -> Widget {
    match model.view_mode {
        ViewMode::List => render_list(model, area),
        ViewMode::Details => render_details(model, area),
        ViewMode::Settings => render_settings(model, area),
    }
}

fn render_list(model: &Model, area: Rect) -> Widget {
    // Pure rendering logic only - no state mutation
    // Returns a ratatui Widget
    List::new(
        model.scan_results.iter().map(|s| {
            ListItem::new(format!("{}: {}", s.frequency, s.signal_strength))
        })
    )
    .highlight_symbol("> ")
}
```

### Update

The Update function interprets messages (user interactions and events) and produces a new Model. It is the "logic" layer that determines how the application responds to events.

**Characteristics:**
- Pure function: same input always produces same output
- Takes current Model and a Message, returns new Model
- Single entry point for all state changes
- No side effects (I/O, async operations handled separately)

**Rust Example:**
```rust
pub enum Message {
    SelectStation(StationId),
    SwitchViewMode(ViewMode),
    ScanCompleted(Vec<Station>),
    ErrorOccurred(String),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::SelectStation(id) => {
            model.selected_station = Some(id);
            model
        }
        Message::SwitchViewMode(mode) => {
            model.view_mode = mode;
            model
        }
        Message::ScanCompleted(results) => {
            model.scan_results = results;
            model.error = None;
            model
        }
        Message::ErrorOccurred(error) => {
            model.error = Some(error);
            model
        }
    }
}
```

## The Update Cycle

1. User interacts with terminal (presses key, etc.)
2. Input handler produces a Message (e.g., `Message::SelectStation(id)`)
3. Update function receives Message and current Model
4. Update returns new Model
5. View function re-renders with new Model
6. Cycle repeats

```
User Input → Message → Update → New Model → View → Rendered UI
                                                        ↓
                                    ← ← ← ← ← ← ← ← ←
```

## Messages

Messages represent events that trigger state changes. They should describe what happened, not how to handle it.

**Good message design:**
```rust
enum Message {
    UserSelectedStation(StationId),
    ScanStarted,
    ScanCompleted(Vec<Station>),
    ErrorOccurred(String),
    ViewModeChanged(ViewMode),
}
```

**Poor message design:**
```rust
enum Message {
    RefreshUI,
    UpdateState,
    DoSomething,
}
```

Messages should describe events clearly and provide all data needed for the update function to make decisions.

## Key Principles

### Unidirectional Data Flow

Data flows in one consistent direction: Model → View → User Action → Message → Update → new Model. This makes state changes predictable and traceable.

### Pure Functions

Both Update and View should be pure functions with no side effects. This makes them testable and predictable. Pure functions are easier to reason about and less prone to subtle bugs.

### Single Source of Truth

The Model is the single source of truth for the application state. All UI rendering is derived from it. This prevents inconsistencies where different parts of the UI show conflicting information.

### Immutability

State changes are expressed as transformations that produce new Models, not mutations of existing ones. In Rust, this can mean taking ownership and returning a new instance, or using interior mutability when performance requires it.

## Handling Side Effects

The Elm Architecture separates pure logic from side effects. Side effects (HTTP requests, file I/O, timers, etc.) should not be mixed into the Update function.

**Pattern for handling side effects:**

```rust
pub enum Command {
    StartScan,
    FetchStationDetails(StationId),
    SaveSettings,
    None,
}

pub fn update(mut model: Model, msg: Message) -> (Model, Command) {
    match msg {
        Message::ScanStarted => {
            model.scanning = true;
            (model, Command::StartScan)
        }
        Message::SelectStation(id) => {
            model.selected_station = Some(id);
            (model, Command::FetchStationDetails(id))
        }
        Message::ScanCompleted(results) => {
            model.scan_results = results;
            model.scanning = false;
            (model, Command::None)
        }
    }
}
```

The update function returns both the new model AND a command describing what side effect should occur. A separate command handler executes these effects and routes the results back as messages.
