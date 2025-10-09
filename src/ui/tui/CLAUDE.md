# The Elm Architecture for Terminal User Interfaces

This module implements The Elm Architecture (TEA) pattern for building maintainable, testable terminal user interfaces using Rust and ratatui.

## Architecture Overview

The Elm Architecture organizes code into three distinct concerns:

- **Model**: Application state and business logic
- **View**: Pure functions that render state to UI components
- **Update**: Pure functions that transform state based on events

This separation enables predictable state management, comprehensive testing, and confident refactoring.

## Core Principles

### State Centralization
- All application state lives in a single `Model` struct
- State mutations happen only through the `update()` method
- No shared mutable state between components
- State transitions are explicit and traceable

### Pure Functions
- View functions are pure: `fn render(model: &Model) -> Widget`
- Update functions are pure: `fn update(model: &mut Model, event: Event)`
- No side effects in rendering or state transitions
- Deterministic behavior enables reliable testing

### Unidirectional Data Flow
```
Events → Update → Model → View → UI
   ↑                              ↓
   └─────── User Input ────────────┘
```

Data flows in one direction, making the system predictable and debuggable.

## Module Structure

### Model (`model.rs`)
Contains all application state and business logic:
- Core data structures representing UI state
- State transition logic in `update()` methods
- Helper methods for querying state
- No dependencies on UI frameworks

### Layout (`layout.rs`)
Handles UI geometry and space allocation:
- Calculates widget constraints and positions
- Provides reusable layout utilities
- Separates layout logic from rendering logic
- Framework-agnostic layout calculations

### Renderers (`renderers/`)
Pure functions that convert state to UI widgets:
- One renderer per logical UI component
- Takes model state and area as input
- Returns rendered widgets
- No state mutation or business logic

```rust
pub fn render_component(f: &mut Frame, area: Rect, model: &Model) {
    // Pure rendering logic only
}
```

## Implementation Guidelines

### Model Design
- Use owned data structures to avoid lifetime complexity
- Implement `Default` for easy initialization
- Keep state flat when possible to simplify updates
- Use enums for discrete states, structs for complex data

### Event Handling
- Define clear event types for all user interactions
- Handle events atomically in single `update()` calls
- Validate state transitions to prevent invalid states
- Log state changes for debugging

### Rendering Strategy
- Break complex UIs into small, focused renderers
- Pass only the data each renderer needs
- Use consistent parameter ordering: `(frame, area, data)`
- Prefer composition over inheritance for complex layouts

### Testing Approach
- Test state transitions independently of UI framework
- Mock events to verify state changes
- Test renderers with known state values
- Use property-based testing for state invariants

## Benefits of This Architecture

### Maintainability
- Clear separation of concerns
- Explicit state management
- Predictable code organization
- Easy to reason about data flow

### Testability
- Pure functions are easy to test
- State transitions can be tested in isolation
- UI rendering can be verified independently
- No hidden dependencies or global state

### Reliability
- Centralized state prevents inconsistencies
- Immutable data flow prevents race conditions
- Explicit error handling at state boundaries
- Deterministic behavior aids debugging

### Scalability
- Modular renderer architecture
- State can be composed from sub-models
- Easy to add new features without breaking existing code
- Performance optimizations can be applied systematically

## Common Patterns

### State Composition
```rust
pub struct Model {
    pub windows: BTreeMap<usize, WindowModel>,
    pub current_selection: Option<usize>,
    pub ui_state: UiState,
}
```

### Event Processing
```rust
impl Model {
    pub fn update(&mut self, event: Event) {
        match event {
            Event::UserInput(input) => self.handle_input(input),
            Event::DataUpdate(data) => self.update_data(data),
            Event::Timer => self.handle_timer(),
        }
    }
}
```

### Renderer Composition
```rust
pub fn render_main(f: &mut Frame, area: Rect, model: &Model) {
    let layout = MainLayout::new(area);

    render_header(f, layout.header, &model.header_state);
    render_content(f, layout.content, &model.content_state);
    render_footer(f, layout.footer, &model.footer_state);
}
```

## Performance Considerations

### Efficient Updates
- Only re-render when state actually changes
- Use dirty flags for expensive computations
- Batch related state changes in single updates
- Avoid unnecessary allocations in hot paths

### Rendering Optimization
- Cache expensive layout calculations
- Use widget recycling for large lists
- Minimize string allocations during rendering
- Profile rendering performance with realistic data

### Memory Management
- Use appropriate collection types for data access patterns
- Clear old data when no longer needed
- Consider using arena allocation for temporary objects
- Monitor memory usage during long-running sessions

## Migration Strategy

When refactoring existing TUI code to this architecture:

1. **Extract State**: Move all mutable state into a central `Model`
2. **Identify Events**: Define events for all state-changing operations
3. **Create Update Logic**: Implement pure update functions
4. **Separate Renderers**: Extract UI rendering into pure functions
5. **Add Tests**: Write tests for state transitions and rendering
6. **Iterate**: Gradually improve separation and add features

This architecture provides a solid foundation for building complex, maintainable terminal applications that can evolve confidently over time.