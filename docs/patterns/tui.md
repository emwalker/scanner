# TUI (Terminal User Interface) Design Patterns

This document catalogs design patterns specific to building terminal user interfaces (TUIs), based on research into modern frameworks like Ratatui, Textual, and production TUI applications.

## Architecture Patterns

### The Elm Architecture (TEA)

A functional pattern originally from the Elm language that provides clear separation of concerns for TUI applications.

**Components:**
- **Model**: Application state represented as data structures (typically a struct in Rust)
- **Update**: Pure functions that transform the model based on messages/events
- **View**: Pure rendering functions that convert model state to UI elements

**When to use:**
- Complex stateful applications where predictable state management is critical
- Applications requiring time-travel debugging or state replay
- Multi-view applications with shared state
- When team members need clear architectural boundaries

**When NOT to use:**
- Simple single-screen utilities with minimal state
- Performance-critical applications where message overhead is problematic
- Real-time streaming applications where state snapshots are impractical

**References:** Ratatui documentation on TEA

---

### Model-View-Controller (MVC)

Traditional UI pattern adapted for terminal applications, separating data, presentation, and input handling.

**Components:**
- **Model**: Application data and business logic
- **View**: Terminal rendering and visual presentation
- **Controller**: Input handling and orchestration

**When to use:**
- Large applications with multiple views of the same data
- Applications requiring separation between UI and business logic for testing
- Teams with members specialized in different layers

**When NOT to use:**
- Simple tools where separation adds unnecessary complexity
- Applications with tight coupling between display and input (e.g., modal editors)
- Real-time applications where controller indirection adds latency

---

### Immediate Mode Rendering

UI paradigm where the entire interface is redrawn from scratch each frame based on current application state.

**How it works:**
- No persistent widget objects between frames
- Application state is the source of truth
- Each frame calls render functions that output the complete UI
- The rendering library handles optimization (buffering, diffing)

**When to use:**
- Applications with dynamic, data-driven UIs
- When UI structure changes frequently based on state
- Applications where simple, predictable rendering is valued over performance
- When building custom widgets or non-standard layouts

**When NOT to use:**
- Applications requiring fine-grained widget state management
- Complex forms with many stateful input fields
- Applications where widget object identity matters (animations, focus management)
- Extremely performance-sensitive applications on slow terminals

**Examples:** Ratatui, Dear ImGui

---

### Retained Mode Rendering

UI paradigm where widget objects persist between frames and update themselves when data changes.

**How it works:**
- Widgets are created once and maintain internal state
- Framework automatically redraws widgets when they're invalidated
- Event handlers are attached to widget objects

**When to use:**
- Traditional form-heavy applications
- Applications with many independent stateful widgets
- When leveraging existing widget libraries with rich behavior
- Applications requiring efficient partial updates

**When NOT to use:**
- Simple data dashboards where full redraws are acceptable
- Applications where widget tree structure changes frequently
- When predictable, reproducible rendering is critical

**Examples:** TurboVision, Textual

---

## Event Handling Patterns

### Event Loop with Channels

Decouples input handling from rendering by using message passing between threads.

**How it works:**
- Main thread runs the render loop
- Separate thread(s) handle input events
- Events are sent via channels (e.g., tokio::mpsc) to main thread
- Render loop processes queued events between frames

**When to use:**
- Applications with multiple event sources (keyboard, mouse, network, timers)
- Async applications using Tokio or similar runtimes
- Applications requiring non-blocking input
- When implementing The Elm Architecture or similar message-passing patterns

**When NOT to use:**
- Simple synchronous applications where blocking on input is acceptable
- Applications where channel overhead impacts performance
- Single-threaded environments without async support

**References:** Ratatui best practices discussion

---

### Command Pattern for Key Bindings

Reifies key presses as command objects, enabling configurable key mappings.

**How it works:**
- Define a Command or Action enum representing all application actions
- Maintain a map from key events to commands
- Input handler looks up commands and executes them
- Configuration files map keys to command names

**When to use:**
- Applications supporting customizable key bindings
- Applications with multiple input modes (normal, insert, command)
- When implementing vim-like modal editing
- Applications requiring keyboard shortcut documentation

**When NOT to use:**
- Simple applications with fixed key bindings
- Applications where key handling is context-dependent and can't be abstracted
- Performance-critical input handling where dispatch overhead matters

---

### Focus Management with Roving Tabindex

Manages keyboard navigation through UI components using tab and arrow keys.

**How it works:**
- Only one widget in a container is tab-reachable at a time
- Tab/Shift-Tab moves focus between containers
- Arrow keys move focus within containers
- Focus state is tracked and updated based on keyboard input

**When to use:**
- Multi-widget forms and complex UIs
- Applications requiring accessible keyboard navigation
- Applications with hierarchical widget structures
- When implementing ARIA-compliant accessibility patterns

**When NOT to use:**
- Single-widget or linear UIs
- Modal editors where all input goes to the editor (vim-style)
- Applications using command palette as primary navigation

**References:** W3C ARIA practices

---

### Modal State Machines

Explicitly models application modes as state machine states.

**How it works:**
- Define an enum representing all application modes
- Each mode has its own event handler
- State transitions are explicit and validated
- Input in one mode doesn't affect other modes

**When to use:**
- Modal editors (vim, emacs)
- Applications with distinct interaction modes (browse, edit, command)
- Applications where mode confusion is dangerous
- Complex applications requiring state machine documentation

**When NOT to use:**
- Modeless applications
- Simple single-purpose tools
- Applications where users expect unified behavior across modes

---

## Layout Patterns

### Constraint-Based Layout

Uses declarative constraints to define widget sizes and positions, similar to CSS flexbox.

**Constraint types:**
- `Length(n)`: Fixed size in cells
- `Percentage(n)`: Percentage of parent space
- `Ratio(m, n)`: Fractional allocation
- `Min(n)` / `Max(n)`: Bounds
- `Fill`: Take remaining space

**When to use:**
- Responsive layouts that adapt to terminal size
- Complex nested layouts
- Applications running in variable terminal sizes
- When building reusable layout templates

**When NOT to use:**
- Simple fixed-size UIs
- Applications targeting specific terminal dimensions
- When precise pixel-perfect layout is required (not possible in terminal)

**Examples:** Ratatui Layout, Textual CSS

---

### Nested Layouts

Recursively divides screen space into smaller regions for widget placement.

**How it works:**
- Create outer layout dividing screen into major regions
- Within each region, create sub-layouts for finer divisions
- Continue nesting as needed for complex UIs
- Each layout independently manages its constraints

**When to use:**
- Complex multi-panel interfaces
- Split-screen or tiled layouts
- Dashboards with many data regions
- Applications mimicking multi-window environments

**When NOT to use:**
- Simple single-panel UIs
- Applications with absolute positioning requirements
- When layout hierarchy becomes too deep (>4-5 levels)

---

### Viewport and Scrolling

Manages rendering of content larger than available screen space.

**Patterns:**
- **Scrollback buffer**: Maintains off-screen content above viewport
- **Paginated scrolling**: Fixed-size page navigation (PgUp/PgDn)
- **Smooth scrolling**: Line-by-line viewport adjustment
- **Virtual scrolling**: Only renders visible items for large lists

**When to use:**
- Log viewers, file browsers, data tables
- Applications displaying large amounts of data
- Document viewers and editors
- Applications with dynamic content that can exceed screen size

**When NOT to use:**
- Fixed-content UIs that fit on screen
- Applications where scrolling is confusing (modal dialogs)
- Real-time dashboards where users shouldn't scroll

---

## Rendering Patterns

### Double Buffering with Diffing

Eliminates flicker by computing differences between frames and only updating changed cells.

**How it works:**
- Maintain two buffers: front buffer (displayed) and back buffer (next frame)
- Render complete frame to back buffer
- Compute diff between buffers (cell-level comparison)
- Output only escape sequences for changed cells
- Swap buffers

**When to use:**
- All production TUI applications
- Applications with frequent redraws
- Applications running over slow connections (SSH)
- When flicker-free rendering is critical

**When NOT to use:**
- Debug/diagnostic tools where raw output is valuable
- Applications on extremely memory-constrained systems
- Simple single-frame rendering (e.g., one-shot CLI output)

**Examples:** Ratatui, r3bl_tui, rxtui

---

### Cursor Management

Controls when and where the terminal cursor is visible.

**Patterns:**
- **Hide during redraw**: Prevent cursor flicker during rendering
- **Restore after render**: Show cursor at appropriate position for input
- **Cursor shape**: Use different shapes for different modes (block, line, underline)

**When to use:**
- All TUI applications with redraws
- Applications with text input fields
- Modal editors where cursor indicates mode
- Applications where cursor position conveys information

**When NOT to use:**
- Read-only dashboards without text input
- Applications where cursor is always hidden

---

### Partial Updates

Updates only changed regions rather than full-screen redraws.

**When to use:**
- Real-time dashboards with independent updating regions
- Applications with mostly static UI and dynamic data regions
- Performance optimization for large terminals
- Reducing bandwidth over network connections

**When NOT to use:**
- When using frameworks that already optimize (double buffering with diffing)
- Simple applications where full redraw is fast enough
- When partial update complexity outweighs performance benefit

---

## Component Patterns

### Component Encapsulation

Encapsulates widget state, rendering, and event handling into reusable components.

**How it works:**
- Define a struct containing component state
- Implement render method that outputs UI for current state
- Implement event handler method for input
- Optionally implement lifecycle methods (init, cleanup)

**When to use:**
- Building reusable custom widgets
- Complex widgets with internal state (tables, forms, editors)
- When building widget libraries
- Sharing UI components across applications

**When NOT to use:**
- One-off UI elements specific to single application
- Stateless rendering functions (use simple functions instead)
- When framework provides sufficient built-in widgets

---

### Stateful vs Stateless Widgets

Design choice between widgets that maintain internal state vs pure rendering functions.

**Stateful:**
- Widget objects hold state
- Widgets update themselves
- Easier to implement complex interactive widgets

**Stateless:**
- Rendering functions take state as parameters
- Application code manages all state
- Easier to reason about, test, and debug

**When to use stateful:**
- Complex interactive widgets (text editors, trees, tables)
- Widgets with significant internal behavior
- When using retained-mode frameworks

**When to use stateless:**
- Simple display widgets (labels, progress bars)
- When using immediate-mode frameworks
- When predictable, pure rendering is valued

---

### Composite Widgets

Builds complex widgets by composing simpler widgets.

**How it works:**
- Container widget manages child widgets
- Container handles layout of children
- Container delegates events to appropriate children
- Children render themselves within assigned regions

**When to use:**
- Building complex UIs from simple primitives
- Creating reusable compound widgets (dialogs, forms)
- Managing widget hierarchies
- When implementing nested layouts

**When NOT to use:**
- Flat single-level UIs
- When composition adds unnecessary indirection
- Performance-critical rendering where composition overhead matters

---

## State Management Patterns

### Unidirectional Data Flow (UDF)

State flows in one direction: user input → state update → rendering.

**Benefits:**
- Predictable state changes
- Easier debugging and testing
- Clear separation of concerns
- Supports state replay and time-travel debugging

**When to use:**
- Complex stateful applications
- Applications requiring state persistence
- Team projects where predictability is valued
- When implementing The Elm Architecture

**When NOT to use:**
- Simple applications with minimal state
- Applications where bidirectional binding is natural (traditional forms)
- When adopting framework with different patterns

**Examples:** The Elm Architecture, Redux/Flux-inspired patterns

---

### State Holder Pattern

Delegates state management to specialized holder objects.

**How it works:**
- Create state holder objects separate from UI components
- State holders encapsulate state and update logic
- UI components observe state holders
- Multiple components can share state holders

**When to use:**
- Applications with shared state across views
- Implementing Model-View-ViewModel (MVVM)
- When separating UI from business logic
- Testing state logic independently of UI

**When NOT to use:**
- Simple applications with component-local state
- When state is naturally coupled to specific widgets
- Applications where indirection complicates rather than clarifies

---

### Global vs Component-Local State

Design decision about where to place state in application hierarchy.

**Global state:**
- Single source of truth for entire application
- Accessible from any component
- Easier to serialize and persist

**Component-local state:**
- State owned by individual widgets
- Natural encapsulation
- Reduces coupling

**When to use global:**
- Application-wide settings and configuration
- Authenticated user information
- Data shared across multiple views
- When implementing single-store patterns (Redux)

**When to use local:**
- Widget-specific UI state (expanded/collapsed, selected item)
- Temporary form input before submission
- State that doesn't affect other components

---

## Interaction Patterns

### Command Palette

Searchable list of all application commands, typically invoked with Ctrl+Shift+P or similar.

**When to use:**
- Applications with many commands
- Applications supporting discoverability
- When keyboard shortcuts are hard to remember
- Implementing VS Code-like interaction

**When NOT to use:**
- Simple applications with few commands
- Applications where command palette is slower than direct shortcuts
- When users are experts who prefer memorized shortcuts

**Examples:** Windows Terminal, Helix editor

---

### Modal Dialogs and Popups

Overlays that temporarily take focus and require user response.

**Patterns:**
- **Modal dialogs**: Block interaction with main UI
- **Non-modal popups**: Allow background interaction
- **Tooltips**: Contextual information overlays

**When to use:**
- Confirmations for destructive actions
- Forms for creating/editing entities
- Error messages requiring acknowledgment
- Contextual help and documentation

**When NOT to use:**
- Non-blocking notifications (use status bar or notifications area)
- When dialog interrupts critical workflows
- Excessive nesting of modals (modal fatigue)

---

### Vim-Style Modal Editing

Multiple input modes where keys have different meanings.

**Common modes:**
- **Normal mode**: Navigation and commands
- **Insert mode**: Text input
- **Visual mode**: Selection
- **Command mode**: Execute text commands

**When to use:**
- Text editors and text-heavy applications
- Applications where users become power users
- When keyboard efficiency is paramount
- Applications targeting vim users

**When NOT to use:**
- Casual-use applications
- Applications for non-technical users
- When mode confusion causes errors
- Simple linear workflows

---

## Async Patterns

### Non-Blocking Event Loop

Prevents long-running operations from freezing the UI.

**How it works:**
- Main thread runs event loop
- Long operations run in background tasks
- Progress and results communicated via channels
- UI updates on main thread based on task messages

**When to use:**
- Network operations (HTTP requests, websockets)
- File I/O on large files
- CPU-intensive computations
- Any operation that might block

**When NOT to use:**
- Simple synchronous applications
- Operations that complete instantly
- When async complexity outweighs benefit

**Examples:** Tokio-based TUI applications, r3bl_tui

---

### In-Band vs Out-of-Band Events

Distinguishes between synchronous user input and asynchronous external events.

**In-band:**
- Keyboard and mouse input
- Processed synchronously in event loop
- No delay

**Out-of-band:**
- Network events
- Timer/interval events
- File system notifications
- Variable delay, requires async handling

**When to use distinction:**
- Applications with multiple event sources
- Prioritizing user input over background events
- Implementing responsive UIs with background tasks

**When NOT to use:**
- Simple single-event-source applications
- When all events can be treated uniformly

---

## Testing Patterns

### Snapshot Testing

Compares rendered output against saved snapshots rather than specific assertions.

**How it works:**
- Render UI to in-memory buffer
- Convert buffer to string representation
- Compare against saved snapshot file
- Update snapshots when intentionally changing UI

**When to use:**
- Testing TUI rendering where visual output is the requirement
- Regression testing to catch unintended changes
- Testing complex layouts difficult to assert programmatically
- Golden master testing of UI

**When NOT to use:**
- Testing business logic (use unit tests)
- Testing interactive behavior (use behavioral tests)
- When snapshots are too large or change too frequently

**Tools:** TUI Test (Microsoft), insta (Rust crate)

---

### Event-Driven Testing

Structures applications as event-driven systems to facilitate testing.

**How it works:**
- Application responds to discrete events
- Tests inject events and verify resulting state
- No direct UI interaction required for testing
- Can test without rendering (headless mode)

**When to use:**
- Testing complex interaction sequences
- Integration testing of TUI applications
- Automated testing in CI/CD pipelines
- When UI framework doesn't support introspection

**When NOT to use:**
- Testing actual rendering (use snapshot tests)
- Testing terminal compatibility (use end-to-end tests)

---

### Page Object Pattern

Models TUI screens as objects with methods for interaction, abstracting test code from UI details.

**How it works:**
- Create classes/structs representing each screen
- Methods perform user actions (navigate, input, select)
- Methods return elements or other page objects
- Tests use page objects rather than direct UI manipulation

**When to use:**
- Large test suites with many UI interaction tests
- When UI changes frequently
- Sharing interaction code across tests
- Team projects with dedicated QA

**When NOT to use:**
- Small applications with few tests
- When abstraction overhead exceeds benefit
- Simple one-screen applications

---

## Theming and Styling

### Color Scheme Abstraction

Separates colors from widgets, allowing user customization.

**Patterns:**
- **Semantic colors**: Named by purpose (primary, error, warning)
- **ANSI colors**: Standard 16-color palette
- **Theme files**: External configuration (JSON, TOML, YAML)

**When to use:**
- Applications supporting user themes
- Accessibility requirements (high contrast, colorblind-friendly)
- Applications in professional/enterprise settings
- When terminal color support varies

**When NOT to use:**
- Simple single-purpose tools
- Applications where color is not significant
- Debug/diagnostic tools with hardcoded formatting

---

### Responsive Design

Adapts UI to terminal size changes.

**Patterns:**
- **Flexible layouts**: Use percentage/ratio constraints
- **Visibility thresholds**: Hide elements when too small
- **Simplified views**: Show abbreviated UI in small terminals
- **Terminal size detection**: Query and respond to SIGWINCH

**When to use:**
- Applications used in various terminal sizes
- Split-screen and tiled terminal environments
- Applications used over SSH on mobile devices
- Professional tools where window management is unpredictable

**When NOT to use:**
- Applications with fixed minimum size requirements
- When simplified view provides poor UX
- Full-screen applications always maximized

---

## Pattern Interactions in Real Systems

Modern TUI applications typically combine multiple patterns:

**Typical dashboard application:**
- Immediate Mode Rendering for simplicity
- Constraint-Based Layout for responsiveness
- Event Loop with Channels for non-blocking updates
- Double Buffering with Diffing for performance
- Viewport and Scrolling for large datasets

**Text editor:**
- Vim-Style Modal Editing for efficiency
- Retained Mode Rendering for complex text widgets
- Focus Management for multi-pane layouts
- Command Palette for discoverability
- State Machine for modes

**Network monitoring tool:**
- The Elm Architecture for state management
- Non-Blocking Event Loop for network I/O
- Snapshot Testing for UI validation
- Responsive Design for various terminals
- Color Scheme Abstraction for customization

---

## References

### Frameworks and Libraries
- Ratatui (Rust): https://ratatui.rs/
- Textual (Python): https://textual.textualize.io/
- Blessed (JavaScript): https://github.com/chjj/blessed
- TurboVision: Classic DOS TUI framework

### Documentation and Articles
- Ratatui Concepts: https://ratatui.rs/concepts/
- Testing TUI Apps: https://blog.waleedkhan.name/testing-tui-apps/
- TUI Test Framework: https://github.com/microsoft/tui-test
- W3C ARIA Keyboard Practices: https://www.w3.org/WAI/ARIA/apg/practices/keyboard-interface/

### Books and Papers
- Immediate Mode GUI Programming (Ryan Fleury)
- Game Programming Patterns (Robert Nystrom) - Event Queue pattern
- The Elm Architecture (Evan Czaplicki)

### Example Projects
- Helix: Modern text editor
- lazygit: Git terminal UI
- bottom: System monitor
- GitUI: Fast terminal UI for git
