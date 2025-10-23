---
name: elm-design
description: This skill should be used when designing TUI technical architectures using Elm Architecture patterns, refactoring existing code to align with Elm principles, creating visual designs showing state machines and data flow, or analyzing code for Elm pattern opportunities and anti-patterns. Use this skill to generate beautiful, maintainable TUI designs that follow proven functional programming principles.
---

# Elm Design for TUI Architecture

## Overview

This skill provides comprehensive guidance for designing and implementing TUI (Terminal User Interface) applications using the Elm Architecture pattern. The Elm Architecture separates applications into three core concerns—Model (state), View (rendering), and Update (logic)—creating maintainable, testable, and predictable systems.

This skill is specifically tailored for the scanner project and includes patterns for integrating Elm Architecture with ECS (Entity-Component-System) systems, handling complex state transitions with finite state machines, and composing large applications from smaller components.

## When to Use This Skill

Use this skill when:

- **Designing new TUI features** following Elm Architecture principles
- **Refactoring existing code** to separate concerns (state, rendering, logic)
- **Creating visual designs** like state machine diagrams, data flow diagrams, or component compositions
- **Analyzing code** for Elm pattern opportunities or anti-pattern warnings
- **Documenting architecture** for complex stateful UI systems
- **Integrating with ECS** and coordinating between core logic and UI layers
- **Teaching** the Elm Architecture to team members with concrete Rust examples

## Core Reference Materials

The skill includes five comprehensive reference documents, each with Rust examples:

### 1. `elm_fundamentals.md`
**What it covers:**
- Core concepts: Model, View, Update, Messages
- Pure functions and unidirectional data flow
- How to handle side effects separately
- Message design patterns

**Use when:**
- Learning the basics of Elm Architecture
- Understanding how Model/View/Update interact
- Designing message types for a feature

### 2. `nested_tea_patterns.md`
**What it covers:**
- Composing applications from independent sub-components
- Message routing between parent and child components
- Avoiding "nested page fatigue"
- Shared state between components
- Module organization for larger applications

**Use when:**
- Breaking down large applications into manageable pieces
- Managing communication between independent UI sections
- Organizing modules in a growing codebase

### 3. `elm_anti_patterns.md`
**What it covers:**
- Seven common architectural mistakes
- Improper state modeling (multiple booleans instead of enum states)
- Mixed concerns (side effects in update/view)
- Implicit component dependencies
- View functions with side effects
- How to fix each anti-pattern with examples

**Use when:**
- Code review and architecture improvement
- Identifying subtle bugs caused by bad state design
- Learning what NOT to do

### 4. `finite_state_machines.md`
**What it covers:**
- Modeling applications as finite state machines
- Encoding valid states and transitions
- Making impossible states unrepresentable
- Testing state transitions
- Visualizing state machines as ASCII diagrams

**Use when:**
- Modeling features with clear state transitions
- Ensuring invalid state combinations are impossible
- Documenting state machines visually

### 5. `elm_ecs_integration.md`
**What it covers:**
- Boundary patterns between Elm UI and ECS core logic
- Data flow from ECS to UI (events → messages)
- Data flow from UI to ECS (commands)
- Event bus patterns for decoupling
- Testing the boundary between systems

**Use when:**
- Designing how the UI communicates with scanning/hardware systems
- Coordinating state between ECS and Elm
- Ensuring clean separation of concerns

## Available Scripts

### `analyze_elm_patterns.py`
Analyzes Rust code for Elm Architecture patterns and anti-patterns.

**Usage:**
```bash
python3 analyze_elm_patterns.py src/ui/tui/model/state.rs
```

**What it detects:**
- Model structs and their fields
- Update, View, and Render functions
- Message/Event enums
- Anti-patterns:
  - Boolean flag state (multiple bool fields that should be an enum)
  - Side effects inside update functions
  - Side effects inside view functions

**Output:** Line-by-line analysis with pattern locations and warnings

### `generate_diagrams.py`
Generates ASCII diagrams for Elm Architecture patterns.

**Usage:**
```bash
# Generate a specific diagram
python3 generate_diagrams.py --elm              # Basic Elm Architecture
python3 generate_diagrams.py --fsm              # Finite State Machine
python3 generate_diagrams.py --nested           # Nested TEA components
python3 generate_diagrams.py --ecs              # ECS integration
python3 generate_diagrams.py --separation       # State/View separation
python3 generate_diagrams.py --all              # All diagrams

# Copy output to documentation
python3 generate_diagrams.py --fsm > scan_fsm.txt
```

**What diagrams show:**
- Model → View → Update → Message cycles
- State machines with transitions
- Component hierarchies
- ECS/Elm integration boundaries

## Workflow: Creating an Elm Architecture Design

Follow this workflow when designing a new TUI feature or refactoring existing code:

### Step 1: Understand the Feature
What states does this feature have? What user interactions matter? What data needs to persist?

### Step 2: Design the State Machine
- Identify all valid states using an enum
- Define transitions between states
- Ensure invalid combinations are impossible
- Consider using the FSM reference

**Example:**
```rust
pub enum ScanState {
    Idle,
    Scanning { frequency: f64 },
    Complete { results: Vec<Station> },
    Failed { error: String },
}
```

### Step 3: Define Messages
What events trigger state changes? List all user interactions and external events.

**Example:**
```rust
pub enum Message {
    UserStartedScan { frequency: f64 },
    ScanProgress { elapsed_ms: u32 },
    ScanCompleted { results: Vec<Station> },
    ScanError { error: String },
}
```

### Step 4: Implement Update Logic
Write pure functions that transform state based on messages.

```rust
pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::UserStartedScan { frequency } => {
            model.state = ScanState::Scanning { frequency };
        }
        // ... other cases
    }
    model
}
```

### Step 5: Implement View Logic
Write pure functions that render state, never mutating state or doing I/O.

```rust
pub fn view(model: &Model) -> Widget {
    match &model.state {
        ScanState::Idle => render_idle(),
        ScanState::Scanning { .. } => render_spinner(),
        ScanState::Complete { results } => render_results(results),
        ScanState::Failed { error } => render_error(error),
    }
}
```

### Step 6: Handle Side Effects
Separate commands/effects from pure logic. Return what needs to happen.

```rust
pub fn update(mut model: Model, msg: Message) -> (Model, Vec<Command>) {
    // ... state changes ...
    (model, commands)
}

pub enum Command {
    PerformScan { frequency: f64 },
    LogEvent(String),
}
```

### Step 7: Document and Test
- Create ASCII diagrams showing state transitions
- Test state transitions independently
- Test that view produces correct output for each state

## Workflow: Refactoring to Elm Architecture

When improving existing code:

### 1. Run the Pattern Analyzer
```bash
python3 scripts/analyze_elm_patterns.py <your_file.rs>
```

Review the report for anti-patterns and opportunities.

### 2. Identify Bad State Patterns
Look for:
- Multiple boolean flags that represent one state
- State checked in multiple places inconsistently
- Updates scattered across the codebase

### 3. Extract to Enum
Replace multiple flags with a single enum representing valid states:

**Before:**
```rust
pub struct Model {
    is_loading: bool,
    has_error: bool,
    data: Vec<Item>,
}
```

**After:**
```rust
pub enum LoadState {
    Idle,
    Loading,
    Success(Vec<Item>),
    Failed(String),
}

pub struct Model {
    state: LoadState,
}
```

### 4. Centralize Updates
If state changes happen in many places, consolidate to a single update function.

### 5. Test Transitions
Write tests for each state transition to ensure they work correctly.

## Workflow: Analyzing and Improving Code

To review someone else's code or your own:

### 1. Check for Model/View/Update Separation
- Does all state live in a single Model?
- Are view functions pure (no I/O, no mutations)?
- Are all state changes in one update function?

### 2. Use the Pattern Analyzer
```bash
python3 scripts/analyze_elm_patterns.py <file.rs>
```

### 3. Generate Diagrams
```bash
python3 scripts/generate_diagrams.py --all
```

Include diagrams in your code review or design document.

### 4. Check for Anti-Patterns
Review `elm_anti_patterns.md` and look for each pattern in the code.

### 5. Suggest Improvements
Use the references to explain why a change would improve the code.

## Working with the Scanner Project

The scanner project uses Elm Architecture for the TUI and ECS for core logic. When working on features:

- **TUI changes**: Use Elm Architecture as described here
- **ECS/Hardware**: Use ECS patterns with systems and components
- **Integration**: Refer to `elm_ecs_integration.md` for boundary patterns

Example: When adding a new scan mode:
1. Design the UI state machine in the TUI layer
2. Define TUI → ECS messages/commands
3. Define ECS → TUI events
4. Implement both layers
5. Document the integration with a diagram

## Common Patterns in Scanner

### Scan Dialog State
```rust
pub enum DialogState {
    Hidden,
    Configuring,
    Scanning { frequency: f64 },
    Complete { results: Vec<Station> },
}
```

### Window Processing State
```rust
pub enum WindowState {
    Opening,
    Processing { progress: f32 },
    Tuned { signal: f32 },
    Rejected,
}
```

### Activity View State
```rust
pub enum ActivityView {
    Tasks { selected: Option<TaskId> },
    Details { task_id: TaskId },
    History,
}
```

## Tips for Effective Elm Architecture

1. **Keep Model flat when possible** - Nested structures work but add complexity
2. **Name messages descriptively** - `UserSelectedStation` not `Select`
3. **Use pattern matching** - Rust's `match` makes state handling clear
4. **Test state transitions** - They're pure functions, test them thoroughly
5. **Write pure functions** - No exceptions, no globals, no side effects
6. **Document state machines** - Use ASCII diagrams in comments or docs
7. **Use enums for state** - Never use multiple booleans to represent one state
8. **Route messages clearly** - Explicit parent/child message flow prevents bugs

## References and Further Reading

**Official Resources:**
- [The Elm Guide](https://guide.elm-lang.org/architecture/)
- [Elm Programming](https://elmprogramming.com/model-view-update-part-1.html)
- [Ratatui Elm Architecture](https://ratatui.rs/concepts/application-patterns/the-elm-architecture/)

**In This Skill:**
- `references/elm_fundamentals.md` - Core concepts
- `references/nested_tea_patterns.md` - Large applications
- `references/elm_anti_patterns.md` - What not to do
- `references/finite_state_machines.md` - State design
- `references/elm_ecs_integration.md` - ECS integration
