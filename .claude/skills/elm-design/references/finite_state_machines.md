# Finite State Machines with Elm Architecture

The Elm Architecture naturally represents applications as finite state machines (FSMs). By encoding valid states as an enum, the compiler ensures you handle all cases and prevents invalid state transitions.

## Core Concept

A finite state machine has:
- A fixed set of valid states
- Explicit transitions between states
- Input that triggers state changes
- Output/actions for each state

The Elm Architecture maps directly to this:
- States = Model enum variants
- Input = Messages
- Output = New state (and side effects)

## Basic FSM Example

**Simple scan operation with clear states:**

```rust
#[derive(Clone)]
pub enum ScanState {
    Idle,
    Scanning { frequency: f64, elapsed_ms: u32 },
    Complete { results: Vec<Station> },
    Failed { error: String },
}

pub struct Model {
    scan_state: ScanState,
}

pub enum Message {
    ScanRequested { frequency: f64 },
    ScanProgress { elapsed_ms: u32 },
    ScanFinished { results: Vec<Station> },
    ScanError { error: String },
    Reset,
}

pub fn update(mut model: Model, msg: Message) -> Model {
    model.scan_state = match (model.scan_state, msg) {
        // Valid transitions
        (ScanState::Idle, Message::ScanRequested { frequency }) => {
            ScanState::Scanning { frequency, elapsed_ms: 0 }
        }
        (ScanState::Scanning { frequency, .. }, Message::ScanProgress { elapsed_ms }) => {
            ScanState::Scanning { frequency, elapsed_ms }
        }
        (ScanState::Scanning { .. }, Message::ScanFinished { results }) => {
            ScanState::Complete { results }
        }
        (ScanState::Scanning { .. }, Message::ScanError { error }) => {
            ScanState::Failed { error }
        }

        // Reset from any state
        (_, Message::Reset) => ScanState::Idle,

        // Invalid transitions (stay in current state)
        (state, _) => state,
    };
    model
}

pub fn view(model: &Model) -> String {
    match &model.scan_state {
        ScanState::Idle => "Ready to scan".to_string(),
        ScanState::Scanning { frequency, elapsed_ms } => {
            format!("Scanning {} MHz ({} ms elapsed)", frequency, elapsed_ms)
        }
        ScanState::Complete { results } => {
            format!("Found {} stations", results.len())
        }
        ScanState::Failed { error } => {
            format!("Scan failed: {}", error)
        }
    }
}
```

**Key benefit:** Invalid transitions are impossible. For example, you cannot transition directly from `Idle` to `Complete` without going through `Scanning`.

## Multi-State Model

For complex applications with multiple independent state machines:

```rust
#[derive(Clone)]
pub enum ScanState {
    Idle,
    Scanning { frequency: f64 },
    Complete { results: Vec<Station> },
    Failed { error: String },
}

#[derive(Clone)]
pub enum DialogState {
    Hidden,
    ConfiguringFrequency,
    ConfirmingScan { frequency: f64 },
}

#[derive(Clone)]
pub enum TabState {
    Results,
    Settings,
    History,
}

pub struct Model {
    scan_state: ScanState,
    dialog_state: DialogState,
    active_tab: TabState,
}

pub enum Message {
    ScanMessage(ScanMessage),
    DialogMessage(DialogMessage),
    TabMessage(TabMessage),
}

pub enum ScanMessage {
    ScanRequested { frequency: f64 },
    ScanCompleted { results: Vec<Station> },
    ScanFailed { error: String },
}

pub enum DialogMessage {
    OpenDialog,
    CloseDialog,
    FrequencyEntered { frequency: f64 },
}

pub enum TabMessage {
    SwitchTab(TabState),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::ScanMessage(scan_msg) => {
            // Update scan state machine
            model.scan_state = update_scan_state(model.scan_state, scan_msg);
        }
        Message::DialogMessage(dialog_msg) => {
            // Update dialog state machine
            model.dialog_state = update_dialog_state(model.dialog_state, dialog_msg);
        }
        Message::TabMessage(tab_msg) => {
            // Update tab state machine
            match tab_msg {
                TabMessage::SwitchTab(new_tab) => model.active_tab = new_tab,
            }
        }
    }
    model
}
```

## Handling Impossible States

Some states are impossible to reach together. Model this with enums:

**Bad: Allows impossible combinations**

```rust
pub struct Model {
    is_authenticated: bool,
    user_id: Option<u64>,
    permissions: Vec<Permission>,
}

// What if is_authenticated is false but user_id and permissions are Some?
```

**Good: Makes impossible states unrepresentable**

```rust
pub enum AuthState {
    Unauthenticated,
    Authenticated {
        user_id: u64,
        permissions: Vec<Permission>,
    },
}

pub struct Model {
    auth_state: AuthState,
}

// Impossible to have user_id without being authenticated
```

## State Machines with Nested Models

Combine FSM patterns with nested TEA:

```rust
pub mod window_state {
    #[derive(Clone)]
    pub enum State {
        Processing,
        Tuned { signal_strength: f32 },
        Rejected,
    }

    pub struct Model {
        pub state: State,
        pub frequency: f64,
    }

    pub enum Message {
        ProcessingStarted,
        Tuned { signal_strength: f32 },
        Rejected,
    }

    pub fn update(mut model: Model, msg: Message) -> Model {
        model.state = match (model.state, msg) {
            (State::Processing, Message::Tuned { signal_strength }) => {
                State::Tuned { signal_strength }
            }
            (State::Processing, Message::Rejected) => State::Rejected,
            (state, _) => state,
        };
        model
    }
}

pub struct Model {
    windows: HashMap<WindowId, window_state::Model>,
}

pub enum Message {
    WindowMessage(WindowId, window_state::Message),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::WindowMessage(id, window_msg) => {
            if let Some(window) = model.windows.get_mut(&id) {
                *window = window_state::update(window.clone(), window_msg);
            }
        }
    }
    model
}
```

## Testing FSMs

State machines are easy to test comprehensively:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_scan_transition() {
        let mut model = Model {
            scan_state: ScanState::Idle,
        };

        let msg = Message::ScanRequested { frequency: 88.1 };
        model = update(model, msg);

        match model.scan_state {
            ScanState::Scanning { frequency, .. } => {
                assert_eq!(frequency, 88.1);
            }
            _ => panic!("Expected Scanning state"),
        }
    }

    #[test]
    fn test_cannot_complete_without_scanning() {
        let mut model = Model {
            scan_state: ScanState::Idle,
        };

        // Try to complete without going through Scanning
        let msg = Message::ScanFinished {
            results: vec![],
        };
        let updated = update(model.clone(), msg);

        // State should remain unchanged
        assert!(matches!(updated.scan_state, ScanState::Idle));
    }

    #[test]
    fn test_reset_from_any_state() {
        let states = vec![
            ScanState::Idle,
            ScanState::Scanning { frequency: 88.1, elapsed_ms: 100 },
            ScanState::Complete { results: vec![] },
            ScanState::Failed { error: "test".to_string() },
        ];

        for state in states {
            let mut model = Model { scan_state: state };
            model = update(model, Message::Reset);
            assert!(matches!(model.scan_state, ScanState::Idle));
        }
    }
}
```

## Visualization

State machines can be documented with ASCII diagrams:

```
                    ScanRequested
                        ↓
    ┌─────────┐      ┌──────────┐
    │  Idle   │─────→│ Scanning │
    └─────────┘      └──────────┘
        ↑                │
        │                │
        └────────────────┤
         (Reset)         │
                         ├─→ ScanFinished → Complete
                         │
                         └─→ ScanError → Failed
                             ↓
                         (Reset to Idle)
```

## Benefits of This Approach

1. **Compiler-Enforced Correctness**: Invalid states become impossible
2. **Exhaustive Pattern Matching**: Compiler warns about unhandled cases
3. **Clear Documentation**: State diagram is embedded in code
4. **Easy Testing**: All state transitions can be tested systematically
5. **Deterministic Behavior**: Same input always produces same output
6. **Clear Intent**: Code expresses business rules explicitly
