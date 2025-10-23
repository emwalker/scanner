# Elm Architecture with ECS Integration

The Elm Architecture and ECS (Entity-Component-System) patterns serve different purposes but can work together effectively in complex applications. The scanner project uses ECS for hardware/scanning logic and Elm for UI/TUI state management.

## Separation of Concerns

**ECS Domain (Core Logic)**
- Hardware enumeration and tuner management
- Scanning operations and signal processing
- Entity lifecycle and component updates
- Real-time data processing

**Elm Domain (UI State)**
- User interface state and interactions
- Display state and view composition
- User preferences and navigation
- UI-driven operations

These domains have distinct responsibilities and can be bridged through carefully designed message-passing.

## Boundary Patterns

### Data Flow from ECS to UI

The UI Model subscribes to ECS events and converts them to Messages:

```rust
// ECS produces events
pub enum EcsEvent {
    DeviceEnumerated { device_id: u64, name: String },
    TunerAllocated { tuner_id: TunerId },
    ScanWindowOpened { window_id: WindowId },
    SignalDetected { frequency: f64, strength: f32 },
    ScanCompleted { results: Vec<Station> },
}

// UI consumes events as messages
pub enum Message {
    // ECS events converted to UI messages
    NewDeviceAvailable { device_id: u64, name: String },
    TunerReady { tuner_id: TunerId },
    WindowStarted { window_id: WindowId },
    StationFound { frequency: f64, strength: f32 },
    ScanDone { stations: Vec<Station> },

    // UI-specific messages
    UserSelectedFrequency(f64),
    ViewModeChanged(ViewMode),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        // Convert ECS events to state changes
        Message::NewDeviceAvailable { device_id, name } => {
            model.available_devices.insert(device_id, name);
        }
        Message::StationFound { frequency, strength } => {
            model.scan_results.push(Station { frequency, strength });
        }
        // Handle UI interactions
        Message::UserSelectedFrequency(freq) => {
            model.selected_frequency = Some(freq);
        }
    }
    model
}
```

### Data Flow from UI to ECS

UI events trigger ECS commands:

```rust
pub enum Command {
    InitiateScan { frequency: f64, duration_ms: u32 },
    AllocateTuner,
    OpenWindow { frequency: f64 },
    SaveUserPreferences { preferences: UserPreferences },
}

pub fn execute_command(cmd: Command, world: &mut World) {
    match cmd {
        Command::InitiateScan { frequency, duration_ms } => {
            // Emit ECS event or directly manipulate entities
            world.spawn(ScanRequest { frequency, duration_ms });
        }
        Command::OpenWindow { frequency } => {
            // Create ECS entity for the window
            world.spawn(WindowEntity {
                frequency,
                state: WindowState::Opening,
            });
        }
    }
}
```

## Practical Example: Scan Workflow

Here's how UI and ECS coordinate:

```rust
// UI Model state
pub struct Model {
    scan_request: ScanRequestState,
    available_frequencies: Vec<f64>,
    scan_results: Vec<Station>,
}

pub enum ScanRequestState {
    Idle,
    Pending { frequency: f64 },
    Active { frequency: f64, windows_open: u32 },
    Complete { stations: Vec<Station> },
}

pub enum Message {
    // User actions
    UserStartedScan { frequency: f64 },

    // ECS events
    WindowOpened { window_id: WindowId },
    StationDetected { station: Station },
    WindowClosed { window_id: WindowId },
    AllWindowsClosed,
}

pub fn update(mut model: Model, msg: Message) -> (Model, Vec<Command>) {
    let mut commands = vec![];

    match msg {
        Message::UserStartedScan { frequency } => {
            model.scan_request = ScanRequestState::Pending { frequency };
            // Tell ECS to start scanning
            commands.push(Command::InitiateScan {
                frequency,
                duration_ms: 5000,
            });
        }

        Message::WindowOpened { window_id } => {
            if let ScanRequestState::Pending { frequency } = model.scan_request {
                model.scan_request = ScanRequestState::Active {
                    frequency,
                    windows_open: 1,
                };
            } else if let ScanRequestState::Active { frequency, windows_open } =
                model.scan_request
            {
                model.scan_request = ScanRequestState::Active {
                    frequency,
                    windows_open: windows_open + 1,
                };
            }
        }

        Message::StationDetected { station } => {
            model.scan_results.push(station);
        }

        Message::WindowClosed { window_id } => {
            if let ScanRequestState::Active { frequency, windows_open } = model.scan_request {
                if windows_open <= 1 {
                    model.scan_request = ScanRequestState::Complete {
                        stations: model.scan_results.clone(),
                    };
                } else {
                    model.scan_request = ScanRequestState::Active {
                        frequency,
                        windows_open: windows_open - 1,
                    };
                }
            }
        }

        Message::AllWindowsClosed => {
            model.scan_request = ScanRequestState::Complete {
                stations: model.scan_results.clone(),
            };
        }
    }

    (model, commands)
}

pub fn view(model: &Model) -> Widget {
    match &model.scan_request {
        ScanRequestState::Idle => render_idle(),
        ScanRequestState::Pending { frequency } => {
            render_waiting_for_scan(*frequency)
        }
        ScanRequestState::Active { frequency, windows_open } => {
            render_scanning(*frequency, *windows_open)
        }
        ScanRequestState::Complete { stations } => {
            render_results(stations)
        }
    }
}
```

## Event Bus Pattern

For larger applications, use an event bus to decouple ECS from UI:

```rust
pub struct EventBus {
    events: Arc<Mutex<Vec<EcsEvent>>>,
}

impl EventBus {
    pub fn emit(&self, event: EcsEvent) {
        let mut events = self.events.lock().unwrap();
        events.push(event);
    }

    pub fn consume_all(&self) -> Vec<EcsEvent> {
        let mut events = self.events.lock().unwrap();
        events.drain(..).collect()
    }
}

// Main loop
pub fn run(mut model: Model, world: &mut World, bus: &EventBus) {
    loop {
        // Consume ECS events
        for event in bus.consume_all() {
            let msg = event_to_message(event);
            let (new_model, commands) = update(model, msg);
            model = new_model;

            // Execute commands
            for cmd in commands {
                execute_command(cmd, world);
            }
        }

        // Render UI
        render(model);
    }
}

fn event_to_message(event: EcsEvent) -> Message {
    match event {
        EcsEvent::SignalDetected { frequency, strength } => {
            Message::StationFound { frequency, strength }
        }
        // ... other conversions
    }
}
```

## Testing Interactions

Test the boundary between Elm and ECS:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_request_triggers_command() {
        let model = Model::default();
        let msg = Message::UserStartedScan { frequency: 88.1 };
        let (updated_model, commands) = update(model, msg);

        assert!(matches!(updated_model.scan_request, ScanRequestState::Pending { .. }));
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0], Command::InitiateScan { .. }));
    }

    #[test]
    fn test_window_events_update_scan_state() {
        let mut model = Model {
            scan_request: ScanRequestState::Pending { frequency: 88.1 },
            ..Default::default()
        };

        // Window 1 opens
        let msg = Message::WindowOpened { window_id: WindowId(0) };
        let (model, _) = update(model, msg);
        assert!(matches!(
            model.scan_request,
            ScanRequestState::Active { windows_open: 1, .. }
        ));

        // Window 2 opens
        let msg = Message::WindowOpened { window_id: WindowId(1) };
        let (model, _) = update(model, msg);
        assert!(matches!(
            model.scan_request,
            ScanRequestState::Active { windows_open: 2, .. }
        ));

        // Window 1 closes
        let msg = Message::WindowClosed { window_id: WindowId(0) };
        let (model, _) = update(model, msg);
        assert!(matches!(
            model.scan_request,
            ScanRequestState::Active { windows_open: 1, .. }
        ));
    }
}
```

## Design Principles

1. **Unidirectional for Core Data**: Core data flows from ECS to UI, not the reverse
2. **Commands for Requests**: UI sends commands to ECS, not direct function calls
3. **Events for Notifications**: ECS emits events that UI consumes as messages
4. **Type Conversion at Boundary**: Convert between ECS and UI types at the boundary only
5. **Decoupled Modules**: ECS shouldn't know about UI, and UI shouldn't know internal ECS structure
6. **Testable Boundaries**: Test that commands and messages convert correctly

## Anti-Patterns to Avoid

❌ **Don't** pass ECS entities directly to UI code
❌ **Don't** have UI call ECS functions directly
❌ **Don't** put ECS-specific types in UI messages
❌ **Don't** have ECS depend on UI code
❌ **Don't** share mutable state directly between systems

Instead:
✅ Define clear message and command types at the boundary
✅ Convert between domains at the boundary
✅ Use event bus or message passing for communication
✅ Keep each domain's logic encapsulated
