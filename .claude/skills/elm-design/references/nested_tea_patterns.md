# Nested TEA Patterns (Composition)

As applications grow, organizing code around a single monolithic Model/Update/View becomes cumbersome. Nested TEA patterns allow breaking an application into smaller, composable sub-applications, each with its own Model, Update, and View.

## Core Concept

Instead of one large update function, organize code into independent modules where each module has:
- Its own Model (local state)
- Its own Update function (local logic)
- Its own View function (local rendering)
- Its own Message enum

The parent module composes these pieces together.

## Basic Pattern: Parent/Child Communication

```rust
// Child module
pub mod scan_dialog {
    #[derive(Clone)]
    pub struct Model {
        pub frequency: f64,
        pub duration_ms: u32,
        pub is_active: bool,
    }

    #[derive(Clone)]
    pub enum Message {
        FrequencyChanged(f64),
        DurationChanged(u32),
        StartScan,
        CancelScan,
    }

    pub fn update(mut model: Model, msg: Message) -> Model {
        match msg {
            Message::FrequencyChanged(f) => {
                model.frequency = f;
            }
            Message::DurationChanged(d) => {
                model.duration_ms = d;
            }
            Message::StartScan => {
                model.is_active = true;
            }
            Message::CancelScan => {
                model.is_active = false;
            }
        }
        model
    }

    pub fn view(model: &Model, area: Rect) -> Widget {
        // Render the scan dialog UI
        Block::default().title("Scan Configuration")
    }
}

// Parent module
pub struct Model {
    pub scan_dialog: scan_dialog::Model,
    pub main_view: MainView,
}

pub enum Message {
    ScanDialog(scan_dialog::Message),
    MainViewMessage(MainMessage),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::ScanDialog(child_msg) => {
            model.scan_dialog = scan_dialog::update(model.scan_dialog, child_msg);
        }
        Message::MainViewMessage(main_msg) => {
            // Handle main view messages
        }
    }
    model
}

pub fn view(model: &Model, area: Rect) -> Widget {
    // Combine child views with parent layout
    let layout = Layout::vertical([
        Constraint::Percentage(80),
        Constraint::Percentage(20),
    ]);

    let [main_area, dialog_area] = layout.areas(area);

    // Parent renders main view and child renders dialog
    scan_dialog::view(&model.scan_dialog, dialog_area)
}
```

## Message Routing Patterns

### Enum Wrapping (Recommended)

Wrap child messages in parent message variants for clear ownership:

```rust
pub enum ParentMessage {
    ScanDialog(ScanDialogMessage),
    Results(ResultsPanelMessage),
    Settings(SettingsPanelMessage),
}

pub fn update(mut model: Model, msg: ParentMessage) -> Model {
    match msg {
        ParentMessage::ScanDialog(m) => {
            model.dialog = scan_dialog::update(model.dialog, m);
        }
        ParentMessage::Results(m) => {
            model.results = results::update(model.results, m);
        }
        ParentMessage::Settings(m) => {
            model.settings = settings::update(model.settings, m);
        }
    }
    model
}
```

**Benefits:**
- Clear ownership chain
- Easy to trace message flow
- Compiler enforces handling all cases

**Drawbacks:**
- More boilerplate with multiple levels of nesting

## Avoiding "Nested Page Fatigue"

A common problem: as you add more nested components, you end up wiring messages and state through many layers.

**Anti-pattern (avoid):**
```rust
pub enum Message {
    OuterDialog(InnerDialog(DialogContent(ButtonClick))),
}
```

**Better approach:**
Use a flatter message structure where appropriate:

```rust
pub enum Message {
    // All messages at one level
    ScanFrequencyChanged(f64),
    ScanStarted,
    DialogClosed,
    ResultsSelected(ResultId),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::ScanFrequencyChanged(f) => {
            model.scan_dialog.frequency = f;
        }
        // All updates in one place
    }
    model
}
```

## Shared State Between Components

Sometimes sibling components need to coordinate or share data.

**Option 1: Lift state to parent**

```rust
pub struct Model {
    pub shared_data: SharedState,
    pub component_a: ComponentA,
    pub component_b: ComponentB,
}

pub enum Message {
    UpdateSharedData(SharedData),
    ComponentAMessage(ComponentAMessage),
    ComponentBMessage(ComponentBMessage),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::UpdateSharedData(data) => {
            model.shared_data = data;
            // Both components can now see the updated data
        }
        Message::ComponentAMessage(m) => {
            model.component_a = component_a::update(model.component_a, m);
        }
        Message::ComponentBMessage(m) => {
            model.component_b = component_b::update(model.component_b, m);
        }
    }
    model
}
```

**Option 2: Create shared commands**

When one component's action should trigger changes in another:

```rust
pub fn update(mut model: Model, msg: Message) -> (Model, Vec<Command>) {
    let mut commands = vec![];

    match msg {
        Message::ComponentAMessage(m) => {
            let (new_a, cmd_a) = component_a::update(model.component_a, m);
            model.component_a = new_a;
            commands.extend(cmd_a);

            // Component A's action may trigger effects in Component B
            if let Some(cross_component_cmd) = handle_component_a_effects(&model) {
                commands.push(cross_component_cmd);
            }
        }
    }

    (model, commands)
}
```

## Module Organization

A typical nested structure:

```
src/ui/tui/
├── model/
│   ├── state.rs           # Main Model
│   ├── messages.rs        # Parent Message enum
│   ├── scan/
│   │   ├── mod.rs         # scan_dialog module
│   │   ├── model.rs       # ScanModel
│   │   └── messages.rs    # ScanMessage
│   ├── results/
│   │   ├── mod.rs         # results panel
│   │   └── model.rs       # ResultsModel
│   └── settings/
│       ├── mod.rs         # settings panel
│       └── model.rs       # SettingsModel
├── update.rs              # Main update function
└── view.rs                # Main view composition
```

## Testing Nested Components

Each component's update and view functions can be tested independently:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_dialog_frequency_update() {
        let mut model = scan_dialog::Model { frequency: 88.0, ..Default::default() };
        let msg = scan_dialog::Message::FrequencyChanged(89.5);
        let updated = scan_dialog::update(model, msg);
        assert_eq!(updated.frequency, 89.5);
    }

    #[test]
    fn test_parent_routes_message() {
        let mut model = Model::default();
        let msg = Message::ScanDialog(scan_dialog::Message::FrequencyChanged(90.0));
        let updated = update(model, msg);
        assert_eq!(updated.scan_dialog.frequency, 90.0);
    }
}
```
