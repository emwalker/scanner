# Elm Architecture Anti-Patterns

Understanding common pitfalls helps avoid architectural mistakes that become expensive to fix later.

## 1. Improper State Modeling for Optional Data

**Anti-pattern: Separate flags and data**

```rust
pub struct Model {
    is_loading: bool,
    results: Vec<Station>,
    error_message: Option<String>,
}

pub fn view(model: &Model) -> String {
    if model.error_message.is_some() {
        // Show error
    } else if model.is_loading {
        // Show loading spinner
    } else if model.results.is_empty() {
        // Show "No Results Found"
    } else {
        // Show results
    }
}
```

**Problem:** This state representation allows invalid combinations:
- `is_loading: true, results: [data]` (contradictory)
- `is_loading: true, error_message: Some("...")` (what to show?)
- Forgetting to check `is_loading` causes UI to show "No Results Found" on slow connections

**Better approach: Use enums to represent state**

```rust
pub enum LoadState {
    Empty,
    Loading,
    Success(Vec<Station>),
    Failed(String),
}

pub struct Model {
    scan_state: LoadState,
}

pub fn view(model: &Model) -> String {
    match model.scan_state {
        LoadState::Empty => "Press scan to begin".to_string(),
        LoadState::Loading => "Scanning...".to_string(),
        LoadState::Success(ref results) => render_results(results),
        LoadState::Failed(ref error) => format!("Error: {}", error),
    }
}
```

**Benefits:**
- Compiler enforces handling all cases
- Invalid states become impossible to represent
- No forgotten checks

## 2. Mixed Concerns in Update Function

**Anti-pattern: Business logic mixed with side effects**

```rust
pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::StartScan => {
            // Business logic
            model.is_scanning = true;
            model.current_frequency = 88.0;

            // Side effect mixed in (BAD)
            let results = perform_scan_blocking(88.0);
            model.results = results;

            // Another side effect
            log_to_file(&format!("Scan completed: {} stations", results.len()));
            model
        }
    }
}
```

**Problems:**
- Impossible to test without actually performing I/O
- Blocking operations freeze the UI
- Difficult to handle async operations
- Side effects scattered throughout code

**Better approach: Separate concerns**

```rust
pub enum Message {
    ScanRequested(f64),
    ScanCompleted(Vec<Station>),
    ScanFailed(String),
}

pub enum Command {
    PerformScan(f64),
    LogEvent(String),
    SaveResults(Vec<Station>),
}

pub fn update(mut model: Model, msg: Message) -> (Model, Vec<Command>) {
    match msg {
        Message::ScanRequested(freq) => {
            model.is_scanning = true;
            model.current_frequency = freq;
            (model, vec![Command::PerformScan(freq)])
        }
        Message::ScanCompleted(results) => {
            model.is_scanning = false;
            model.results = results.clone();
            let log_cmd = Command::LogEvent(
                format!("Scan completed: {} stations", results.len())
            );
            (model, vec![log_cmd, Command::SaveResults(results)])
        }
        Message::ScanFailed(error) => {
            model.is_scanning = false;
            model.error = Some(error);
            (model, vec![])
        }
    }
}
```

**Benefits:**
- Update function is pure and testable
- Side effects handled separately
- Async operations can be properly managed
- Clear separation of concerns

## 3. Implicit Dependencies Between Components

**Anti-pattern: Hidden coupling**

```rust
pub struct Model {
    scan_dialog: ScanDialogModel,
    results: ResultsModel,
}

// scan_dialog mysteriously needs to know about results
pub fn scan_dialog_update(mut dialog: ScanDialogModel, msg: Message) -> ScanDialogModel {
    match msg {
        Message::StartScan => {
            // How does the dialog know if scan succeeded?
            // Implicit dependency on results being updated elsewhere
            dialog.is_active = false;
            dialog
        }
    }
}
```

**Problem:** Dependencies are unclear, making code hard to understand and maintain.

**Better approach: Make dependencies explicit**

```rust
pub enum Message {
    DialogRequested,
    ScanDialogMessage(ScanDialogMessage),
    ScanCompleted(Vec<Station>),
}

pub fn update(mut model: Model, msg: Message) -> Model {
    match msg {
        Message::ScanCompleted(results) => {
            model.results.data = results;
            // Explicitly update dialog state
            model.scan_dialog.is_active = false;
            model
        }
        Message::ScanDialogMessage(m) => {
            model.scan_dialog = scan_dialog::update(model.scan_dialog, m);
            model
        }
    }
}
```

## 4. Unstructured Message Dispatch

**Anti-pattern: Vague or generic messages**

```rust
pub enum Message {
    UpdateUI,
    RefreshData,
    ProcessEvent,
    HandleClick,
}
```

**Problems:**
- Unclear what triggered the message
- Multiple things could produce the same message
- Update function can't make intelligent decisions
- Difficult to trace state changes

**Better approach: Descriptive, specific messages**

```rust
pub enum Message {
    UserSelectedStation(StationId),
    FrequencyInputChanged(String),
    ScanStarted { frequency: f64, duration_ms: u32 },
    ScanCompleted(Result<Vec<Station>, ScanError>),
    ViewModeChanged(ViewMode),
    ErrorDismissed,
}
```

## 5. Putting Too Much in Model

**Anti-pattern: Model as a dumping ground**

```rust
pub struct Model {
    pub data: Vec<Item>,
    pub selected_index: usize,
    pub scroll_position: usize,
    pub search_query: String,
    pub filter_by_name: bool,
    pub filter_by_date: bool,
    pub sort_ascending: bool,
    pub theme: Theme,
    pub font_size: u32,
    pub is_fullscreen: bool,
    pub sidebar_width: u32,
    pub last_scroll_time: u64,
    pub is_animating: bool,
    // ... 20 more fields
}
```

**Problems:**
- Difficult to understand what state is important
- Updates become complex to reason about
- Hard to test (too many fields)
- Unnecessary state often lingers

**Better approach: Organize hierarchically**

```rust
pub struct Model {
    pub view_state: ViewState,
    pub ui_preferences: UiPreferences,
    pub data_display: DataDisplay,
}

pub struct ViewState {
    pub current_view: CurrentView,
    pub search_query: String,
}

pub struct UiPreferences {
    pub theme: Theme,
    pub font_size: u32,
    pub sidebar_width: u32,
}

pub struct DataDisplay {
    pub data: Vec<Item>,
    pub selected_index: usize,
    pub scroll_position: usize,
}
```

## 6. View Functions With Side Effects

**Anti-pattern: View that does I/O or state mutation**

```rust
pub fn view(model: &Model) -> Widget {
    // Reading from global state (BAD)
    let config = GLOBAL_CONFIG.lock().unwrap();

    // Performing I/O (BAD)
    let fresh_data = fetch_from_server();

    // Mutating state (BAD)
    if should_auto_refresh() {
        REFRESH_COUNTER.store(REFRESH_COUNTER.load() + 1, Ordering::SeqCst);
    }

    // After all that, finally render
    render_ui(&model, &fresh_data)
}
```

**Problems:**
- View is non-deterministic (same input, different output)
- Difficult to test
- Performance issues (fetching data on every render)
- State changes hidden inside view

**Better approach: Pure view function**

```rust
pub fn view(model: &Model) -> Widget {
    // Only use the model parameter
    // No I/O, no mutations, no external state
    match model.state {
        State::Loading => render_loading_spinner(),
        State::Success(ref data) => render_data(data),
        State::Failed(ref error) => render_error(error),
    }
}
```

## 7. Forgetting That View Must Be Idempotent

**Anti-pattern: View has different output depending on when it's called**

```rust
pub fn view(model: &Model) -> Widget {
    static mut CALL_COUNT: u32 = 0;
    unsafe { CALL_COUNT += 1 }

    // Same input produces different output!
    if unsafe { CALL_COUNT % 2 == 0 } {
        render_with_animation(model)
    } else {
        render_static(model)
    }
}
```

**Better approach:**

```rust
pub fn view(model: &Model) -> Widget {
    // Given this model state, always produce same output
    render(model)
}

// If animation state is needed, it belongs in the Model
pub struct Model {
    pub animation_frame: u32,
}
```

## Summary: Red Flags

Watch for these patterns in code reviews:

- Separate boolean flags instead of enum state
- Side effects inside update or view functions
- Hidden dependencies between components
- Generic/vague message names
- View functions that do I/O
- Large monolithic Model structs
- Updates that behave differently on successive calls with same input
