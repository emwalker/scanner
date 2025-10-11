# Plan 018: View Model Layer for Tuner State Testing

## Guidance for Updates

When updating this plan as work progresses, avoid adding:
- Lists of accomplishments or completion summaries
- Self-aggrandizement or subjective quality assessments
- Rationales and benefits sections (unless specifically requested)
- Speculation about future improvements or possibilities
- Time estimates or risk assessments

Keep updates matter-of-fact and focused on concrete technical details. Simply check off completed tasks and add technical notes as needed.

## Problem

The TUI can regress where tuner status labels don't accurately reflect tuner state. When scanning is underway, the SDR tuner should show it's scanning. When listening to a station, the tuner should show it's listening. The model state logic is already tested, but nothing verifies that the correct label text appears for each state.

## Proposal 1: View Model Functions

Add pure functions that transform Model state into display data structures, making the state-to-label mapping explicitly testable without coupling to visual rendering.

### Implementation

Create view model types and functions in the model layer:

```rust
pub struct TunerDisplayState {
    pub device_id: DeviceId,
    pub label: String,
    pub status_label: &'static str,
}

impl Model {
    pub fn tuner_display_states(&self) -> Vec<TunerDisplayState> {
        self.tuners.iter().map(|tuner| {
            let state = self.tuner_state(&tuner.id);
            TunerDisplayState {
                device_id: tuner.id.clone(),
                label: tuner.label.clone(),
                status_label: state.display(),
            }
        }).collect()
    }
}
```

### Tasks
- [x] Add `TunerDisplayState` struct to `src/ui/tui/model/types.rs`
- [x] Implement `Model::tuner_display_states()` in `src/ui/tui/model/queries.rs`
- [x] Add regression tests in `src/ui/tui/model/tests/tuner_state.rs`:
  - Test scanning tuner returns "Scanning" label
  - Test listening tuner returns "Listening" label
  - Test available tuner returns "Available" label
  - Test state transitions update labels correctly
  - Test multiple tuners show correct individual labels
- [x] Renderer already has good separation, no changes needed
- [x] All 256 library tests pass

## Proposal 2: Property-Based State Machine Testing

Use `proptest-state-machine` to generate random sequences of tuner state transitions and verify invariants hold across all possible event sequences.

### Implementation

Property-based state machine test that:
1. Generates sequences of 5-20 transitions (AddTuner, AllocateForScanning, AllocateForListening, ReturnToAvailable)
2. Maintains reference state tracking expected tuner activities
3. Applies transitions to Model via `TuiEvent::ActiveTunersUpdated` events
4. Checks invariants after each transition: tuner labels match expected activities

### Tasks
- [x] Add `proptest-state-machine = "0.4"` to `Cargo.toml` dev-dependencies
- [x] Create `src/ui/tui/model/tests/tuner_state_proptest.rs`
- [x] Define `TunerStateReference` implementing `ReferenceStateMachine`:
  - State: HashMap of device_id → RefActivity
  - Transitions: AddTuner, AllocateForScanning, AllocateForListening, ReturnToAvailable
  - Preconditions: prevent invalid transitions
- [x] Define `TunerStateMachineTest` implementing `StateMachineTest`:
  - SystemUnderTest: Model
  - init_test: create model from reference state
  - apply: update model via ActiveTunersUpdated events
  - check_invariants: verify labels match expected activities
- [x] Add test using `prop_state_machine!` macro with sequential 5..=20 transitions
- [x] All 257 library tests pass

---

## Research Findings

### Testing Patterns for TUI State

**Snapshot Testing with insta**
- Ratatui provides `TestBackend` for rendering UI to in-memory buffer
- `insta` crate compares rendered output against saved snapshot files
- Tests fail when visual output changes unexpectedly
- Use `cargo-insta` for reviewing and updating snapshots
- Pattern: render to TestBackend → serialize buffer → compare with snapshot
- Brittle during active UI development (deferred for this project)
- Reference: https://ratatui.rs/recipes/testing/snapshots/

**View Model Pattern (MVVM)**
- Separates display logic (ViewModel) from business logic (Model) and rendering (View)
- ViewModel transforms Model state into display-ready data structures
- Makes state-to-display mapping explicitly testable
- Tests verify data transformations without rendering
- UI can change freely as long as ViewModel contract holds
- Pattern from Microsoft's WPF/MVVM architecture, adapted for Rust TUIs
- Reference: https://learn.microsoft.com/en-us/dotnet/architecture/maui/mvvm

**Property-Based Testing**
- Generates random input sequences automatically
- Tests properties that should hold for all inputs
- Shrinks failing cases to minimal reproduction
- `proptest` is Rust's property-based testing library inspired by Hypothesis (Python) and QuickCheck (Haskell)
- Reference: https://proptest-rs.github.io/proptest/

**State Machine Testing**
- Models system as states and transitions
- Generates random transition sequences
- Maintains reference model for expected behavior
- Checks invariants after each transition
- `proptest-state-machine` provides `ReferenceStateMachine` and `StateMachineTest` traits
- Inspired by Erlang's `eqc_statem` (QuickCheck for Erlang)
- Pattern: Reference model tracks expected state → Apply same transitions to SUT → Check invariants
- Currently only supports sequential testing (concurrent testing planned)
- Reference: https://proptest-rs.github.io/proptest/proptest/state-machine.html

### The Elm Architecture Testing

TEA provides natural separation for testing:
- **Model layer** - Pure data structures, tested independently
- **Update functions** - Pure state transformations, tested by applying events
- **View functions** - Pure rendering, can use snapshot testing

Current tests verify model state transitions but miss the rendering contract. View model functions bridge this gap by making display logic explicit and testable without rendering.

Pattern from Elm language, widely used in TUI applications (Ratatui, Textual). Core principle: unidirectional data flow (Events → Update → Model → View) makes behavior predictable and testable.

### Relevant Tools

- `insta` (0.21+) - Snapshot testing with automatic review workflow
- `cargo-insta` - CLI tool for reviewing snapshot changes
- `proptest` (1.4+) - Property-based testing framework
- `proptest-state-machine` (0.4+) - State machine testing extension
- Ratatui `TestBackend` - In-memory terminal backend for testing
- `prop_state_machine!` macro - Declares and runs state machine tests

### Key Insights from Research

**Why View Models Work for TUIs**
- TEA already separates Model from View
- View model sits between: Model → ViewModel → Renderer
- ViewModel makes implicit rendering decisions explicit
- Tests verify "what should be displayed" without "how it's displayed"
- Aligns with TEA's pure function philosophy

**Property-Based Testing Benefits**
- Discovers edge cases not covered by hand-written tests
- Tests found issue in Erlang DETS (disk storage) after 20 years
- Effective for testing state machines, parsers, and data structures
- Shrinking makes debugging practical despite random generation

**State Machine Testing Best Practices**
- Keep reference state simpler than system under test
- Use preconditions to restrict invalid transitions
- Check invariants (properties that always hold) not exact values
- Start with sequential testing, add concurrent later if needed
- Model real system operations, not implementation details

**When to Use Each Pattern**
- Snapshot tests: Visual regression testing for stable UIs
- View models: Testing display logic during active development
- Property-based tests: Finding edge cases in state machines
- State machine tests: Complex stateful systems with many transitions
