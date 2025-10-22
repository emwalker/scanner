# Global Pause Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement spacebar-triggered global pause that stops all scanning and audio processing, reduces CPU to near-zero, and allows resumption to previous state.

**Architecture:** Add GlobalPauseResource (ECS Resource) as single source of truth for pause state. Extend ScanPauseState with PausedGlobally variant that captures previous state for resume. TUI handles spacebar key, updates resource, and coordinates pause/resume across all scan and audio entities. Signal processing threads are cancelled and joined (not suspended) to achieve zero CPU usage.

**Tech Stack:** Rust, ECS pattern, Arc<Mutex<T>> for resources, CancellationToken for thread coordination, ratatui for TUI

---

## Task 1: Create GlobalPauseResource

**Files:**
- Create: `src/ecs/resources/mod.rs`
- Create: `src/ecs/resources/global_pause.rs`
- Modify: `src/ecs/mod.rs`

**Step 1: Write test for GlobalPauseState transitions**

```rust
// src/ecs/resources/global_pause.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_pause_state_active_to_paused() {
        let state = GlobalPauseState::Active;
        assert!(matches!(state, GlobalPauseState::Active));

        let paused = GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: false,
        };
        assert!(matches!(
            paused,
            GlobalPauseState::Paused {
                had_active_scans: true,
                had_active_audio: false
            }
        ));
    }

    #[test]
    fn test_global_pause_state_paused_to_active() {
        let state = GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: true,
        };
        let active = GlobalPauseState::Active;
        assert!(matches!(active, GlobalPauseState::Active));
    }

    #[test]
    fn test_global_pause_resource_creation() {
        let resource = GlobalPauseResource::new(GlobalPauseState::Active);
        let state = resource.lock().unwrap();
        assert!(matches!(*state, GlobalPauseState::Active));
    }

    #[test]
    fn test_global_pause_resource_mutation() {
        let resource = GlobalPauseResource::new(GlobalPauseState::Active);

        {
            let mut state = resource.lock().unwrap();
            *state = GlobalPauseState::Paused {
                had_active_scans: false,
                had_active_audio: true,
            };
        }

        let state = resource.lock().unwrap();
        assert!(matches!(
            *state,
            GlobalPauseState::Paused {
                had_active_scans: false,
                had_active_audio: true
            }
        ));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib ecs::resources::global_pause::tests`
Expected: FAIL (module does not exist)

**Step 3: Create resources module structure**

```rust
// src/ecs/resources/mod.rs
//! ECS Resources - global shared state

pub mod global_pause;

pub use global_pause::{GlobalPauseResource, GlobalPauseState};
```

```rust
// src/ecs/resources/global_pause.rs
//! Global pause state resource

use std::sync::{Arc, Mutex};

/// Global pause state for the entire application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalPauseState {
    /// Application is active (scanning, audio playing)
    Active,
    /// Application is globally paused
    Paused {
        /// Whether there were active scans before pausing
        had_active_scans: bool,
        /// Whether audio was playing before pausing
        had_active_audio: bool,
    },
}

/// Resource type for global pause state (thread-safe)
pub type GlobalPauseResource = Arc<Mutex<GlobalPauseState>>;
```

**Step 4: Export resources from ecs module**

```rust
// src/ecs/mod.rs (add after existing module declarations)
pub mod resources;

// Add to the pub use section
pub use resources::{GlobalPauseResource, GlobalPauseState};
```

**Step 5: Run tests to verify they pass**

Run: `cargo test --lib ecs::resources::global_pause::tests`
Expected: PASS

**Step 6: Commit**

```bash
git add src/ecs/resources/mod.rs src/ecs/resources/global_pause.rs src/ecs/mod.rs
git commit -m "feat: add GlobalPauseResource for global pause state

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Extend ScanPauseState with PausedGlobally

**Files:**
- Modify: `src/ecs/components/scan/progress.rs`
- Test: Run existing tests to ensure no breakage

**Step 1: Write tests for new PausedGlobally state**

```rust
// Add to src/ecs/components/scan/progress.rs tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_globally_from_scanning() {
        let mut progress = ScanProgressComponent::new(5);
        progress.start_window(2);
        assert!(progress.is_scanning());

        progress.pause_globally(2, PreviousPauseState::WasScanning);
        assert!(matches!(
            progress.state,
            ScanPauseState::PausedGlobally { .. }
        ));
        assert!(!progress.is_scanning());
    }

    #[test]
    fn test_pause_globally_from_listening() {
        let mut progress = ScanProgressComponent::new(5);
        progress.start_listening(3);
        assert!(progress.is_listening());

        progress.pause_globally(
            3,
            PreviousPauseState::WasListening {
                window_num: 3,
                station_frequency_hz: 88.9e6,
            },
        );
        assert!(matches!(
            progress.state,
            ScanPauseState::PausedGlobally { .. }
        ));
        assert!(!progress.is_listening());
    }

    #[test]
    fn test_resume_from_globally_paused_to_scanning() {
        let mut progress = ScanProgressComponent::new(5);
        progress.pause_globally(2, PreviousPauseState::WasScanning);

        progress.resume_from_global_pause();
        assert!(progress.is_scanning());
    }

    #[test]
    fn test_resume_from_globally_paused_to_listening() {
        let mut progress = ScanProgressComponent::new(5);
        progress.pause_globally(
            3,
            PreviousPauseState::WasListening {
                window_num: 3,
                station_frequency_hz: 88.9e6,
            },
        );

        progress.resume_from_global_pause();
        assert!(progress.is_listening());
        if let ScanPauseState::Listening { paused_at_window } = progress.state {
            assert_eq!(paused_at_window, 3);
        } else {
            panic!("Expected Listening state");
        }
    }

    #[test]
    fn test_is_globally_paused() {
        let mut progress = ScanProgressComponent::new(5);
        assert!(!progress.is_globally_paused());

        progress.pause_globally(2, PreviousPauseState::WasScanning);
        assert!(progress.is_globally_paused());

        progress.resume_from_global_pause();
        assert!(!progress.is_globally_paused());
    }
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test --lib ecs::components::scan::progress::tests`
Expected: FAIL (methods and enum variants don't exist)

**Step 3: Add PreviousPauseState enum**

```rust
// src/ecs/components/scan/progress.rs (add after ScanPauseState)

/// Captures what was happening before global pause for resume
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreviousPauseState {
    /// Was actively scanning
    WasScanning,
    /// Was listening to a station
    WasListening {
        window_num: usize,
        station_frequency_hz: f64,
    },
}
```

**Step 4: Add PausedGlobally variant to ScanPauseState**

```rust
// src/ecs/components/scan/progress.rs (modify ScanPauseState enum)
/// Pause state for a scan
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanPauseState {
    /// Scan has been requested but not yet started
    Pending,
    /// Scan is actively running
    Scanning,
    /// Scan is paused at a specific window
    PausedAtWindow { window_index: usize },
    /// Globally paused by user (spacebar)
    PausedGlobally {
        at_window: usize,
        previous_state: PreviousPauseState,
    },
    /// Scan has completed all windows
    Completed,
    /// User is listening to a station (paused for audio)
    Listening { paused_at_window: usize },
}
```

**Step 5: Add methods to ScanProgressComponent**

```rust
// src/ecs/components/scan/progress.rs (add to impl ScanProgressComponent)

    /// Pause globally (user-initiated via spacebar)
    pub fn pause_globally(&mut self, window_index: usize, previous_state: PreviousPauseState) {
        self.state = ScanPauseState::PausedGlobally {
            at_window: window_index,
            previous_state,
        };
    }

    /// Resume from global pause, restoring previous state
    pub fn resume_from_global_pause(&mut self) {
        if let ScanPauseState::PausedGlobally {
            at_window,
            previous_state,
        } = self.state
        {
            match previous_state {
                PreviousPauseState::WasScanning => {
                    self.state = ScanPauseState::Scanning;
                }
                PreviousPauseState::WasListening { window_num, .. } => {
                    self.state = ScanPauseState::Listening {
                        paused_at_window: window_num,
                    };
                }
            }
        }
    }

    /// Check if globally paused
    pub fn is_globally_paused(&self) -> bool {
        matches!(self.state, ScanPauseState::PausedGlobally { .. })
    }
```

**Step 6: Update is_paused to include PausedGlobally**

```rust
// src/ecs/components/scan/progress.rs (modify is_paused method)
    /// Check if scan is paused
    pub fn is_paused(&self) -> bool {
        matches!(
            self.state,
            ScanPauseState::PausedAtWindow { .. }
                | ScanPauseState::PausedGlobally { .. }
                | ScanPauseState::Listening { .. }
        )
    }
```

**Step 7: Run tests to verify they pass**

Run: `cargo test --lib ecs::components::scan::progress::tests`
Expected: PASS

**Step 8: Run all tests to check for breakage**

Run: `cargo test --lib`
Expected: PASS (existing tests still work)

**Step 9: Commit**

```bash
git add src/ecs/components/scan/progress.rs
git commit -m "feat: add PausedGlobally variant to ScanPauseState

Add PreviousPauseState to capture what was happening before pause
for accurate resume. Update is_paused() to include new variant.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Integrate GlobalPauseResource in Coordinator

**Files:**
- Modify: `src/ecs/coordinator.rs`
- Modify: `src/ecs/system.rs`

**Step 1: Write test for coordinator with global pause resource**

```rust
// Add to src/ecs/coordinator.rs tests
#[test]
fn test_coordinator_with_global_pause_resource() {
    let pool = Arc::new(Pool::new(PoolFilter::allow_all(), None));
    let config = Arc::new(ScanningConfig::default());
    let shutdown = Arc::new(ShutdownCoordinator::new());

    let global_pause = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
    let coordinator = Coordinator::new(&pool, &config, &shutdown)
        .with_global_pause_resource(global_pause);

    let retrieved = coordinator.global_pause_resource();
    let state = retrieved.lock().unwrap();
    assert!(matches!(*state, crate::ecs::GlobalPauseState::Active));
}
```

**Step 2: Run test to verify failure**

Run: `cargo test --lib ecs::coordinator::tests::test_coordinator_with_global_pause_resource`
Expected: FAIL (method doesn't exist)

**Step 3: Add global_pause_resource field to Coordinator**

```rust
// src/ecs/coordinator.rs (add to Coordinator struct)
pub struct Coordinator {
    scheduler: Scheduler,
    tuner_entities: Arc<Mutex<EntityWorld<TunerEntity>>>,
    scan_entities: Option<Entities<ScanEntity>>,
    window_entities: Option<Entities<WindowEntity>>,
    station_entities: Option<Entities<StationEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    candidate_entities: Option<Entities<CandidateEntity>>,

    audio_streams: Resource<HashMap<AudioId, cpal::Stream>>,
    audio_segments: Resource<HashMap<AudioId, Segment>>,
    tuner_request_queue: Resource<TunerRequestQueue>,
    pause_request_queue: Resource<PauseRequestQueue>,
    global_pause_resource: crate::ecs::GlobalPauseResource,  // NEW

    pool: Arc<Pool>,
    config: Arc<ScanningConfig>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}
```

**Step 4: Update Coordinator::new to initialize resource**

```rust
// src/ecs/coordinator.rs (modify new method)
    pub fn new(
        pool: &Arc<Pool>,
        config: &Arc<ScanningConfig>,
        shutdown_coordinator: &Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            scheduler: Scheduler::new(),
            tuner_entities: Arc::clone(&pool.tuner_entities),
            scan_entities: None,
            window_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
            #[allow(clippy::arc_with_non_send_sync)]
            audio_streams: Arc::new(Mutex::new(HashMap::new())),
            #[allow(clippy::arc_with_non_send_sync)]
            audio_segments: Arc::new(Mutex::new(HashMap::new())),
            tuner_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            pause_request_queue: Arc::new(Mutex::new(VecDeque::new())),
            global_pause_resource: Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active)),
            pool: Arc::clone(pool),
            config: Arc::clone(config),
            shutdown_coordinator: Arc::clone(shutdown_coordinator),
        }
    }
```

**Step 5: Add builder method and accessor**

```rust
// src/ecs/coordinator.rs (add methods)
    /// Set the global pause resource (replaces the default Active state)
    pub fn with_global_pause_resource(
        mut self,
        resource: crate::ecs::GlobalPauseResource,
    ) -> Self {
        self.global_pause_resource = resource;
        self
    }

    /// Get a clone of the global pause resource for external access (e.g., TUI)
    pub fn global_pause_resource(&self) -> crate::ecs::GlobalPauseResource {
        Arc::clone(&self.global_pause_resource)
    }
```

**Step 6: Pass resource to SystemContext in tick()**

```rust
// src/ecs/coordinator.rs (modify tick method)
    pub fn tick(&mut self) -> Result<()> {
        let mut context = SystemContext::new()
            .with_tuner_entities(Arc::clone(&self.tuner_entities))
            .with_audio_streams(Arc::clone(&self.audio_streams))
            .with_audio_segments(Arc::clone(&self.audio_segments))
            .with_tuner_request_queue(Arc::clone(&self.tuner_request_queue))
            .with_pause_request_queue(Arc::clone(&self.pause_request_queue))
            .with_global_pause_resource(Arc::clone(&self.global_pause_resource))  // NEW
            .with_pool(Arc::clone(&self.pool))
            .with_config(Arc::clone(&self.config))
            .with_shutdown_coordinator(Arc::clone(&self.shutdown_coordinator));

        // ... rest of method unchanged
```

**Step 7: Add field to SystemContext**

```rust
// src/ecs/system.rs (add to SystemContext struct)
pub struct SystemContext {
    pub tuner_entities: Option<Arc<Mutex<EntityWorld<TunerEntity>>>>,
    pub scan_entities: Option<Entities<ScanEntity>>,
    pub window_entities: Option<Entities<WindowEntity>>,
    pub station_entities: Option<Entities<StationEntity>>,
    pub audio_entities: Option<Entities<AudioEntity>>,
    pub candidate_entities: Option<Entities<CandidateEntity>>,

    pub audio_streams: Option<Resource<HashMap<AudioId, cpal::Stream>>>,
    pub audio_segments: Option<Resource<HashMap<AudioId, Segment>>>,
    pub tuner_request_queue: Option<Resource<TunerRequestQueue>>,
    pub pause_request_queue: Option<Resource<PauseRequestQueue>>,
    pub global_pause_resource: Option<crate::ecs::GlobalPauseResource>,  // NEW

    pub pool: Option<Arc<Pool>>,
    pub config: Option<Arc<ScanningConfig>>,
    pub shutdown_coordinator: Option<Arc<ShutdownCoordinator>>,
}
```

**Step 8: Initialize field in SystemContext::new**

```rust
// src/ecs/system.rs (modify new method)
    pub fn new() -> Self {
        Self {
            tuner_entities: None,
            scan_entities: None,
            window_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
            audio_streams: None,
            audio_segments: None,
            tuner_request_queue: None,
            pause_request_queue: None,
            global_pause_resource: None,  // NEW
            pool: None,
            config: None,
            shutdown_coordinator: None,
        }
    }
```

**Step 9: Add builder method to SystemContext**

```rust
// src/ecs/system.rs (add method)
    pub fn with_global_pause_resource(
        mut self,
        resource: crate::ecs::GlobalPauseResource,
    ) -> Self {
        self.global_pause_resource = Some(resource);
        self
    }
```

**Step 10: Run tests**

Run: `cargo test --lib ecs::coordinator::tests::test_coordinator_with_global_pause_resource`
Expected: PASS

Run: `cargo test --lib ecs::coordinator`
Expected: PASS (all coordinator tests)

**Step 11: Commit**

```bash
git add src/ecs/coordinator.rs src/ecs/system.rs
git commit -m "feat: integrate GlobalPauseResource in Coordinator

Add global_pause_resource field to Coordinator and SystemContext.
Resource is created by default in Active state and can be replaced
via builder. Passed to SystemContext on each tick.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Add is_globally_paused to Model

**Files:**
- Modify: `src/ui/tui/model/mod.rs`

**Step 1: Write test for is_globally_paused**

```rust
// Add to src/ui/tui/model/mod.rs (or create tests/mod.rs if preferred)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_globally_paused_without_resource() {
        let model = Model::new();
        assert!(!model.is_globally_paused());
    }

    #[test]
    fn test_is_globally_paused_active() {
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
        let mut model = Model::new();
        model.set_global_pause_resource(resource);
        assert!(!model.is_globally_paused());
    }

    #[test]
    fn test_is_globally_paused_paused() {
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: false,
        }));
        let mut model = Model::new();
        model.set_global_pause_resource(resource);
        assert!(model.is_globally_paused());
    }
}
```

**Step 2: Run test to verify failure**

Run: `cargo test --lib ui::tui::model::tests`
Expected: FAIL (methods don't exist)

**Step 3: Find Model struct definition**

Read: `src/ui/tui/model/mod.rs` to locate Model struct

**Step 4: Add global_pause_resource field to Model**

```rust
// src/ui/tui/model/mod.rs (add to Model struct)
pub struct Model {
    // ... existing fields ...
    pub global_pause_resource: Option<crate::ecs::GlobalPauseResource>,
}
```

**Step 5: Initialize field in Model::new**

```rust
// src/ui/tui/model/mod.rs (modify new method)
    pub fn new() -> Self {
        Self {
            // ... existing initializations ...
            global_pause_resource: None,
        }
    }
```

**Step 6: Add setter and query methods**

```rust
// src/ui/tui/model/mod.rs (add methods to impl Model)
    /// Set the global pause resource reference
    pub fn set_global_pause_resource(&mut self, resource: crate::ecs::GlobalPauseResource) {
        self.global_pause_resource = Some(resource);
    }

    /// Check if globally paused
    pub fn is_globally_paused(&self) -> bool {
        if let Some(ref resource) = self.global_pause_resource {
            if let Ok(state) = resource.lock() {
                return matches!(*state, crate::ecs::GlobalPauseState::Paused { .. });
            }
        }
        false
    }
```

**Step 7: Run tests**

Run: `cargo test --lib ui::tui::model::tests`
Expected: PASS

**Step 8: Commit**

```bash
git add src/ui/tui/model/mod.rs
git commit -m "feat: add is_globally_paused query to Model

Add global_pause_resource field and query method to Model.
Returns false if resource not set or lock fails.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Add pause indicator to status bar

**Files:**
- Modify: `src/ui/tui/renderers/instructions.rs`
- Test: Manual visual verification

**Step 1: Write test for pause indicator rendering**

```rust
// Add to src/ui/tui/renderers/instructions.rs tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::model::Model;
    use crate::ui::tui::themes::{ColorScheme, Theme};
    use ratatui::style::Color;
    use std::sync::{Arc, Mutex};

    struct MockTheme;
    impl Theme for MockTheme {
        fn instructions_dim(&self) -> Color {
            Color::Gray
        }
        fn active_highlight_fg(&self) -> Color {
            Color::Green
        }
    }
    impl ColorScheme for MockTheme {}

    #[test]
    fn test_pause_indicator_not_shown_when_active() {
        let mut model = Model::new();
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
        model.set_global_pause_resource(resource);

        let left_text = build_left_instructions_text(&model);
        assert!(!left_text.contains("[PAUSED]"));
    }

    #[test]
    fn test_pause_indicator_shown_when_paused() {
        let mut model = Model::new();
        let resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: false,
        }));
        model.set_global_pause_resource(resource);

        let left_text = build_left_instructions_text(&model);
        assert!(left_text.contains("[PAUSED]"));
    }
}
```

**Step 2: Run test to verify failure**

Run: `cargo test --lib ui::tui::renderers::instructions::tests`
Expected: FAIL (build_left_instructions_text doesn't exist)

**Step 3: Extract left instructions text to helper function**

```rust
// src/ui/tui/renderers/instructions.rs (add helper function)
fn build_left_instructions_text(model: &Model) -> String {
    let pause_prefix = if model.is_globally_paused() {
        "[PAUSED] "
    } else {
        ""
    };

    let instructions = match &model.ui_mode {
        UiMode::Listening { .. } if !model.all_complete() => {
            "  ⌃C Exit  ↑↓ Browse  ↵ Continue scan"
        }
        UiMode::AwaitingTune { .. } if !model.all_complete() => {
            "  ⌃C Exit  ↑↓ Browse  ↵ Continue scan"
        }
        UiMode::NavigatingScanner { .. } => "  ⌃C Exit  ↑↓ Navigate  ↵ Listen",
        _ => "  ⌃C Exit  ↑↓ Navigate",
    };

    format!("{}{}", pause_prefix, instructions)
}
```

**Step 4: Update render_instructions to use helper**

```rust
// src/ui/tui/renderers/instructions.rs (modify render_instructions)
pub fn render_instructions(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    theme: &dyn Theme,
    theme_name: &str,
    model: &Model,
    all_themes: &[String],
) {
    if model.theme_selector_open {
        render_theme_selector(f, area, theme, model, all_themes);
    } else {
        let left_text = build_left_instructions_text(model);

        let mut spans = vec![];
        if model.is_globally_paused() {
            spans.push(Span::styled(
                "[PAUSED] ",
                Style::default()
                    .fg(theme.active_highlight_fg())
                    .add_modifier(Modifier::BOLD),
            ));
        }

        spans.push(Span::styled(
            &left_text[if model.is_globally_paused() { 9 } else { 0 }..],
            Style::default().fg(theme.instructions_dim()),
        ));

        let instruction = Paragraph::new(Line::from(spans));
        f.render_widget(instruction, area);

        let right_text = match &model.ui_mode {
            UiMode::Listening { .. } => {
                if let Some(info) = model.selected_candidate_info() {
                    format!("[Listening: {:.1} MHz]  ", info.candidate_frequency / 1e6)
                } else {
                    format!("{}  ", theme_name)
                }
            }
            _ => format!("{}  ", theme_name),
        };

        let theme_display = Paragraph::new(right_text)
            .alignment(Alignment::Right)
            .style(Style::default().fg(theme.instructions_dim()));
        f.render_widget(theme_display, area);
    }
}
```

**Step 5: Run tests**

Run: `cargo test --lib ui::tui::renderers::instructions::tests`
Expected: PASS

**Step 6: Build to check for compilation errors**

Run: `cargo build`
Expected: SUCCESS

**Step 7: Commit**

```bash
git add src/ui/tui/renderers/instructions.rs
git commit -m "feat: add [PAUSED] indicator to status bar

Show bold green [PAUSED] prefix in bottom status bar when
globally paused. Extract instruction text building to helper
function for testability.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Add spacebar key handling in TUI

**Files:**
- Modify: `src/ui/tui/mod.rs`

**Step 1: Locate keyboard input handling section**

Read: `src/ui/tui/mod.rs` lines 610-660 to understand key handling

**Step 2: Add spacebar handler between theme selector and navigation**

```rust
// src/ui/tui/mod.rs (add after theme selector handling, before navigation)
                    // Handle spacebar for global pause/resume
                    if matches!(key.code, KeyCode::Char(' ')) {
                        self.handle_spacebar_pause();
                        return Ok(false);
                    }

                    // Handle navigation
                    if matches!(
                        key.code,
                        KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right
                    ) {
                        self.handle_navigation_keys(key.code);
                        return Ok(false);
                    }
```

**Step 3: Implement handle_spacebar_pause method**

```rust
// src/ui/tui/mod.rs (add method to impl TuiProgressDisplay)
    /// Handle spacebar key for global pause/resume
    fn handle_spacebar_pause(&mut self) {
        // Toggle global pause state
        if let Some(ref resource) = self.model.global_pause_resource {
            if let Ok(mut state) = resource.lock() {
                match *state {
                    crate::ecs::GlobalPauseState::Active => {
                        debug!("TUI: Spacebar pressed - pausing globally");

                        let had_active_scans = self.has_active_scans();
                        let had_active_audio = self.has_active_audio();

                        *state = crate::ecs::GlobalPauseState::Paused {
                            had_active_scans,
                            had_active_audio,
                        };

                        self.pause_all_scans();
                        self.pause_all_audio();
                    }
                    crate::ecs::GlobalPauseState::Paused { .. } => {
                        debug!("TUI: Spacebar pressed - resuming from global pause");

                        *state = crate::ecs::GlobalPauseState::Active;

                        self.resume_all_scans();
                        self.resume_all_audio();
                    }
                }
            }
        }
    }
```

**Step 4: Implement has_active_scans helper**

```rust
// src/ui/tui/mod.rs (add method)
    /// Check if there are any active scans
    fn has_active_scans(&self) -> bool {
        if let Some(ref scan_entities) = self.scan_entities {
            if let Ok(entities) = scan_entities.try_read() {
                return entities.iter().any(|scan| scan.is_scanning());
            }
        }
        false
    }
```

**Step 5: Implement has_active_audio helper**

```rust
// src/ui/tui/mod.rs (add method)
    /// Check if there is any active audio
    fn has_active_audio(&self) -> bool {
        if let Some(ref audio_entities) = self.audio_entities {
            if let Ok(entities) = audio_entities.try_read() {
                return entities.iter().any(|audio| audio.is_playing());
            }
        }
        false
    }
```

**Step 6: Implement pause_all_scans**

```rust
// src/ui/tui/mod.rs (add method)
    /// Pause all active scans globally
    fn pause_all_scans(&mut self) {
        if let Some(ref scan_entities) = self.scan_entities {
            if let Ok(mut entities) = scan_entities.try_write() {
                for scan in entities.iter_mut() {
                    let current_window = scan.current_window();

                    let previous_state = if scan.is_scanning() {
                        crate::ecs::components::scan::progress::PreviousPauseState::WasScanning
                    } else if scan.is_listening() {
                        if let Some(station) = self.get_listening_station(scan.id()) {
                            crate::ecs::components::scan::progress::PreviousPauseState::WasListening {
                                window_num: current_window,
                                station_frequency_hz: station.frequency,
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    scan.pause_globally(current_window, previous_state);
                    debug!(
                        scan_id = ?scan.id(),
                        window = current_window,
                        "TUI: Paused scan globally"
                    );
                }
            }
        }
    }
```

**Step 7: Implement pause_all_audio**

```rust
// src/ui/tui/mod.rs (add method)
    /// Pause all active audio
    fn pause_all_audio(&mut self) {
        if let Some(ref audio_entities) = self.audio_entities {
            if let Ok(mut entities) = audio_entities.try_write() {
                for audio in entities.iter_mut() {
                    if audio.is_playing() {
                        audio.request_stop_listening();
                        debug!(
                            audio_id = ?audio.id(),
                            frequency_hz = audio.frequency(),
                            "TUI: Requested stop for audio entity"
                        );
                    }
                }
            }
        }
    }
```

**Step 8: Implement resume_all_scans**

```rust
// src/ui/tui/mod.rs (add method)
    /// Resume all globally paused scans
    fn resume_all_scans(&mut self) {
        if let Some(ref scan_entities) = self.scan_entities {
            if let Ok(mut entities) = scan_entities.try_write() {
                for scan in entities.iter_mut() {
                    if scan.is_globally_paused() {
                        scan.resume_from_global_pause();
                        debug!(
                            scan_id = ?scan.id(),
                            "TUI: Resumed scan from global pause"
                        );
                    }
                }
            }
        }
    }
```

**Step 9: Implement resume_all_audio**

```rust
// src/ui/tui/mod.rs (add method)
    /// Resume all audio that was playing before global pause
    fn resume_all_audio(&mut self) {
        if let Some(ref scan_entities) = self.scan_entities {
            if let Ok(entities) = scan_entities.try_read() {
                for scan in entities.iter() {
                    if let crate::ecs::components::scan::progress::ScanPauseState::Listening {
                        paused_at_window,
                    } = scan.progress().state
                    {
                        if let Some(station) = self.get_listening_station(scan.id()) {
                            debug!(
                                scan_id = ?scan.id(),
                                frequency_hz = station.frequency,
                                "TUI: Resuming audio playback"
                            );
                            // Audio will resume via normal listening mode logic
                        }
                    }
                }
            }
        }
    }
```

**Step 10: Implement get_listening_station helper**

```rust
// src/ui/tui/mod.rs (add method)
    /// Get the station being listened to for a scan
    fn get_listening_station(&self, _scan_id: crate::ecs::ScanId) -> Option<StationInfo> {
        // Simplified for now - in full implementation would look up actual station
        // from selected candidate info
        if let Some(info) = self.model.selected_candidate_info() {
            return Some(StationInfo {
                frequency: info.candidate_frequency,
            });
        }
        None
    }
```

**Step 11: Add StationInfo struct**

```rust
// src/ui/tui/mod.rs (add near top of file)
struct StationInfo {
    frequency: f64,
}
```

**Step 12: Wire up resource in TUI initialization**

```rust
// src/ui/tui/mod.rs (add method to TuiProgressDisplay)
    pub fn with_global_pause_resource(
        mut self,
        resource: crate::ecs::GlobalPauseResource,
    ) -> Self {
        self.model.set_global_pause_resource(Arc::clone(&resource));
        self
    }
```

**Step 13: Build to check for compilation errors**

Run: `cargo build`
Expected: SUCCESS (may have warnings about unused methods)

**Step 14: Commit**

```bash
git add src/ui/tui/mod.rs
git commit -m "feat: add spacebar key handling for global pause

Implement handle_spacebar_pause that toggles GlobalPauseResource
and coordinates pause/resume across all scan and audio entities.
Add helper methods for checking active state and controlling
entity transitions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Wire up GlobalPauseResource in main loop

**Files:**
- Modify: `src/main.rs` (or wherever coordinator is initialized)

**Step 1: Find coordinator initialization code**

Search: `Coordinator::new` in codebase to find initialization location

**Step 2: Create GlobalPauseResource before coordinator**

```rust
// In main.rs or wherever coordinator is created
let global_pause_resource = Arc::new(Mutex::new(crate::ecs::GlobalPauseState::Active));
```

**Step 3: Pass resource to coordinator**

```rust
let coordinator = Coordinator::new(&pool, &config, &shutdown)
    .with_global_pause_resource(Arc::clone(&global_pause_resource))
    // ... other builder calls
```

**Step 4: Pass resource to TUI**

```rust
let tui = TuiProgressDisplay::new(receiver, shutdown_token)
    .with_entities(scan_entities, station_entities, audio_entities, candidate_entities)
    .with_pause_request_queue(coordinator.pause_request_queue())
    .with_global_pause_resource(Arc::clone(&global_pause_resource));
```

**Step 5: Build and test**

Run: `cargo build`
Expected: SUCCESS

Run: `RUST_LOG=debug cargo run -- scan --stations 88.9e6 --duration 1`
Expected: Program runs, spacebar press shows [PAUSED] indicator

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire up GlobalPauseResource in main loop

Create GlobalPauseResource and pass to both Coordinator and TUI
during initialization. Both now share the same pause state.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Integration test for pause indicator

**Files:**
- Create: `tests/ui/tui/pause_indicator_test.rs`

**Step 1: Write integration test**

```rust
// tests/ui/tui/pause_indicator_test.rs
use scanner::ecs::{GlobalPauseResource, GlobalPauseState};
use scanner::ui::tui::model::Model;
use std::sync::{Arc, Mutex};

#[test]
fn test_pause_indicator_toggling() {
    let resource = Arc::new(Mutex::new(GlobalPauseState::Active));
    let mut model = Model::new();
    model.set_global_pause_resource(Arc::clone(&resource));

    assert!(!model.is_globally_paused());

    {
        let mut state = resource.lock().unwrap();
        *state = GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: false,
        };
    }

    assert!(model.is_globally_paused());

    {
        let mut state = resource.lock().unwrap();
        *state = GlobalPauseState::Active;
    }

    assert!(!model.is_globally_paused());
}

#[test]
fn test_pause_state_persistence_across_queries() {
    let resource = Arc::new(Mutex::new(GlobalPauseState::Paused {
        had_active_scans: true,
        had_active_audio: true,
    }));
    let mut model = Model::new();
    model.set_global_pause_resource(Arc::clone(&resource));

    for _ in 0..10 {
        assert!(model.is_globally_paused());
    }
}
```

**Step 2: Run test**

Run: `cargo test test_pause_indicator_toggling`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/ui/tui/pause_indicator_test.rs
git commit -m "test: add integration tests for pause indicator

Verify pause state toggling and persistence across queries.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Manual smoke test

**Files:**
- None (manual testing)

**Step 1: Run scanner in FM band mode**

Run: `RUST_LOG=scanner=debug timeout 30s cargo run -- scan --band fm --duration 1 --scanning-windows 2`
Expected: Scanner starts, TUI displays

**Step 2: Test spacebar pause**

Action: Press spacebar during scan
Expected: [PAUSED] indicator appears in bottom status bar

**Step 3: Verify CPU drops**

Action: In another terminal, run `top` and find scanner process
Expected: CPU usage drops significantly (should be < 5%)

**Step 4: Test spacebar resume**

Action: Press spacebar again
Expected: [PAUSED] indicator disappears, scanning resumes

**Step 5: Test pause during listening**

Action: Press ENTER on a station, wait for audio, press spacebar
Expected: Audio stops, [PAUSED] appears

**Step 6: Test resume listening**

Action: Press spacebar again
Expected: Audio resumes, [PAUSED] disappears

**Step 7: Document results**

Create: `docs/manual-test-results.md` with timestamp and findings

**Step 8: Commit documentation**

```bash
git add docs/manual-test-results.md
git commit -m "docs: manual smoke test results for global pause

Verified pause indicator, CPU reduction, and resume behavior.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Run full test suite

**Files:**
- None (test execution)

**Step 1: Run all tests**

Run: `cargo test --lib`
Expected: PASS (all tests)

**Step 2: Run integration tests**

Run: `cargo test --test '*'`
Expected: PASS

**Step 3: Check for dead code warnings**

Run: `make lint`
Expected: No dead code warnings (per CLAUDE.md policy)

**Step 4: If dead code found, remove it**

Action: Remove any functions/fields flagged as dead code
Run: `make lint`
Expected: Clean

**Step 5: Commit cleanup if needed**

```bash
git add <files>
git commit -m "refactor: remove dead code from global pause implementation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Success Criteria

After completing all tasks, verify:

- [ ] Spacebar toggles pause/resume in all UI states
- [ ] [PAUSED] indicator appears in bottom status bar when paused
- [ ] Audio stops immediately when paused
- [ ] CPU usage drops below 5% when paused
- [ ] Resume restores exact previous state (scanning or listening)
- [ ] ENTER during pause removes pause after station tunes
- [ ] All 500+ existing tests pass
- [ ] No dead code warnings
- [ ] `make lint` passes clean

## Notes for Implementation

**TDD Discipline:**
- Write test FIRST for each feature
- Watch it FAIL before implementing
- Write minimal code to pass
- Commit immediately after green

**YAGNI Principle:**
- Don't add pause request queue variants yet (not needed for spacebar)
- Don't add system checks for global pause yet (will add when needed)
- Focus only on spacebar → pause → resume flow

**DRY Principle:**
- Extract repeated pattern into helpers (build_left_instructions_text)
- Reuse existing patterns (Arc<Mutex<T>> for resources)
- Follow existing ECS resource pattern from coordinator

**Common Pitfalls:**
- Don't forget to export new types from modules (GlobalPauseResource in ecs/mod.rs)
- Remember to pass resource through entire chain: main → coordinator → context → TUI
- Check lock() results - they can fail, handle gracefully
- Update is_paused() to include PausedGlobally variant
