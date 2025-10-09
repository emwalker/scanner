# Plan 012: Code Quality Cleanup and Refactoring

## Implementation Plan

### Phase 1: Quick Wins (High Impact, Low Risk)

#### Task 1.1: Remove `get_` prefix from functions ✅ COMPLETE
**Priority:** HIGH
**Effort:** Low
**Risk:** Low
**Status:** Completed 2025-10-08

Rename 18 functions to follow Rust conventions:

- `src/frequency_tracking.rs:69, 229`: `get_confidence()` → `confidence()`
- `src/testing/signal_generation.rs:78`: `get_expected_peaks()` → `expected_peaks()`
- `src/testing/signal_generation.rs:82`: `get_signal_labels()` → `signal_labels()`
- `src/peaks/noise_floor.rs:228`: `get_statistics()` → `statistics()`
- `src/peaks/multi_frame.rs:214`: `get_confirmed_peaks()` → `confirmed_peaks()`
- `src/peaks/multi_frame.rs:232`: `get_statistics()` → `statistics()`
- `src/terminal/mod.rs:127`: `get_events()` → `events()`
- `src/terminal/tui/model.rs:496`: `get_or_create_window()` → `or_create_window()`
- `src/terminal/tui/model.rs:646`: `get_displayable_windows()` → `displayable_windows()`
- `src/terminal/tui/model.rs:654`: `get_displayable_window_count()` → `displayable_window_count()`
- `src/terminal/tui/model.rs:664`: `get_displayable_candidates()` → `displayable_candidates()`
- `src/terminal/tui/model.rs:677`: `get_selectable_candidates()` → `selectable_candidates()`
- `src/terminal/tui/model.rs:692`: `get_displayable_candidate_count()` → `displayable_candidate_count()`
- `src/terminal/tui/model.rs:697`: `get_selectable_candidate_count()` → `selectable_candidate_count()`
- `src/logging.rs:17`: `get_string()` → `string()` or `into_string()`
- `src/main_thread.rs:630`: `get_messages()` → `messages()` (test mock)
- `src/testing/performance_regression.rs:214`: `get_memory_usage_mb()` → `memory_usage_mb()`

**Validation:** Run tests to ensure no breakage.

### Task 1.2: Standardize Import Patterns ✅ COMPLETE
**Priority:** HIGH
**Effort:** Medium
**Risk:** Low
**Status:** Completed 2025-10-08

Fix 35+ non-idiomatic imports per CLAUDE.md guidelines.

**Pattern 1: Import `std::sync::mpsc` types directly**

Files to update:
- `src/mpsc.rs`: Add `use std::sync::mpsc::{SyncSender, TrySendError};`
  - Replace `std::sync::mpsc::SyncSender<AudioPacket>` → `SyncSender<AudioPacket>`
  - Replace `std::sync::mpsc::TrySendError::Full` → `TrySendError::Full`
- `src/fm/mod.rs`: Import `SyncSender` directly
- `src/main_thread.rs`: Add `use std::sync::mpsc::{Sender, Receiver};`

**Pattern 2: Import math constants**

Files to update (8 files):
- `src/sdr/sample_source.rs:90`: Add `use std::f32::consts::PI;` (4 occurrences)
- `src/testing/test_helpers.rs:224, 244, 245`: Import `PI`
- `src/testing/signal_generation.rs:105`: Import `PI`
- `src/fm/deemph.rs:19`: Import `PI`
- `src/sdr/mock.rs:102`: Import `PI`

**Pattern 3: Import atomic types and Ordering**

Files to update:
- `src/mpsc.rs`: Add `use std::sync::atomic::Ordering;`
- `src/broadcast.rs`: Import atomic types and `Ordering`

**Pattern 4: Other standard library types**

- `src/peaks/noise_floor.rs`: Import `std::cmp::Ordering`
- `src/types.rs`: Import `std::time::SystemTime`
- `src/peaks/multi_frame.rs`: Import `std::time::Instant`
- `src/testing/test_helpers.rs`: Import `tokio::sync::broadcast::error::TryRecvError`
- `src/sdr/sample_source.rs`: Import `TryRecvError`

**Validation:** `cargo check` should pass with no new warnings.

### Task 1.3: Extract Event Handlers from `run_app` ✅ COMPLETE
**Priority:** HIGH
**Effort:** High
**Risk:** Medium
**Status:** Completed 2025-10-08

`src/terminal/tui/mod.rs:131` - 220 lines → 34 lines ✅, complexity 53 → <10 ✅

Extract focused event handlers:

```rust
// In src/terminal/tui/mod.rs

impl<B: Backend> App {
    fn handle_keyboard_event(&mut self, key: KeyCode) -> Result<bool> {
        // Extract keyboard handling logic
    }

    fn handle_theme_selector(&mut self, key: KeyCode) {
        // Extract theme selection logic
    }

    fn handle_navigation(&mut self, key: KeyCode) {
        // Extract navigation logic
    }

    fn handle_tuning_action(&mut self, /* params */) -> Result<()> {
        // Extract tuning logic
    }

    fn process_tui_events(&mut self, terminal: &mut Terminal<B>) -> Result<bool> {
        // Extract TUI event processing
    }

    fn update_animation_frame(&mut self) {
        // Extract animation updates
    }

    fn run_app(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        // Simplified main loop
        loop {
            terminal.draw(|f| self.ui(f))?;

            if self.should_quit() { break; }

            let needs_redraw = self.handle_keyboard_event(event)?;
            if needs_redraw {
                terminal.draw(|f| self.ui(f))?;
            }

            self.process_tui_events()?;
        }
        Ok(())
    }
}
```

**Expected Outcome:**
- `run_app` reduced to <30 lines
- Complexity reduced from 53 → <10
- Each handler <20 lines

### Phase 2: Structural Improvements

#### Task 2.1: Decompose `src/terminal/tui/model.rs`
**Priority:** HIGH
**Effort:** High
**Risk:** High

Current: 3,657 lines (7x the 500-line threshold)

Split into focused modules:

```
src/terminal/tui/model/
├── mod.rs           # Public API and coordination (~100 lines)
├── types.rs         # CandidateProgress, WindowProgress structs (~300 lines)
├── state.rs         # UiMode, FocusState, state machine (~400 lines)
├── updates.rs       # Event update logic (~800 lines)
└── queries.rs       # Read-only query methods (~500 lines)
```

**Migration Strategy:**
1. Create `model/` directory
2. Extract `types.rs` first (no dependencies)
3. Extract `state.rs` (depends on types)
4. Extract `queries.rs` (read-only, safe)
5. Extract `updates.rs` (complex, do last)
6. Update `mod.rs` to re-export public API
7. Update imports in dependent files

**Validation:** All tests pass, no functional changes.

#### Task 2.2: Decompose `src/pool/mod.rs`
**Priority:** HIGH
**Effort:** Medium
**Risk:** Medium

Current: 1,267 lines mixing multiple responsibilities

Split into:

```
src/pool/
├── mod.rs           # Public API (~200 lines)
├── filter.rs        # PoolFilter, TuningMode (~150 lines)
├── state.rs         # PoolInner, core state (~300 lines)
└── lifecycle.rs     # Tuner acquisition/release (~400 lines)
```

**Critical:** Maintain all shutdown safety patterns during refactoring.

#### Task 2.3: Extract renderer sub-functions
**Priority:** MEDIUM
**Effort:** Medium
**Risk:** Low

**`src/terminal/tui/renderers/scan.rs:16`** - `render_scan` (211 lines, complexity 31)

Extract:
```rust
fn render_scan_header(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme)
fn render_window_list(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme)
fn render_candidate_section(f: &mut Frame, area: Rect, window: &WindowProgress, theme: &dyn Theme)
fn render_scan_footer(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme)

fn render_scan(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    let layout = create_scan_layout(area);
    render_scan_header(f, layout.header, model, theme);
    render_window_list(f, layout.windows, model, theme);
    render_scan_footer(f, layout.footer, model, theme);
}
```

Repeat for `src/terminal/tui/renderers/scan_caladan.rs:15` (204 lines)

#### Task 2.4: Introduce parameter context structs
**Priority:** MEDIUM
**Effort:** Medium
**Risk:** Low

Fix 43 functions with >4 parameters. Focus on worst offenders:

**`src/terminal/tui/renderers/scan.rs:252`** - `render_candidate_progress` (7 params)

```rust
struct CandidateRenderContext<'a> {
    candidate: &'a CandidateProgress,
    window_id: usize,
    is_current: bool,
    is_selected: bool,
    is_playing: bool,
    theme: &'a dyn Theme,
}

fn render_candidate_progress(
    f: &mut Frame,
    area: Rect,
    ctx: &CandidateRenderContext,
) {
    // Use ctx.candidate, ctx.is_selected, etc.
}
```

**`src/main_thread.rs:230`** - `handle_command` (5 params)

```rust
struct CommandContext<'a> {
    window_num: usize,
    total_windows: usize,
    audio_session: &'a mut Option<AudioSession>,
}

fn handle_command(
    &mut self,
    command: ScannerCommand,
    ctx: &mut CommandContext,
) -> Result<Option<ScannerCommand>>
```

**`src/fm/mod.rs:86`** - `analyze_spectral_characteristics` (5 params)

```rust
struct SpectralAnalysisParams {
    magnitudes: Vec<f32>,
    fft_size: usize,
    sample_rate: f64,
    center_freq: f64,
    peak_freq: f64,
}

fn analyze_spectral_characteristics(params: SpectralAnalysisParams) -> SpectralCharacteristics
```

**`src/peaks/mod.rs:173`** - `process_samples_for_peaks` (6 params)

```rust
struct PeakProcessingConfig<'a> {
    config: &'a ScanningConfig,
    center_freq: f64,
    fft_size: usize,
    sample_rate: f64,
}

fn process_samples_for_peaks(
    samples: Vec<Complex>,
    accumulated_frames: &mut Vec<Vec<f32>>,
    config: &PeakProcessingConfig,
) -> Vec<Peak>
```

#### Task 2.5: Refactor state machine complexity
**Priority:** MEDIUM
**Effort:** Medium
**Risk:** Medium

**`src/main_thread.rs:456`** - `scan_band` (139 lines, complexity 16, 8 nesting levels)

Extract state handlers:

```rust
enum LoopControl {
    Continue,
    Break,
}

impl MainThread {
    fn handle_shutdown_state(&mut self) -> LoopControl {
        // Shutdown logic
    }

    fn handle_scan_complete(&mut self, ...) -> Result<LoopControl> {
        // Scan complete logic
    }

    fn handle_paused_state(&mut self, i: usize, ...) -> Result<LoopControl> {
        // Paused state logic
    }

    fn handle_listening_state(&mut self, i: usize, ...) -> Result<LoopControl> {
        // Listening state logic
    }

    fn handle_scanning_state(&mut self, i: usize, ...) -> Result<LoopControl> {
        // Scanning state logic
    }

    fn scan_band(&mut self) -> Result<()> {
        let window_centers = self.config.band.windows(...);
        let mut i = 0;

        loop {
            if self.shutdown_coordinator.is_shutdown() {
                self.scanner_state.shutdown();
            }

            let control = match &self.scanner_state.mode {
                ScanMode::ShuttingDown => self.handle_shutdown_state(),
                ScanMode::ScanComplete { .. } => self.handle_scan_complete(...)?,
                ScanMode::Paused { .. } => self.handle_paused_state(i, ...)?,
                ScanMode::Listening { .. } => self.handle_listening_state(i, ...)?,
                ScanMode::Scanning => self.handle_scanning_state(i, ...)?,
            };

            if control == LoopControl::Break { break; }
        }
        Ok(())
    }
}
```

### Phase 3: Architectural Improvements

#### Task 3.1: Decouple window processing from pool
**Priority:** MEDIUM
**Effort:** High
**Risk:** High

Current: Window directly couples to pool for tuner management, making testing difficult.

Introduce abstraction:

```rust
// src/sdr/tuner_provider.rs
pub trait TunerProvider {
    fn acquire(&self, requirements: &TunerRequirements) -> Result<Tuner>;
    fn release(&self, tuner: Tuner);
}

impl TunerProvider for Pool {
    fn acquire(&self, requirements: &TunerRequirements) -> Result<Tuner> {
        self.acquire_tuner(requirements)
    }

    fn release(&self, tuner: Tuner) {
        self.release_tuner(tuner)
    }
}

// Update Window to use trait
pub struct Window {
    tuner_provider: Arc<dyn TunerProvider>,
    // ...
}
```

**Benefits:**
- Enable testing without full pool
- Support alternative tuner management strategies
- Follow dependency inversion principle

#### Task 3.2: Consolidate audio graph creation
**Priority:** MEDIUM
**Effort:** Medium
**Risk:** Low

Current: FM pipeline building duplicated across multiple files

Create fluent builder API in `src/fm/pipeline/builder.rs`:

```rust
pub struct FmPipelineBuilder {
    sample_rate: f64,
    center_freq: f64,
    // ...
}

impl FmPipelineBuilder {
    pub fn new(sample_rate: f64, center_freq: f64) -> Self {
        Self { sample_rate, center_freq }
    }

    pub fn with_xlating_filter(mut self, ...) -> Self {
        // Configure xlating filter
        self
    }

    pub fn with_demodulator(mut self, ...) -> Self {
        // Configure FM demod
        self
    }

    pub fn with_audio_output(mut self, ...) -> Self {
        // Configure audio output
        self
    }

    pub fn build(self) -> Result<Graph> {
        // Construct complete graph
    }
}
```

Centralize all graph creation patterns to single source of truth.

#### Task 3.3: Reduce deep nesting patterns
**Priority:** LOW
**Effort:** Medium
**Risk:** Low

**`src/discovery/enumerator.rs`** - `enumerate` (9 nesting levels, 56 deep lines)

Flatten with early returns and helper functions:

```rust
fn should_include_device(device: &DeviceInfo, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(f) => device.matches(f),
    }
}

fn enumerate_backend_devices(
    backend: &dyn Backend,
    driver_filter: Option<&str>,
) -> Result<Vec<DeviceInfo>> {
    let devices = backend.enumerate_devices()?;
    Ok(devices.into_iter()
        .filter(|d| should_include_device(d, driver_filter))
        .collect())
}

fn enumerate(backends: &[Box<dyn Backend>], filter: Option<&str>) -> Result<Vec<DeviceInfo>> {
    let mut all_devices = Vec::new();

    for backend in backends {
        let devices = enumerate_backend_devices(backend.as_ref(), filter)?;
        all_devices.extend(devices);
    }

    Ok(all_devices)
}
```

**`src/window.rs:873`** - `wait_for_threads_with_timeout` (82 lines, 8 nesting)

Extract helpers:

```rust
fn should_stop_waiting(&self, start_time: Instant, timeout: Duration) -> bool {
    self.shutdown_token.is_cancelled()
        || self.pause_signal.as_ref().map_or(false, |s| s.is_paused())
        || start_time.elapsed() >= timeout
}

fn join_finished_threads(
    &self,
    threads: Vec<JoinHandle<Result<()>>>,
) -> (usize, Vec<JoinHandle<Result<()>>>) {
    let mut completed = 0;
    let mut still_running = Vec::new();

    for handle in threads {
        if handle.is_finished() {
            // Join and count
            completed += 1;
        } else {
            still_running.push(handle);
        }
    }

    (completed, still_running)
}
```

#### Task 3.4: Split `src/window.rs`
**Priority:** LOW
**Effort:** High
**Risk:** High

Current: 956 lines mixing window processing with audio infrastructure

Split into:

```
src/window/
├── mod.rs           # Public API and orchestration
├── audio.rs         # Audio device setup, stream creation, FM graph
├── processing.rs    # Peak/candidate processing logic
└── config.rs        # WindowConfig, WindowMetadata
```

### Phase 4: Module Reorganization (Future)

#### Task 4.1: Reorganize module structure
**Priority:** LOW
**Effort:** Very High
**Risk:** Very High

Proposed new structure for improved boundaries:

```
src/
├── core/               # Core domain types
│   ├── types.rs
│   └── events.rs
├── scanning/           # Scanning orchestration
│   ├── window/
│   ├── state.rs
│   └── session.rs
├── signal/             # Signal processing (rename from fm/)
│   ├── fm/
│   │   ├── pipeline/
│   │   ├── analysis/
│   │   └── blocks/
│   └── peaks/
├── hardware/           # SDR hardware (rename from sdr/)
│   ├── pool/
│   ├── backends/
│   └── discovery/
├── ui/                 # User interface (split from terminal/)
│   ├── tui/
│   │   ├── state/
│   │   ├── renderers/
│   │   └── themes/
│   └── console/
└── audio/              # Audio infrastructure
    ├── quality/
    └── infrastructure/
```
