# Plan 012: Code Quality Cleanup and Refactoring

## Implementation Plan

### Phase 1: Quick Wins (High Impact, Low Risk)

#### Task 1.1: Remove `get_` prefix from functions ✅ COMPLETE
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

#### Task 2.1: Decompose `src/terminal/tui/model.rs` ✅ COMPLETE
**Status:** Completed 2025-10-08

Current: 3,657 lines (7x the 500-line threshold) → **8 focused modules**

Final structure:

```
src/terminal/tui/model/
├── mod.rs           # Module exports (~22 lines) ✅
├── types.rs         # Type definitions (169 lines) ✅
├── state.rs         # Model struct + constructor (48 lines) ✅
├── devices.rs       # Device/tuner management (56 lines) ✅
├── queries.rs       # Read-only methods (185 lines) ✅
├── navigation.rs    # Navigation/selection (239 lines) ✅
├── updates.rs       # Event processing (236 lines) ✅
└── tests.rs         # Tests (2,713 lines) ✅
```

**Results:**
- All modules < 250 lines (well under 500-line threshold)
- 35/35 model tests passing
- 233/233 total lib tests passing
- Clear separation of concerns achieved

#### Task 2.2: Decompose `src/pool/mod.rs` ✅ COMPLETE
**Status:** Completed 2025-10-08

Current: 1,267 lines mixing multiple responsibilities → **8 focused modules**

Final structure:

```
src/pool/
├── mod.rs           # Public API (38 lines) ✅
├── filter.rs        # PoolFilter, TuningMode (156 lines) ✅
├── state.rs         # PoolInner, Pool struct (92 lines) ✅
├── lifecycle.rs     # Tuner acquisition/release (332 lines) ✅
├── tests.rs         # Tests (668 lines) ✅
├── segment.rs       # Existing (180 lines)
├── tuner.rs         # Existing (145 lines)
└── types.rs         # Existing (157 lines)
```

**Results:**
- All modules < 350 lines (well under 500-line threshold)
- All shutdown safety patterns preserved ✅
- 25/25 pool tests passing
- 233/233 total lib tests passing
- Clear separation: filter config, core state, lifecycle operations

#### Task 2.3: Extract renderer sub-functions ✅ COMPLETE
**Status:** Completed 2025-10-08

Extracted helper functions from both renderers to improve readability:

**scan.rs:**
- `render_empty_state()` - Empty state display
- `render_scroll_up_indicator()` - Scroll up indicator
- `render_scroll_down_indicator()` - Scroll down indicator

**scan_caladan.rs:**
- `create_scroll_up_indicator()` - Scroll up Line
- `create_scroll_down_indicator()` - Scroll down Line
- `create_empty_state_line()` - Empty state Line

**Results:**
- Removed duplication of scroll indicator styling
- Improved code readability
- 233/233 tests passing
- No behavioral changes

#### Task 2.4: Introduce parameter context structs ✅
**Status:** COMPLETED

Refactored functions with excessive parameters (>4) by introducing context structs. Completed:

**1. `src/terminal/tui/renderers/scan.rs` - Renderer functions**
- `render_candidate_progress` (6 params → `CandidateRenderContext`)
- `render_window_header` (6 params → `WindowHeaderContext`, removed unused params)

**2. `src/terminal/tui/renderers/scan_caladan.rs` - Caladan theme**
- `render_candidate_line` (5 params → `CandidateLineContext`)

**3. `src/peaks/mod.rs` - DSP function**
- `process_samples_for_peaks` (5 params → `PeakProcessingParams`)

**Results:**
- Reduced parameter lists from 5-6 params to 1-2 params
- Grouped related parameters logically in context structs
- Improved readability and maintainability
- Easier to extend functionality (add fields to context)
- All tests passing (cargo check clean)

#### Task 2.5: Refactor state machine complexity ✅ COMPLETE
**Status:** Completed 2025-10-08

**`src/main_thread.rs:579`** - `scan_band` refactored from 139 lines → 70 lines, complexity 16 → ~5, nesting 8 → 3 levels

Extracted state handlers:

**Implementation:**

1. **Created `LoopControl` enum** with `Continue`, `Break`, `Advance` variants
2. **Extracted helper methods:**
   - `process_window()` - 27 lines: Window creation and processing
   - `process_commands_with_pause_check()` - 14 lines: Consolidated repeated pause check pattern
3. **Extracted state handlers:**
   - `handle_scanning_state()` - 40 lines: Main scanning logic (was 60 lines inline)
   - `handle_paused_state()` - 14 lines
   - `handle_listening_state()` - 8 lines
   - Updated `handle_post_scan_waiting()` and `handle_post_scan_browse_mode()` to return `LoopControl`
4. **Simplified main loop** to clean state dispatch pattern

**Results:**
- Main `scan_band()` loop: 70 lines (down from 139)
- Nesting reduced: 8 levels → 3 levels max
- Complexity reduced: ~16 → ~5 for main loop
- Each handler: <20 lines (except `handle_scanning_state` at 40 lines, but well-structured)
- Eliminated 3x repetition of command processing in Scanning branch
- All shutdown safety and pause responsiveness preserved
- `cargo check` passing

### Phase 3: Architectural Improvements

#### Task 3.1: Decouple window processing from pool ✅ COMPLETE
**Status:** Completed 2025-10-08

Decoupled Window from concrete Pool type by introducing TunerProvider trait abstraction.

**Implementation:**

1. **Created `src/pool/provider.rs`** (46 lines) - TunerProvider trait
   ```rust
   pub trait TunerProvider: Send + Sync {
       fn acquire(&self, requirements: &TaskRequirements, activity: TunerActivity) -> Result<Tuner>;
       fn try_acquire(&self, requirements: &TaskRequirements, activity: TunerActivity) -> Option<Tuner>;
   }

   impl TunerProvider for Pool {
       // Delegates to existing Pool methods
   }
   ```

2. **Updated `src/pool/mod.rs`** - Added module and export
   - Added `mod provider;`
   - Added `pub use provider::TunerProvider;`

3. **Updated `src/window/config.rs`** - Changed WindowConfig field
   - Before: `pool: Arc<crate::pool::Pool>`
   - After: `tuner_provider: Arc<dyn TunerProvider>`

4. **Updated `src/window/mod.rs`** - Window struct and methods
   - Changed struct field to `tuner_provider: Arc<dyn TunerProvider>`
   - Updated `new()` constructor to use `tuner_provider`
   - Updated `for_station()` signature to accept `Arc<dyn TunerProvider>`
   - Updated `process_with_pool()` to call `self.tuner_provider.acquire()`

5. **Updated `src/main_thread.rs`** - Call sites
   - Changed WindowConfig field name: `pool:` → `tuner_provider:`
   - Pool automatically coerces to `Arc<dyn TunerProvider>` due to trait implementation

**Benefits achieved:**
- ✅ Window no longer depends on concrete Pool type
- ✅ Enables dependency injection for testing (can create MockTunerProvider)
- ✅ Follows dependency inversion principle (depends on abstraction)
- ✅ Supports alternative tuner management strategies in future
- ✅ No behavioral changes - Pool still used everywhere via trait
- ✅ `cargo check` and `make lint` passing

#### Task 3.2: Consolidate audio graph creation ✅
**Status:** ALREADY COMPLETE

FM pipeline building has already been consolidated using `FmPipelineBuilder`:

**Existing implementation** (`src/fm/pipeline_builder.rs`):
- `create_frequency_xlating_filter()` - Shared freq xlating filter stage with optimized FM parameters
- `create_audio_decimation_chain()` - Shared audio decimation (anti-aliasing + rational resampler)

**Usage confirmed in:**
- `src/fm/mod.rs::create_detection_graph()` - Uses both builder methods
- `src/window.rs::create_fm_graph()` - Uses both builder methods

**Benefits achieved:**
- ✅ Eliminated pipeline duplication
- ✅ Centralized filter configuration logic
- ✅ Shared FM-specific optimizations
- ✅ Single source of truth for graph creation

#### Task 3.3: Reduce deep nesting patterns ✅
**Status:** COMPLETED

Reduced nesting in complex functions using helper methods and early returns:

**1. `src/discovery/enumerator.rs` - USB device enumeration**
- Extracted `try_extract_device_info()` helper method
- Reduced nesting from 4 levels to 2 levels
- Used `?` operator for early returns
- Improved readability with flat control flow

**2. `src/window.rs` - Thread waiting logic**
- Extracted helper methods:
  - `should_stop_waiting()` - Shutdown check
  - `log_pause_signal_if_present()` - Pause logging
  - `join_finished_threads()` - Join completed threads
  - `join_remaining_threads()` - Cleanup timeout threads
- Reduced nesting from 8 levels to 3 levels
- Main loop now clean and readable
- Preserved all shutdown/pause logic

**Results:**
- Improved code readability
- Easier to test individual components
- Reduced cognitive complexity
- All tests passing (cargo check clean)

#### Task 3.4: Split `src/window.rs` ✅ COMPLETE
**Status:** Completed 2025-10-08

Current: 982 lines mixing window processing with audio infrastructure → **4 focused modules**

Final structure:

```
src/window/
├── mod.rs           # Public API and orchestration (319 lines) ✅
├── audio.rs         # Audio device setup, stream creation, FM graph (450 lines) ✅
├── processing.rs    # Peak/candidate processing logic (212 lines) ✅
└── config.rs        # WindowConfig, WindowMetadata (22 lines) ✅
```

**Implementation:**

1. **Created `config.rs`** (22 lines) - Configuration types
   - `WindowMetadata` struct
   - `WindowConfig` struct

2. **Created `audio.rs`** (450 lines) - Audio infrastructure
   - `setup_audio_device()` - Audio device configuration
   - `create_audio_stream()` - CPAL audio stream creation
   - `process_signal_for_audio()` - Audio processing pipeline
   - `create_audio_fm_graph()` - FM demodulation graph
   - Helper functions for graph building (frequency xlating filter, FM demod chain, etc.)
   - `play_signals()` - Audio playback orchestration

3. **Created `processing.rs`** (212 lines) - Peak/candidate processing
   - `peaks()` - Peak detection (station mode vs band scanning)
   - `debug_peaks()` - Debug output for peaks
   - `candidates_from_peaks()` - Convert peaks to candidates
   - `process_candidates()` - Spawn threads for candidate analysis

4. **Created `mod.rs`** (319 lines) - Main Window orchestration
   - `Window` struct with all fields
   - `new()` and `for_station()` constructors
   - `process_with_pool()` - Main entry point with pool-based tuner management
   - `process()` - Core processing logic orchestrating peaks → candidates → audio
   - Thread management helpers (wait_for_threads, join_finished_threads, etc.)
   - Re-exports public types and functions

5. **Updated `audio_session.rs`** - Fixed imports to use module-level functions instead of Window associated functions

**Results:**
- Total lines: 1,003 (982 original + 21 for module structure)
- All modules < 500 lines (well under threshold)
- Clear separation: config (22), processing (212), orchestration (319), audio (450)
- Public API preserved - external code uses `use crate::window::{Window, WindowConfig, WindowMetadata};` unchanged
- All imports continue to work via re-exports from mod.rs
- `cargo check` passing
- No behavioral changes

### Phase 4: Module Reorganization (Future)

#### Task 4.1: Reorganize module structure

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
