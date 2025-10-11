# Code Cleanup and Refactoring Plan

**Date:** 2025-10-10
**Status:** Complete
**Completed:** Tasks 1, 2, 3, 4, 5, 6, 7, 11, 13
**Skipped:** Tasks 8, 9, 10, 12
**Goal:** Identify and address code quality issues to improve maintainability and prepare for future modifications

## Executive Summary

### Initial Issues Identified

- **6 files** exceeded 600 lines (largest was 719 lines)
- **257 unwrap/expect calls** across 43 files (mostly in tests, but some in production code)
- **285 .clone() calls** across 67 files (mostly Arc clones)
- **379 deeply nested if statements** (≥3 levels of indentation)
- **1 large function** (`scan_band()`: 107 lines) with complex state machine logic

### Approach

Address high-impact issues first, review patterns that may already be idiomatic, skip tasks where costs outweigh benefits.

---

## Priority 1: High-Impact Structural Improvements

### 1. Split Large Files into Focused Modules

**Date completed:** 2025-10-10

**Rationale:** Files over 500 lines become difficult to navigate and understand.

**Completed work:**

#### `src/main_thread/mod.rs` (719 → 153 lines, -79%)
- **Split into:**
  - `mod.rs` - Core MainThread struct and public API (153 lines)
  - `runner.rs` - Main scanning loop and band scanning logic (159 lines)
  - `window_processing.rs` - Individual window processing (109 lines)
  - `tests.rs` - Extracted test module (313 lines)

#### `src/audio/quality/random_forest.rs` (641 → 4 modules)
- **Split into:**
  - `random_forest/mod.rs` - Public API and tests (79 lines)
  - `random_forest/model.rs` - Model structure and features (374 lines)
  - `random_forest/training.rs` - Training logic (87 lines)
  - `random_forest/inference.rs` - Prediction logic (116 lines)

#### Other Large Files

The following were reviewed but left as-is:

- `src/hardware/pool/tests.rs` (671 lines) - Test files at this size are acceptable
- `src/scanner_state.rs` (716 lines) - Well-organized module structure
- `src/ui/tui/renderers/scan.rs` (629 lines) - Following Elm Architecture pattern
- `src/signal/squelch.rs` (629 lines) - Cohesive signal processing logic

---

### 2. Refactor `scan_band()` State Machine

**Date completed:** 2025-10-11

**Location:** `src/main_thread/runner.rs:52-83` (32 lines, was 107 lines)

#### Created `ScanContext` struct (`src/main_thread/state_manager.rs`)
- Encapsulates scanning state (window centers, current index, audio session)
- Added `determine_next_action()` method that delegates to mode handlers
- Extracted 6 focused handler methods (10-42 lines each):
  - `handle_shutting_down_mode()` - 3 lines
  - `handle_scan_complete_mode()` - 7 lines
  - `handle_scan_complete_paused_mode()` - 9 lines
  - `handle_paused_mode()` - 15 lines
  - `handle_listening_mode()` - 8 lines
  - `handle_scanning_mode()` - 42 lines

#### Refactored `scan_band()` method
```rust
pub(super) fn scan_band(&mut self) -> Result<()> {
    signal::clear_processed_frequencies();

    let window_centers = self.config.band.windows(
        self.config.samp_rate,
        self.config.signal_processing.window_overlap,
    );
    debug!(
        "Scanning {} windows across {:?} band",
        window_centers.len(),
        self.config.band
    );

    let windows_to_process = match self.config.scanning_windows {
        Some(n) => n.min(window_centers.len()),
        None => window_centers.len(),
    };

    let mut context = state_manager::ScanContext::new(self, window_centers, windows_to_process);

    loop {
        let control = context.determine_next_action()?;

        match control {
            state_manager::LoopControl::Break => break,
            state_manager::LoopControl::Continue => continue,
            state_manager::LoopControl::Advance => context.advance(),
        }
    }

    Ok(())
}
```

Reduced from 107 lines to 32 lines. Eliminated 70-line nested match statement. Each mode handler has single responsibility.

---

### 3. Consider Builder Pattern for Complex Constructors

**Date completed:** 2025-10-10

**Decision:** No changes needed - already using config structs idiomatically.

The codebase already uses config structs for complex constructors:
- `Window::new()` uses `WindowConfig` struct
- `SquelchBlock::new()` uses `SquelchConfig` struct
- `CommandHandler::new()` uses `CommandHandlerConfig` struct
- `MainThread` uses `ScanningConfig` struct
- `MainThread` has `with_command_receiver()` and `with_tui_event_sender()` builder methods

Most parameters are required dependencies. Only 2 constructor variants for distinct use cases (headless vs TUI). Current hybrid pattern (constructor + `with_*` methods) is appropriate.

---

## Priority 2: Code Quality and Maintainability

### 4. Introduce More Granular Error Variants

**Date completed:** 2025-10-11

Replace generic string-based errors with specific, typed variants.

#### Phase 1: High-Impact Errors (11 variants)
- **Hardware discovery** (3 variants): `NoSdrDevicesFound`, `DeviceFilteredOut`, `UnsupportedDeviceIdFormat`
- **ML model errors** (5 variants): `ModelNotTrained`, `InsufficientTrainingData`, `InvalidModelFile`, `ModelFeatureMismatch`, `ModelSaveFailed`
- **Configuration** (3 variants): `InvalidSquelchThreshold`, `InvalidTheme`, `IqCaptureMaxFiles`

#### Phase 2: Internal Validation (2 variants)
- **Initialization**: `GraphInitTimeout` with component and timeout duration
- **Signal processing**: `EmptyAudioBuffer` with minimum sample requirements
- **I/O improvement**: Removed unnecessary `ConfigurationError` wrapping, now uses native `io::Error`

#### Phase 3: Low-Impact Errors (Analysis Complete)
- Remaining errors (`ThreadPanic`, `UnsupportedAudioFormat`) are rare/unrecoverable - kept as-is

15 of 20 production string errors converted. 13 new specific error variants added.

---

### 5. Extract Testable Traits for Hardware Abstraction

**Date completed:** 2025-10-10

**Decision:** No changes needed.

Pool's allocation logic already testable via existing `DeviceTrait` abstraction. All 18 pool tests use `MockDevice`. Allocation logic is ~50 lines in a single method. Extracting a `TunerAllocationStrategy` trait would be premature abstraction.

---

### 6. Consider Type-State Pattern for Scanner State

**Date completed:** 2025-10-10

**Implementation:** Enum-of-typestates pattern - each state is a struct, wrapped in an enum for runtime flexibility.

Created state structs (`Scanning`, `Paused`, `Listening`, etc.) and updated `ScanMode` enum to wrap them. Added state transition methods that consume self and return new state.

Type-safe transitions (can't call `pause()` on `Paused` struct). State data explicit in struct fields. Enum allows pattern matching on dynamic user events.

---

## Priority 3: Performance and Algorithmic Improvements

### 7. Audit Unnecessary Cloning

**Date completed:** 2025-10-11

285 `.clone()` calls found across 67 files. >90% are cheap Arc clones. No hot loop clones of Vec/String.

`ScanningConfig` was cloned ~100 times per scan. Wrapped in Arc throughout codebase. Per-window config clones now cheap (ref count increment).

---

### 8. Leverage Iterator Trait More Extensively (Skipped)

**Date:** 2025-10-11

Current iterator usage is already idiomatic. Data transformations use iterator chains. Main scanning loops use explicit control flow for debuggability. No concrete improvements identified.

---

## Priority 4: Forward-Looking Architectural Improvements

### 9. Prepare for Async/Await in I/O Operations (Skipped)

**Date:** 2025-10-11

Current I/O volume is low (infrequent file writes for audio capture, IQ recording). Thread-based architecture with blocking I/O in dedicated threads is simple and maintainable. No I/O bottlenecks identified. Mixing sync DSP with async I/O would add complexity without benefit.

---

### 10. Introduce Feature Flags for ML Model Selection (Skipped)

**Date:** 2025-10-11

Runtime selection via `AudioAnalyzer` enum is more flexible. All classifiers (~2K lines) co-exist in binary. Can switch analyzers via config without recompilation. Single binary works for all use cases with simple CI/CD. Feature flags would require testing all combinations and rebuilding to test different classifiers.

---

## Priority 5: Safety and Robustness

### 11. Reduce Unwrap/Expect Usage in Production Code

**Date completed:** 2025-10-11

257 unwrap/expect calls across 43 files. >95% in test code (acceptable). 10 expects found in production code and replaced with proper error handling:

- src/logging.rs (6 expects) - Tracing subscriber initialization now uses `?` operator
- src/cli/signals.rs (1 expect) - Signal handler setup now returns Result
- src/scanning/window/audio.rs (3 expects) - Audio device setup uses appropriate error types

---

### 12. Expand Loom Testing for Concurrent Code (Skipped)

**Date:** 2025-10-11

5 existing Loom tests in `tests/loom_shutdown_test.rs` cover shutdown scenarios. Core concurrent patterns already tested (shutdown + pause interaction). Full Pool acquisition tests would require extensive mocking. No evidence of concurrency issues in production. Expanding tests without concrete bugs to prevent would be speculative.

---

## Priority 6: Documentation and Discoverability

### 13. Add Module-Level Documentation

**Date completed:** 2025-10-11

6 modules documented or enhanced (~450 lines added). 2 modules documented from scratch (src/signal/mod.rs, src/broadcast.rs). 4 modules enhanced (src/signal/peaks/mod.rs expanded from 4 to 92 lines, src/scanner_state.rs expanded from 3 to 146 lines, src/audio/quality/mod.rs and src/ui/tui/mod.rs enhanced with research links). All CLAUDE.md research files now referenced from relevant modules.

---

## Summary

9 tasks completed:
1. Split large files - 2 files refactored into focused modules
2. Refactor state machine - scan_band() reduced from 107 to 32 lines
3. Builder pattern review - current approach is idiomatic
4. Granular error variants - 15 of 20 string errors converted
5. Hardware abstraction review - existing DeviceTrait abstraction is sufficient
6. Typestate pattern - implemented enum-of-typestates
7. Clone optimization - ScanningConfig wrapped in Arc
11. Production unwraps eliminated - all 10 production expects replaced
13. Module documentation - 6 modules documented or enhanced

4 tasks skipped:
8. Iterator usage - current mix is idiomatic
9. Async I/O - thread-based architecture appropriate for current I/O patterns
10. Feature flags - runtime selection more flexible
12. Loom testing expansion - current 5 tests sufficient

All 239 library tests passing. Zero compilation warnings.
