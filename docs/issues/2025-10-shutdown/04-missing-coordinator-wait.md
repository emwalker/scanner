# Missing shutdown_coordinator.wait() Call

**Date**: October 2025
**Status**: ✅ FIXED

## Problem Statement

After the pool migration, the scanner failed to shut down cleanly:
- TUI mode: Empty tuner list displayed, process wouldn't exit cleanly (required `pkill`)
- Headless mode: Similar shutdown hang

## Root Cause

The `ShutdownCoordinator` tracks threads spawned via `spawn_sdr_thread()`, but its `wait()` method was never called during shutdown. This meant audio graph threads and other coordinator-managed threads were never joined before process exit.

`AudioSession` spawns audio graph threads using `shutdown_coordinator.spawn_sdr_thread()` (src/audio_session.rs:87). When `AudioSession::stop_current_station()` is called, it cancels these threads but doesn't join them. The code comment says "coordinator manages the thread join", but `shutdown_coordinator.wait()` was never being called in `bin/scanner.rs`.

## Code Analysis

**What AudioSession does** (src/audio_session.rs:87):
```rust
self.shutdown_coordinator
    .spawn_sdr_thread(move |shutdown_token| {
        // Audio graph processing...
        audio_graph.run()
    })?;
```

**What stop_current_station() does** (src/audio_session.rs:120-123):
```rust
if let Some(cancel_token) = self.current_graph_cancel.take() {
    cancel_token.cancel();
    // Comment says: "coordinator manages the thread join"
    // But wait() was never called!
}
```

**What was missing** (bin/scanner.rs):
- No call to `shutdown_coordinator.wait()` in TUI mode shutdown path
- No call to `shutdown_coordinator.wait()` in headless mode shutdown path

## The Fix

### Changes Made

**1. Modified ShutdownCoordinator::wait() signature** (src/shutdown.rs:119):
```rust
// Before: pub fn wait(self) -> Result<()>
// After:  pub fn wait(&self) -> Result<()>
```

Changed from consuming `self` to taking `&self` because we use `Arc<ShutdownCoordinator>` and can't easily unwrap it when there are multiple clones.

Implementation changed to drain handles from the mutex:
```rust
let mut handles_guard = self.thread_handles.lock().unwrap();
let handles: Vec<_> = handles_guard.drain(..).collect();
drop(handles_guard);
```

**2. Added wait() call in TUI mode** (bin/scanner.rs:559):
```rust
match main_handle.join() {
    Ok(r) => r?,
    Err(e) => {
        return Err(scanner::types::ScannerError::ThreadJoin(e));
    }
}

// CRITICAL: Join all threads spawned via coordinator (e.g., audio graph threads)
// Without this, audio threads keep running and prevent clean shutdown
shutdown_coordinator.wait()?;
```

**3. Added wait() call in headless mode** (bin/scanner.rs:609):
```rust
Ok(r) => {
    shutdown_coordinator.shutdown();
    shutdown_coordinator.wait()?;  // ← Added this line
    r?
}
```

## Shutdown Flow (After Fix)

**TUI mode**:
1. User presses 'q' or Ctrl+C
2. `shutdown_coordinator.shutdown()` triggers cancellation (line 539)
3. Discovery/forwarder threads joined (lines 542-544)
4. Main thread joined (line 550)
5. **`shutdown_coordinator.wait()` joins all coordinator-tracked threads** (line 559)
6. Process exits cleanly

**Headless mode**:
1. Scan completes or Ctrl+C pressed
2. Main thread joined (line 605)
3. `shutdown_coordinator.shutdown()` triggers cancellation (line 608)
4. **`shutdown_coordinator.wait()` joins all coordinator-tracked threads** (line 609)
5. Process exits cleanly

## Testing

After fix:
- ✅ TUI mode: Tuner list populates correctly
- ✅ TUI mode: Clean shutdown with 'q' key
- ✅ TUI mode: Clean shutdown with Ctrl+C
- ✅ Headless mode: Clean shutdown on completion
- ✅ Headless mode: Clean shutdown with Ctrl+C
- ✅ No SDR device corruption
- ✅ No need for `sudo systemctl restart sdrplay` between runs

## Files Modified

- `src/shutdown.rs:119` - Changed `wait(self)` to `wait(&self)`
- `bin/scanner.rs:559` - Added `shutdown_coordinator.wait()` in TUI mode
- `bin/scanner.rs:609` - Added `shutdown_coordinator.wait()` in headless mode

## Key Points

1. `ShutdownCoordinator::wait()` must be called to join threads spawned via `spawn_sdr_thread()`
2. Changed `wait(self)` to `wait(&self)` to work with `Arc<ShutdownCoordinator>`
3. Added `wait()` calls in both TUI and headless shutdown paths

## Related Issues

- `01-shutting-down.md` - Original shutdown investigation
- `02-fix-plan.md` - Previous shutdown fixes (signal handler, TUI exit race, etc.)
- `03-structured-concurrency-implementation.md` - Introduction of ShutdownCoordinator
