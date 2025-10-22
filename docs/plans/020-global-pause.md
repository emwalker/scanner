# Global Pause Feature Design

## Overview

Add spacebar-triggered global pause functionality to the TUI that stops all scanning and audio processing, reduces CPU to near-zero, and allows resumption to the previous state.

## Requirements

### User Experience
- Spacebar toggles pause/resume from any UI state
- Pause indicator appears in bottom status bar
- Audio stops immediately when paused
- CPU usage drops to near-zero (no busy waits)
- Resume restores previous state (scanning or listening)

### Interaction with Existing Features
- ENTER during global pause: removes pause after station tunes
- Spacebar during listening: pauses audio, spacebar again resumes same station
- Works with multiple concurrent scans (pauses all, resumes all)
- Scan completion during listening: spacebar still pauses/resumes audio

## Architecture

### State Machine Extension

Extend `ScanPauseState` to distinguish user-initiated global pause from system-initiated pause-for-tuning:

```rust
pub enum ScanPauseState {
    Scanning,
    PausedAtWindow { window_num: usize },  // Existing: ENTER pause for tuning
    PausedGlobally {
        at_window: usize,
        previous_state: PreviousPauseState,  // Remember what to resume
    },
    Listening { window_num: usize },
}

pub enum PreviousPauseState {
    WasScanning,
    WasListening { window_num: usize, station_frequency_hz: f64 },
}
```

### State Transitions

```
Scanning --[spacebar]--> PausedGlobally(WasScanning)
PausedGlobally(WasScanning) --[spacebar]--> Scanning

Listening --[spacebar]--> PausedGlobally(WasListening{...})
PausedGlobally(WasListening) --[spacebar]--> Listening

PausedGlobally --[ENTER on station]--> PausedAtWindow --> Listening --> Scanning (auto-resume)
```

### Global Pause Resource (Single Source of Truth)

Use ECS Resource pattern for shared global state:

```rust
// ECS resource (not a component)
pub type GlobalPauseResource = Arc<Mutex<GlobalPauseState>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalPauseState {
    Active,
    Paused {
        had_active_scans: bool,
        had_active_audio: bool,
    },
}
```

Integration:
- Coordinator creates and owns the resource
- Passes to TUI and SystemContext
- TUI queries for display and handles spacebar
- Systems check before processing

### Signal Processing Layer

**Critical: No busy waits**

When pausing:
1. Cancel all rustradio graph threads via `CancellationToken`
2. Join threads (wait for clean exit from `graph.run()` loop)
3. Drop graph and thread handles
4. Remember configuration for resume

When resuming:
1. Recreate graphs from saved configuration
2. Spawn fresh threads
3. Graphs process normally

**Why not suspend threads?** The `rustradio::graph::Graph::run()` loop is a tight polling loop. Keeping it running while paused would busy-wait at high CPU. Research confirms the pattern: cancel → join → recreate.

### Thread Management

**Ephemeral threads (graph processing):**
- Managed by AudioSession, WindowEntity, AudioEntity
- NOT tracked by ShutdownCoordinator
- Can be stopped/started for pause/resume

**Persistent threads (SDR hardware I/O):**
- Tracked by ShutdownCoordinator
- Run until process shutdown
- Cannot be paused (hardware requires continuous polling)

### Shutdown Safety

**During pause (no graph threads exist):**
- ShutdownCoordinator only joins persistent threads
- No risk of double-join

**During active (graph threads running):**
- Graph threads not in coordinator tracking
- Managed separately by AudioSession
- Cancel + join happens in Drop or on explicit stop

**Shutdown from paused state:**
- No graph threads to clean up
- Persistent threads cancel + join normally

**Shutdown from active state:**
- AudioSession cancels graphs in Drop
- ShutdownCoordinator handles persistent threads
- No overlap, no deadlocks

### UI State and Display

**Model queries resource:**
```rust
impl Model {
    pub fn is_globally_paused(&self) -> bool {
        self.global_pause_resource
            .lock()
            .map(|state| matches!(*state, GlobalPauseState::Paused { .. }))
            .unwrap_or(false)
    }
}
```

**Status bar shows indicator:**
```rust
if model.is_globally_paused() {
    spans.push(Span::styled(
        "[PAUSED] ",
        Style::default()
            .fg(theme.active_highlight_fg())
            .add_modifier(Modifier::BOLD)
    ));
}
```

**Spacebar handler:**
```rust
KeyCode::Char(' ') => {
    let mut state = self.global_pause_resource.lock()?;

    match *state {
        GlobalPauseState::Active => {
            *state = GlobalPauseState::Paused { ... };
            self.pause_all_scans()?;
            self.pause_all_audio()?;
        }
        GlobalPauseState::Paused { .. } => {
            *state = GlobalPauseState::Active;
            self.resume_all_scans()?;
            self.resume_all_audio()?;
        }
    }
}
```

### Multi-Scan Coordination

When multiple scans are active:
- Spacebar pause transitions ALL scans to `PausedGlobally`
- Stops ALL audio entities
- Single pause indicator in UI
- Spacebar resume restores ALL previous states

Implementation iterates over all entity worlds:
```rust
fn pause_all_scans(&mut self) -> Result<()> {
    let entities = self.scan_entities.lock()?;
    for scan in entities.iter_mut() {
        if scan.is_scanning() || scan.is_listening() {
            let previous = capture_state(scan);
            scan.transition_to_paused_globally(previous);
        }
    }
}
```

## Implementation Locations

### New Files
- `src/ecs/resources/global_pause.rs` - GlobalPauseState and resource type
- `src/ecs/components/scan/pause_request.rs` - Add `Global` variant to PauseRequest

### Modified Files
- `src/ecs/components/scan/progress.rs` - Add `PausedGlobally` variant
- `src/ecs/coordinator.rs` - Create and expose GlobalPauseResource
- `src/ecs/system.rs` - Add global_pause to SystemContext
- `src/ui/tui/mod.rs` - Add spacebar handler, integrate resource
- `src/ui/tui/model/types.rs` - Add is_globally_paused() method
- `src/ui/tui/renderers/instructions.rs` - Show pause indicator
- `src/audio/session.rs` - Track graph handles separately from coordinator
- `src/ecs/systems/scan/request_processor.rs` - Handle Global pause requests

## Testing Strategy

### Unit Tests
- GlobalPauseState transitions (Active ↔ Paused)
- ScanPauseState with PausedGlobally variant
- PreviousPauseState serialization

### Integration Tests
- Spacebar toggles pause/resume
- ENTER during pause (transitions correctly)
- Pause during listening (audio stops/resumes)
- Multi-scan pause (all pause, all resume)
- Shutdown from paused (no deadlocks)
- Shutdown from active (clean thread join)

### Regression Tests
- CPU usage drops to near-zero when paused (verify with `top`)
- No busy waits (profile with perf/flamegraph)
- Pause indicator visibility
- Audio stops immediately
- Graph threads fully exit (not sleeping)

## Design Validation

### Research Validation
- ECS resource pattern validated against ecs-design skill
- Thread cancellation pattern confirmed via Rust community best practices
- "Join your threads" principle followed (no thread suspension, only cancel + recreate)
- Single source of truth maintained (GlobalPauseResource)

### Shutdown Safety
- Separate ephemeral (pauseable) vs persistent (hardware) threads
- No double-join risk (graph threads not in coordinator tracking)
- All cancellation tokens properly propagated
- Drop implementations handle cleanup correctly

### CPU Efficiency
- Tight `graph.run()` loop cannot be paused without busy wait
- Solution: cancel threads completely, recreate on resume
- Research confirms: "break the loop and terminate, recreate when needed"

## Open Questions

None - design is complete and validated.

## Success Criteria

- Spacebar pauses all activity (scanning + audio)
- CPU usage < 1% when paused
- Resume restores exact previous state
- Pause indicator visible in UI
- No deadlocks during shutdown from any state
- ENTER during pause works as expected (removes pause after tune)
- All 500+ existing tests continue to pass
