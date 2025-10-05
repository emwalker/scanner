# TUI State Machine Design

## Problem (RESOLVED)

The TUI model previously used boolean flags for state:
- `selection_mode: bool`
- `browsing_mode: bool`
- `pending_tune: bool`
- `playback_active: bool`

This created **16 possible combinations** (2^4), but only a few were valid. The compiler could not enforce which combinations were legal.

## Solution Implemented

We replaced the boolean flags with an enum-based state machine:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum UiMode {
    /// Watching scan progress (no candidate selected)
    Idle,

    /// Candidate selected, navigating scanner results while scan may still be running
    NavigatingScanner { selected_index: usize },

    /// Scan paused, waiting for Paused event before tuning to station
    AwaitingTune {
        navigation_index: usize,
        tuning_index: usize,
    },

    /// Actively listening to a station (scan paused, audio playing)
    Listening {
        navigation_index: usize,
        playing_index: usize,
        playing_candidate_id: String,
    },
}
```

### Key Design Decision: Separate Navigation and Playback

The critical insight was that **navigation** (where the cursor/selection is) and **playback** (which station is highlighted/playing) are two separate concerns:

- **navigation_index**: Follows arrow keys, shows which station the user is browsing
- **tuning_index/playing_index**: Stays fixed on the station being tuned/played, controls highlighting

This separation prevents the "highlight follows arrow keys" bug where the highlight would incorrectly move when navigating while a station was playing.

## Migration Completed

### Phase 1: ✅ Added UiMode enum
- Created enum with Idle, NavigatingScanner, AwaitingTune, Listening states
- Kept working alongside existing booleans initially

### Phase 2: ✅ Switched to UiMode
- Updated all rendering logic to use UiMode
- Created computed properties for backward compatibility:
  - `selection_mode()` - derived from UiMode
  - `browsing_mode()` - derived from UiMode
  - `selected_candidate_index()` - derived from UiMode

### Phase 3: ✅ Removed boolean flags
- Deleted `selection_mode`, `browsing_mode`, `pending_tune` fields
- Made them computed properties instead
- Updated all state transitions to use UiMode

### Phase 4: ✅ Separated Navigation and Playback
- Split `selected_index` into two fields in AwaitingTune/Listening modes:
  - `navigation_index` - updated by arrow keys
  - `tuning_index`/`playing_index` - fixed on playing station
- This prevents race conditions and ensures correct highlighting

## State Transitions

```
Idle
  └─> [UP/DOWN arrow] ─> NavigatingScanner
                              │
                              └─> [ENTER] ─> AwaitingTune
                                              │
                                              └─> [AudioPlaybackStarted] ─> Listening
                                                                              │
                                                                              ├─> [ENTER on different station] ─> AwaitingTune
                                                                              │
                                                                              └─> [Resume scan] ─> Idle
```

### State Machine Enforcement

The state machine enforces correct behavior at compile time:

1. **ENTER in NavigatingScanner**: Only allowed when not in browsing mode
   - Transitions to `AwaitingTune` with both indices set to selected station

2. **ENTER in Listening**: Only allowed when in Listening mode (not AwaitingTune)
   - Prevents rapid-fire tune commands that could cause SIGSEGV
   - Transitions to `AwaitingTune` for new station

3. **Arrow keys**: Update `navigation_index` in all modes
   - In NavigatingScanner: Updates `selected_index`
   - In AwaitingTune/Listening: Updates `navigation_index` but preserves `tuning_index`/`playing_index`

## Benefits Achieved

1. ✅ **Compiler enforcement** - Invalid states are impossible
2. ✅ **Exhaustive matching** - Pattern matching ensures all cases are handled
3. ✅ **Clear intent** - Mode names document what the UI is doing
4. ✅ **Easier testing** - Can test specific mode transitions
5. ✅ **Prevents race conditions** - State machine prevents rapid station switching
6. ✅ **Correct highlighting** - Separation of navigation and playback indices

## Bugs Fixed

1. **Wrong station highlighted** - Fixed by using UiMode instead of booleans
2. **Highlight follows arrow keys while playing** - Fixed by separating navigation_index from playing_index
3. **SIGSEGV on rapid station switching** - Fixed by state machine preventing ENTER in AwaitingTune mode
4. **Delayed highlight on ENTER** - Fixed by adding immediate redraw after state transitions

## Testing Strategy

Comprehensive regression tests added:

```rust
#[test]
fn test_browsing_mode_only_true_when_scan_paused()
#[test]
fn test_enter_key_tunes_to_selected_station()
#[test]
fn test_navigation_and_highlight_separate_in_listening_mode()
```

Tests verify:
- State transitions work correctly
- Navigation and playback indices are independent
- Highlighting stays on playing station while arrow keys move navigation
- Race conditions are prevented

## Code Organization

All state management follows The Elm Architecture:
- **Model** (`model.rs`): All state and business logic
- **View** (`renderers/`): Pure rendering functions
- **Update** (`update()` methods): State transitions based on events

See `src/terminal/tui/CLAUDE.md` for architectural details.
