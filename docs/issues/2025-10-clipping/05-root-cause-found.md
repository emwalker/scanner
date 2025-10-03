# Root Cause Found - Audio Graph Leak on Pause

## The Bug

In `src/window.rs:445-451`, the `process_signal_for_audio` function has a critical bug:

```rust
// Check if pause requested
if let Some(pause) = pause_signal
    && pause.is_paused()
{
    debug!("Pause requested during audio processing, stopping early");
    break;  // ← BREAKS WITHOUT CANCELLING GRAPH!
}
```

When the loop breaks due to pause, it **skips the graph cancellation code** at lines 454-458:

```rust
debug!("Cancelling audio graph...");
cancel_token.cancel();
debug!("Waiting for audio graph thread to finish...");
let _ = graph_handle.join();
debug!("Audio graph thread finished");
```

## What Happens

1. User is in scan mode, listening to Station A
2. Scan mode calls `process_signal_for_audio` which starts audio graph thread
3. User switches to browse mode → pause signal is set
4. Loop at line 434 detects pause and breaks at line 450
5. **Function returns WITHOUT cancelling the audio graph**
6. Audio graph thread (with BroadcastSource) **keeps running forever**
7. User tunes to Station B in browse mode → creates new SDR segment
8. Old BroadcastSource (from Station A) is still running, trying to receive from old broadcast channel
9. Old BroadcastSink tries to send to old channel with no receivers → sleeps 10ms repeatedly
10. Both old and new graphs interfere → choppy audio

## Evidence from Logs

```
BroadcastSource work() called count=860000  ← From scan mode, never stopped!
Creating NEW SDR segment for 89.2 MHz      ← Browse mode creates new segment
BroadcastSource work() called count=870000  ← Old one still running!
broadcast channel issue (no receivers or full), consuming all samples
```

The BroadcastSource from scan mode has been running for 860,000+ iterations, proving it was never cancelled.

## The Fix

Ensure audio graph is ALWAYS cancelled, even when breaking early:

```rust
// Check if pause requested
if let Some(pause) = pause_signal
    && pause.is_paused()
{
    debug!("Pause requested during audio processing, stopping early");
    break;
}
```

After the while loop, the cancellation code MUST run regardless of why the loop exited (duration expired, shutdown, or pause).

The cancellation code at lines 454-458 is already positioned correctly AFTER the loop. The issue is that it's executed whether we break or complete normally, so this should already work...

Wait, let me re-examine the code structure.

## Re-analysis

Looking at the code again:
```rust
while !remaining.is_zero() {
    // ... sleep and checks ...
    if pause.is_paused() {
        break;
    }
}

debug!("Cancelling audio graph...");
cancel_token.cancel();
// ...
```

Actually, the cancellation SHOULD happen after the break! The code looks correct.

Unless... let me check if there's a return statement somewhere, or if an error causes early return.

## Actual Issue

Wait - look at line 702 in the caller (`play_signals`):

```rust
if let Err(e) = Window::process_signal_for_audio(...) {
    debug!("Error processing signal for audio: {}", e);
}
```

Errors are silently caught! If `process_signal_for_audio` returns an error BEFORE reaching the cancellation code, the audio graph leaks!

But there's no early return in the code I see... unless the issue is different.

Let me check the actual log sequence more carefully. The scan mode should complete normally and cancel its graph. Unless it never completes?

## Critical Discovery - Global Work Counter

The `WORK_CALL_COUNT` in BroadcastSource (src/broadcast.rs:98) is a **STATIC** variable - it's GLOBAL across ALL BroadcastSource instances!

```rust
static WORK_CALL_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
```

This means `count=860000` doesn't prove an old BroadcastSource is still running - it just means ALL BroadcastSources created during the session have collectively called work() 860,000 times.

**This invalidated the "leaked audio graph" theory.**

## Actual Root Cause (Discovered Later)

The real problem was architectural:

1. **Browse mode created a NEW audio stream for EACH station switch**
2. **Scan mode created ONE audio stream and reused it for all signals**

When browse mode created new streams:
- Old stream paused/dropped
- Old audio graph cancelled (takes time)
- During cancellation, SDR segment still alive, broadcast channel fills
- New stream created before old segment dropped → timing gap
- New audio graph starts but old segment not fully cleaned up
- Broadcast channel backup → BroadcastSink sleeps → underruns

The solution was to make browse mode work like scan mode: **persistent audio stream, swap out audio graphs**. See `06-resolution.md` for the complete fix.
