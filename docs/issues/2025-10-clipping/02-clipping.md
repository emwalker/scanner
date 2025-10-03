# Revised Root Cause Analysis - Audio Interference Issue

## What We Discovered

The original hypothesis in `clipping.md` was **incorrect**. The buffered samples (`receiver_len`) were a symptom, not the cause.

## The Real Problem

**Multiple audio graphs are simultaneously consuming from the same broadcast channel, causing audio interference.**

### Evidence

1. **Wrong station audio playing**: User hears audio from station 90.1 MHz even when window is on 88.9 MHz
2. **Multiple receivers on same channel**: Logs show `sender_receiver_count` incrementing: 1 → 2 → 3 → 4 → 5 → 7 → 9
3. **Receiver count never decreases during session**: 42 BroadcastSources were dropped, but new ones kept being created
4. **Resubscribe fix didn't help**: Even with `receiver_len=0`, choppy audio persists

### The Architecture Flaw

```
SDR Segment (one SoapySdrManager)
└── Broadcast Channel (524K capacity)
    ├── Receiver 1 (Window 1 audio graph) ← May still be running
    ├── Receiver 2 (Window 2 audio graph) ← May still be running
    ├── Receiver 3 (Browse mode audio graph) ← NEW
    └── Receiver N (Another browse switch) ← NEWER
```

**Problem**: When switching stations in browse mode:
1. New audio graph creates new broadcast receiver
2. Old audio graph's receiver may still be active (not properly cancelled)
3. Both receivers compete for samples from the same channel
4. Tokio broadcast channel sends samples to ALL active receivers
5. Multiple audio outputs play simultaneously → interference/choppy audio

### Why Scan Mode Works

During scan mode:
- Each window plays for 3 seconds
- Audio graph is created and destroyed cleanly
- No overlapping audio graphs
- `sender_receiver_count=1` during playback

### Why Browse Mode Fails

During browse mode station switching:
- User rapidly switches between stations
- New audio graph created before old one fully stops
- Old receiver still exists in broadcast channel
- Multiple receivers consume samples → audio interference

## Why Previous Hypotheses Failed

### Hypothesis 1: Stale Buffer Clipping (REFUTED)
- Resubscribing gave `receiver_len=0` but audio still choppy
- Not about buffered samples at all

### Hypothesis 2: Excessive FM Gain (PARTIALLY TRUE BUT NOT MAIN ISSUE)
- 1.2x quality boost does cause minor clipping
- But doesn't explain wrong station audio or interference

### Hypothesis 3: Buffered IQ Samples (REFUTED)
- Correlation between `receiver_len>0` and choppy audio was coincidence
- Real issue is multiple active receivers, not buffered samples

## Root Cause

**Lack of exclusive audio graph ownership**

When switching stations in browse mode:
1. Old audio graph task continues running (not awaited/cancelled properly)
2. Old BroadcastSource receiver still active in broadcast channel
3. New audio graph creates new receiver on SAME channel
4. Both audio graphs try to play simultaneously
5. Audio interference causes choppy/stuttering and wrong station audio

## The "broadcast channel issue" Messages

These messages occur when `sender.send()` fails. This happens when:
- Channel is full (524K samples buffered)
- OR no receivers exist

In our case, it's the channel full condition because multiple slow receivers cause backlog.

## The Underrun Pattern

```
setup_audio_graph_source called receiver_len=0
AUDIO UNDERRUN: missing_samples=8192 (completely empty)
broadcast channel issue (no receivers or full)
AUDIO UNDERRUN: missing_samples=4095 (still mostly empty)
```

This happens because:
1. New audio graph starts
2. But old audio graph's receiver is consuming samples
3. New receiver gets starved → underruns
4. Sender can't send (channel full from backlog) → "broadcast channel issue"

## Solution Direction

The fix is NOT about buffer management. It's about ensuring:

1. **Exclusive audio output**: Only ONE audio graph can play at a time
2. **Proper cleanup**: Old audio graph must be fully stopped before new one starts
3. **Atomic switching**: Switching stations must be atomic (stop old, start new)

## Potential Fixes

### Option 1: Serial Audio Graph Execution
Ensure old audio graph task is fully cancelled and awaited before starting new one.

### Option 2: Single Audio Graph Instance
Reuse the same audio graph and just change its parameters when switching stations.

### Option 3: Use FanoutSink Pattern
The `src/fanout.rs` file implements a pattern where dropped subscribers automatically stop receiving samples. This could replace the broadcast channel approach.

### Option 4: Audio Graph Registry
Track active audio graphs and ensure only one is active per segment.

## Files Involved

- `src/broadcast.rs`: BroadcastSink/BroadcastSource using Tokio broadcast channel
- `src/soapy.rs:90`: Broadcast channel creation (524K capacity)
- `src/window.rs:468-488`: Audio graph setup with resubscribe (ineffective fix)
- `src/window.rs:786-889`: play_single_signal_with_receiver (audio graph lifecycle)
- `src/fanout.rs`: Alternative fanout pattern (not currently used)

## Update: Analysis Was Incorrect

This entire analysis was on the wrong track. The "multiple receivers" issue was a red herring.

**The actual root cause** (discovered later):

Browse mode was **creating a NEW audio stream for each station switch**, while scan mode **reused one audio stream** for all signals.

The problems this caused:
1. **Timing gaps** during stream creation/destruction
2. **Broadcast channel backup** when SDR segment stayed alive during audio graph cancellation
3. **SDR segment dropping too early** (before audio graph finished using it)
4. **Audio underruns** from the gaps and delays

**The fix**: Create `AudioSession` with a persistent audio stream that stays alive for the entire browse session. The segment is stored in the session to ensure it stays alive while the audio graph is using it.

This matched scan mode's architecture: one stream, swap out audio graphs sequentially.

See `06-resolution.md` for the complete solution.
