# Stream Lifecycle Fix - Analysis of Failure

## What We Tried

Added explicit `stream.pause()` and `drop(stream)` before audio graph cleanup in `play_single_signal_with_receiver`:

```rust
debug!("Stopping audio stream before cleanup");
stream.pause()?;
drop(stream);
debug!("Audio stream stopped and dropped");

cancel_token.cancel();
let _ = graph_handle.join();
```

## Result: Did Not Fix The Issue

Choppy audio and underruns continue in browse mode.

## New Discovery from Logs

The most revealing finding: **"broadcast channel issue" messages continue AFTER the SDR segment is dropped!**

```
Stopping audio stream before cleanup
Audio stream stopped and dropped
broadcast channel issue (no receivers or full), consuming all samples
Audio graph completed successfully
BroadcastSource dropped - receiver being cleaned up [3mdrop_count[0m[2m=[0m9
Audio graph dropped, resources released
SDR SEGMENT DROPPING: Stopping SDR graph and closing broadcast channel [3mreceiver_count[0m[2m=[0m0
Cancelling SDR graph
Waiting for SDR graph thread to finish
SDR graph thread exited
SDR graph thread finished
SDR segment dropped, broadcast channel closed
broadcast channel issue (no receivers or full), consuming all samples  <-- AFTER EVERYTHING IS DROPPED!
```

## The Real Problem

**A BroadcastSink from a PREVIOUS SDR segment is still running and trying to send samples even after:**
1. The audio stream is stopped and dropped
2. The audio graph is cancelled and cleaned up
3. The SDR segment is dropped
4. The SDR graph thread has exited
5. The broadcast channel is closed

## How This Is Possible

When switching stations in browse mode:
1. **New SDR segment is created** (because center frequency may change)
2. Old SDR segment is dropped (triggers `Drop` impl)
3. Old SDR graph is cancelled (`cancel_token.cancel()`)
4. Old SDR graph thread exits (`graph.run()` returns)
5. **But somehow a BroadcastSink is still active!**

## Hypothesis

The issue may be:

### Option A: Graph Blocks Not Stopping
The rustradio graph's `cancel_token` cancels the graph, but individual blocks (like BroadcastSink) might not stop immediately. The BroadcastSink's `work()` method continues to be called even after cancellation.

### Option B: Multiple SDR Segments Overlapping
When rapidly switching stations, a new SDR segment is created before the old one is fully cleaned up. Multiple SDR graphs from different segments may be running simultaneously.

### Option C: Thread Synchronization Issue
The `graph_handle.join()` waits for the thread, but the thread's cleanup might not be synchronous. The graph and its blocks might still be active for a brief period after the thread "exits".

## Evidence from Logs

1. **"Selected candidate info" appears multiple times** - UI is sending rapid station switch commands
2. **Underruns start before any station switch** - The problem begins during normal playback
3. **"broadcast channel issue" persists after everything drops** - Something is fundamentally wrong with graph lifecycle

## What We Learned

The stream lifecycle fix (pause + drop) didn't help because the real issue was:

1. **Creating a NEW stream for each station** instead of reusing one stream
2. **Timing gaps** during the transition from old stream to new stream
3. **SDR segment staying alive** during audio graph cancellation, causing broadcast channel backup

## Actual Solution (Found Later)

The fix was architectural: create `AudioSession` with a **persistent audio stream** that stays alive for the entire browse session, just like scan mode does.

See `06-resolution.md` for the complete solution.
