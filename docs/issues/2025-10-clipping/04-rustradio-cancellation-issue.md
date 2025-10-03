# RustRadio Cancellation Issue - Root Cause Found

## How RustRadio Cancellation Actually Works

From `/home/walker/code/rustradio/src/graph.rs:118-123`:

```rust
loop {
    let mut done = true;
    let mut all_idle = true;
    if self.cancel_token.is_canceled() {  // ← Checked ONLY here
        break;
    }
    for (n, b) in self.blocks.iter_mut().enumerate() {
        // ... call b.work() ...
    }
}
```

**Critical insight**: The cancellation token is checked **only at the start of each loop iteration**, NOT during block execution.

## The Problem with BroadcastSink

Our BroadcastSink::work() implementation (src/broadcast.rs:31-57):

```rust
fn work(&mut self) -> Result<BlockRet<'_>> {
    let (input_buf, _metadata) = self.input.read_buf()?;
    let samples = input_buf.slice();

    let mut sent = 0;
    for sample in samples {
        match self.sender.send(*sample) {
            Ok(_) => sent += 1,
            Err(_) => {
                sent = samples.len();
                debug!("broadcast channel issue (no receivers or full), consuming all samples");
                std::thread::sleep(std::time::Duration::from_millis(10));  // ← BLOCKS HERE!
                break;
            }
        }
    }

    input_buf.consume(sent);
    Ok(BlockRet::Again)
}
```

**The issue**:
1. BroadcastSink starts work() method
2. Tries to send samples, gets error (channel full or no receivers)
3. **Sleeps for 10ms** (line 46)
4. During this sleep, cancellation token is set
5. work() returns `BlockRet::Again`
6. **Next loop iteration checks cancellation and exits**
7. But the BroadcastSink has already sent its "broadcast channel issue" message!

## Why This Causes the Log Pattern

From logs:
```
SDR graph thread finished
SDR segment dropped, broadcast channel closed
broadcast channel issue (no receivers or full), consuming all samples  ← After everything!
```

**What's happening**:
1. SDR segment Drop called → `stop()` called → `cancel_token.cancel()`
2. Graph's run() loop is executing → currently in BroadcastSink.work()
3. BroadcastSink tries to send → fails → logs message → sleeps 10ms
4. Cancellation token is now cancelled, but work() hasn't returned yet
5. work() returns after sleep
6. **Next loop iteration sees cancellation and breaks**
7. Thread exits, logs "SDR graph thread finished"
8. But we already saw the "broadcast channel issue" message from step 3!

## The Real Root Cause

**The "broadcast channel issue" messages appearing after shutdown are NOT from a still-running graph.**

They're from the **final work() call** that was in-progress when cancellation occurred. The message appears in logs, then work() returns, then the next loop iteration sees the cancellation and exits.

## Why This Causes Choppy Audio

The 10ms sleep in BroadcastSink happens when:
- Channel is full (multiple receivers backing up)
- No receivers exist (during transitions)

During rapid station switching in browse mode:
1. Old audio graph still consuming from channel
2. New audio graph starts consuming from channel
3. Both receivers lagging → channel fills up
4. BroadcastSink keeps hitting "channel full" → sleeps 10ms each time
5. During these sleeps, few samples reach audio output → **underruns**
6. Audio output plays silence or repeats samples → **choppy/stuttering audio**

The issue isn't about blocks running after cancellation - it's about blocks **sleeping during work()** when they encounter errors, which starves the audio output.

## Analysis Was Partially Correct

The BroadcastSink sleep when it can't send is real, but the root cause analysis was wrong. The issue wasn't about multiple graphs consuming simultaneously.

**The actual problem** (discovered later):

1. Browse mode created a **NEW audio stream for EACH station switch**
2. Each new stream meant NEW audio_tx/audio_rx channel
3. Creating new streams has overhead and timing gaps
4. During audio graph cancellation (which takes time), the SDR segment was still alive
5. Broadcast channel would fill during this gap → BroadcastSink sleeps → delays

The "broadcast channel issue" messages were from the **current** BroadcastSink hitting a full channel during the timing gap, not from an old leaked graph.

## Actual Solution

Make browse mode work like scan mode: **persistent audio stream, swap out audio graphs**.

Created `AudioSession` that owns:
- One audio stream (persists across station switches)
- One audio_tx/audio_rx (reused for all stations)
- Current audio graph (swapped out on each switch)
- Current SDR segment (kept alive while station plays)

See `06-resolution.md` for complete details.
