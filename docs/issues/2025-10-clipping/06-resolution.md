# Final Solution - AudioSession for Browse Mode

## The Root Cause

The choppy/stuttering audio in browse mode was caused by **creating a new audio stream for each station switch**, while scan mode **reused a single audio stream** for all signals.

### Why This Caused Problems

When switching stations in browse mode (old implementation):

1. **Old audio infrastructure created**:
   - Audio stream + audio_tx/audio_rx channel
   - Audio graph consuming from SDR segment
   - SDR segment producing samples

2. **User switches stations** → command received

3. **Old infrastructure torn down**:
   - Stream paused and dropped
   - Audio graph cancelled and joined
   - Function returns → **SDR segment dropped**

4. **NEW infrastructure created immediately**:
   - NEW audio stream + NEW audio_tx/audio_rx
   - NEW SDR segment
   - NEW audio graph

5. **Critical timing issue**:
   - Step 3 (cancelling audio graph) takes time
   - During this time, SDR graph is still running
   - Audio graph already stopped consuming from broadcast channel
   - Broadcast channel fills to capacity (524K samples)
   - BroadcastSink hits "channel full" → sleeps 10ms
   - Audio underruns occur during these delays

## The Solution: AudioSession

Created a persistent audio session for browse mode that mirrors scan mode's architecture:

```rust
pub struct AudioSession {
    audio_tx: std::sync::mpsc::SyncSender<f32>,
    _stream: cpal::Stream,                              // Persistent!
    current_graph: Option<GraphHandle>,
    current_segment: Option<Box<dyn crate::sdr::Segment>>,
}
```

**Naming rationale**: `AudioSession` is an industry-standard name used across major platforms (Apple's `AVAudioSession`, Android's `MediaSession`, Windows' `MediaPlaybackSession`) for managing audio playback state and resources.

### Key Design Points

1. **Persistent audio stream**: Created once when entering browse mode, dropped when exiting
2. **Persistent audio_tx/audio_rx**: Same channel used for all stations
3. **Sequential audio graphs**: Each station switch cancels old graph, starts new one with same audio_tx
4. **Proper segment ownership**: Segment stored in BrowseContext, kept alive while station plays

### Station Switch Flow (New Implementation)

1. **Enter browse mode** (Pause command):
   ```rust
   audio_session = Some(AudioSession::new(&config));
   ```
   - Creates audio stream
   - Stream starts playing immediately
   - Stored for entire browse session

2. **Tune to station** (TuneToCandidate command):
   ```rust
   audio_session.tune_to_station(&signal, segment, &config);
   ```
   - Cancels old audio graph (if any)
   - Drops old segment (if any)
   - Creates new segment
   - Subscribes to new segment's broadcast channel
   - Creates new audio graph with **same audio_tx**
   - Stores segment (keeps it alive)

3. **Exit browse mode** (ResumeScan command):
   ```rust
   audio_session = None;  // Drops AudioSession
   ```
   - Cancels audio graph
   - Drops segment
   - Drops audio stream

## Why This Works

### Same Architecture as Scan Mode

**Scan mode** (always worked):
```rust
// Create stream once
let (audio_tx, audio_rx) = sync_channel(...);
let stream = create_audio_stream(..., audio_rx);
stream.play();

// For each signal
for signal in signals {
    let sdr_rx = segment.audio_subscriber();
    process_signal_for_audio(signal, sdr_rx, audio_tx.clone(), ...);
    // Graph runs, completes, next signal starts
}
```

**Browse mode** (now fixed):
```rust
// Create AudioSession with stream
let audio_session = AudioSession::new(&config);
// stream + audio_tx created and stored

// For each station switch
audio_session.tune_to_station(&signal, segment, &config);
// Cancels old graph, creates new graph with same audio_tx
```

### No Timing Gaps

- Audio stream is **already running** when new audio graph starts
- New audio graph immediately starts producing samples to existing audio_tx
- No gap where audio_rx is empty
- No gap where broadcast channel backs up

### Proper Cleanup Order

1. Cancel audio graph → stops consuming from broadcast channel
2. Join audio graph thread → waits for complete shutdown
3. Drop segment → stops SDR graph, closes broadcast channel
4. Start new graph → fresh start with new channel

The old implementation had segment dropping happening in the middle of audio graph cancellation, causing race conditions.

## Files Modified

### New Files
- `src/audio_session.rs`: New AudioSession struct managing persistent audio infrastructure

### Modified Files
- `src/lib.rs`: Added audio_session module
- `src/main_thread.rs`:
  - Updated `handle_command` to create/manage AudioSession
  - Updated `handle_tune_command` to use AudioSession
  - Added audio_session parameter to all handle_command calls
- `src/window.rs`:
  - Made `create_audio_fm_graph` public (needed by AudioSession)
  - Removed `play_frequency` and `play_single_signal_with_receiver` (no longer needed)

## Lessons Learned

1. **Audio stream creation has overhead** - creating new streams for each switch caused delays and timing issues

2. **Segment lifetime is critical** - if segment drops while audio graph is using it, the broadcast channel closes and everything breaks

3. **Architecture matters** - browse mode needed to match scan mode's architecture, not reinvent it

4. **Persistent resources should be in a session object** - AudioSession encapsulates the lifetime of browse mode resources, following industry-standard naming conventions

5. **The simplest solution is often the best** - instead of trying to fix timing issues with synchronization, we eliminated the timing issues by reusing the audio stream
