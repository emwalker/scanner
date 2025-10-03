# Audio Clipping Investigation

## Problem Statement

Users experience severe audio distortion (choppy/stuttering) when switching between stations in browse mode. The issue is nondeterministic - sometimes audio sounds fine, sometimes it's severely distorted.

**User Experience:**
- **Scan mode**: Audio sounds fine, no noticeable distortion
- **Browse mode**: Nondeterministic severe choppy/stuttering audio
- **Correlation**: Distortion seemed to correlate with "broadcast channel issue" messages

## Investigation Process

### Initial Hypothesis: Stale Buffer Clipping (REFUTED)

**Hypothesis**: When switching stations in browse mode, stale IQ samples from the previous station remain buffered. When the new demodulator processes these samples (tuned to the wrong frequency), they produce excessive audio levels causing clipping.

**Evidence Against**:
- Both scan and browse modes showed samples >1.0 in logs
- Scan mode had MORE clipping events than browse mode in some cases
- Sessions with `quality_adjustment=1.0` still had clipping in logs
- User reported scan mode sounded fine despite logged clipping events

**Conclusion**: The "AUDIO CLIPPING DETECTED" messages in logs don't correlate with the severe audible distortion user experienced.

### Second Hypothesis: Excessive FM Demodulator Gain (PARTIALLY CONFIRMED)

**Hypothesis**: The `quality_adjustment=1.2` multiplier for "Moderate" audio quality stations is too aggressive, causing audio peaks to exceed ±1.0.

**Evidence**:
```
Line 539 in src/window.rs:
AudioQuality::Moderate => 1.2,  // 20% boost
```

**Correlation found**:
- All stations with `quality_adjustment=1.2` showed clipping in logs
- Stations with `quality_adjustment=1.0` showed no clipping in logs
- Diagnostic logging confirmed samples exceeded safe threshold (±0.833) before output

**Example from diagnostics**:
```
GAIN DIAGNOSTICS: Audio levels vs quality boost
  quality_adjustment = 1.2
  safe_threshold = 0.833 (= 1.0 / 1.2)
  max_audio_sample = 1.051
  exceeds_threshold_count = 16
  would_clip_after_boost = true
```

**Conclusion**: The 1.2x quality boost does cause minor clipping (samples >1.0 get clamped), but this is **NOT** the severe distortion user experienced. This is a separate, less severe issue.

### Final Hypothesis: Stale IQ Buffer Causes Choppy Audio (CONFIRMED)

**Hypothesis**: When switching stations in browse mode with buffered IQ samples in the broadcast channel, the audio pipeline becomes unstable, causing choppy/stuttering audio.

**Key Evidence**:

| Session | Mode | receiver_len | Underruns | Broadcast Issues | Audio Quality |
|---------|------|--------------|-----------|------------------|---------------|
| 89.7 MHz | Browse | **6,961** | 5 | 2 | Choppy/stuttering |
| 90.1 MHz | Browse | **62,839** | 36 | 27 | Choppy/stuttering |
| 88.9 MHz | Browse | 0 | 9 | 1 | Fine |
| 89.7 MHz | Browse | 0 | 7 | 1 | Fine |
| All scan mode | Scan | 0 | ~2-5 | 0-1 | Fine |

**What happens with buffered samples:**

Example from 90.1 MHz with 62,839 buffered samples:
```
setup_audio_graph_source: receiver_len=62839
AUDIO UNDERRUN: filled_samples=0, missing_samples=8192
broadcast channel issue (no receivers or full), consuming all samples
AUDIO UNDERRUN: filled_samples=1892, missing_samples=2203
broadcast channel issue (no receivers or full), consuming all samples
AUDIO UNDERRUN: filled_samples=1240, missing_samples=808
broadcast channel issue (no receivers or full), consuming all samples
AUDIO UNDERRUN: filled_samples=0, missing_samples=2048
broadcast channel issue (no receivers or full), consuming all samples
AUDIO UNDERRUN: filled_samples=1573, missing_samples=475
...
```

Audio samples arrive in **irregular bursts**: 1892 → 0 → 1240 → 0 → 1573 → 422 → 1150...

This creates the choppy/stuttering effect.

**Why it's nondeterministic:**
- Depends on timing of when user switches stations
- If switch happens while old SDR graph is still producing samples → large buffer → choppy audio
- If switch happens after old SDR graph stops → `receiver_len=0` → clean audio

**Why scan mode is unaffected:**
- Each station plays for only 3 seconds
- SDR graph is stable during playback (no switching)
- All scan sessions had `receiver_len=0`

## Root Cause

When switching stations in browse mode:

1. **Old SDR graph shutdown overlaps with new SDR graph startup**
2. **IQ samples from old graph accumulate in broadcast channel** (e.g., 6,961 or 62,839 samples)
3. **New audio graph starts with this backlog** (`setup_audio_graph_source: receiver_len=62839`)
4. **Pipeline can't keep up** with processing backlog while new samples arrive
5. **Result**: Constant underruns, irregular sample delivery, choppy/stuttering audio

The "broadcast channel issue" messages are a symptom, not the cause - they indicate the broadcast channel is full or has no receivers during the SDR graph transition.

## Diagnostic Logging Added

### 1. Buffer State Tracking (src/broadcast.rs)

**BroadcastSource initial buffer logging**:
```rust
if !INITIAL_BUFFER_LOGGED.load(...) {
    let initial_buffer_len = self.receiver.len();
    if initial_buffer_len > 0 {
        debug!(
            initial_buffered_samples = initial_buffer_len,
            "BUFFER DIAGNOSTICS: BroadcastSource starting with buffered samples"
        );
    }
}
```

**Buffer drain monitoring**:
```rust
if count.is_multiple_of(5000) {
    let remaining_buffer = self.receiver.len();
    if remaining_buffer > 1000 {
        debug!(
            work_count = count,
            remaining_buffered = remaining_buffer,
            "BUFFER DIAGNOSTICS: Still draining buffer"
        );
    }
}
```

### 2. Audio Level Diagnostics (src/broadcast.rs)

**AudioDiagnostic block** measures audio sample magnitudes relative to quality adjustment:
```rust
let safe_threshold = 1.0 / quality_adjustment;
// For quality_adjustment=1.2, safe_threshold=0.833

debug!(
    sample_count = count + samples.len(),
    max_audio_sample = max_val,
    min_audio_sample = min_val,
    quality_adjustment = self.quality_adjustment,
    safe_threshold = safe_threshold,
    exceeds_threshold_count = exceeds_threshold_count,
    would_clip_after_boost = would_clip_after_boost,
    "GAIN DIAGNOSTICS: Audio levels vs quality boost"
);
```

### 3. Clipping Detection (src/window.rs)

**Output stage clipping detection**:
```rust
if audio_sample > 1.0 || audio_sample < -1.0 {
    clipped_count += 1;
}
max_sample = max_sample.max(audio_sample);
min_sample = min_sample.min(audio_sample);

if clipped_count > 0 {
    debug!(
        clipped_samples = clipped_count,
        total_samples = filled,
        max_sample = max_sample,
        min_sample = min_sample,
        clip_percentage = (clipped_count as f32 / filled as f32) * 100.0,
        "AUDIO CLIPPING DETECTED: Samples exceeded ±1.0 range"
    );
}
```

## Key Findings

### Finding 1: Two Distinct Audio Issues

1. **Minor clipping** (from 1.2x quality boost):
   - Affects all "Moderate" quality stations
   - Sample values exceed ±1.0
   - Gets clamped at output
   - **Not the severe distortion user experienced**

2. **Severe choppy/stuttering** (from buffered IQ samples):
   - Only in browse mode, nondeterministic
   - Caused by pipeline instability during SDR graph transitions
   - Creates irregular audio delivery
   - **This is the actual problem user reported**

### Finding 2: "broadcast channel issue" Messages

These messages are **symptoms, not the cause**:
- Appear when broadcast channel temporarily has no receivers
- Common during SDR graph startup/shutdown transitions
- Presence doesn't guarantee choppy audio (e.g., 88.9 MHz had 1 issue but sounded fine)
- Correlation is actually with `receiver_len`, not the messages themselves

### Finding 3: receiver_len is the Key Indicator

The `setup_audio_graph_source: receiver_len=N` value indicates:
- `receiver_len=0`: Clean start, audio will sound fine
- `receiver_len>0`: Buffered samples present, high risk of choppy audio
- Higher values (62,839) → worse distortion than lower values (6,961)
