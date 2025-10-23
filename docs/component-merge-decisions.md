# Component Merge Decisions: Station to Signal

Analysis of component differences between StationEntity and SignalEntity components.

## InfoComponent Comparison

**StationInfoComponent** (`src/ecs/components/station/info.rs`):
- Fields: `frequency`, `signal_strength` (f32), `audio_quality`, `name`
- Methods: `new()`, `update_signal_strength()`, `update_audio_quality()`, `set_name()`

**SignalInfoComponent** (`src/ecs/components/signal/mod.rs`, lines 36-422):
- Fields: `frequency_hz`, `signal_strength` (Option<f64>), `audio_quality`, `name`
- Methods: `new()`, `frequency()`, `signal_strength()`, `set_signal_strength()`, `audio_quality()`, `set_audio_quality()`, `name()`, `set_name()`

**Decision**: Keep SignalInfoComponent (superior implementation)
- Better API with getter methods
- Uses Option<f64> for signal strength (more precise, handles missing data)
- More complete method set

## HistoryComponent Comparison

**StationHistoryComponent** (`src/ecs/components/station/history.rs`):
- Fields: `last_heard`, `play_count`, `total_play_duration`, `current_play_start`
- Methods: `record_play_start()`, `record_play_end()`, `update_last_heard()`, `is_playing()`, `current_play_duration()`
- Features: Increments play_count in record_play_start(), has current_play_duration tracking

**SignalHistoryComponent** (`src/ecs/components/signal/mod.rs`, lines 197-383):
- Fields: `last_heard`, `play_count`, `total_play_duration`, `current_play_start`
- Methods: `start_play_session()`, `end_play_session()`, `play_count()`, `total_play_duration()`, `last_heard()`
- Features: Cleaner session-based API

**Decision**: Keep SignalHistoryComponent but merge improvements from Station
- Need to add: `is_playing()`, `current_play_duration()`, `update_last_heard()`
- Consider: Station's approach of incrementing count on start vs Signal's approach on end

## PlaybackComponent Comparison

**StationPlaybackComponent** (`src/ecs/components/station/playback.rs`):
- States: `Idle`, `Playing` (simpler)
- Methods: `start_playing(audio_id)`, `stop_playing()`, `is_idle()`

**SignalPlaybackComponent** (`src/ecs/components/signal/mod.rs`, lines 208-341):
- States: `NotPlaying`, `Playing`, `Completed` (more states)
- Methods: `transition_to()`, `set_audio_id()`, `state_changed_at()`, `is_playing()`
- Features: Tracks state change timestamps

**Decision**: Keep SignalPlaybackComponent (more sophisticated)
- Has timestamp tracking
- More comprehensive state machine
- Better separation of concerns

## DiscoveryComponent Comparison

**StationDiscoveryComponent** (`src/ecs/components/station/discovery.rs`):
- Fields: `discovered_at`, `window_id`
- Methods: `discovered_ago()`
- Features: Has `discovered_ago()` helper method

**SignalDiscoveryComponent** (`src/ecs/components/signal/mod.rs`, lines 44-194):
- Fields: `discovered_at`, `window_id`
- Methods: `discovered_at()`, `window_id()`

**Decision**: Keep SignalDiscoveryComponent but add discovered_ago() method
- Signal version has better accessor methods
- Need to add the convenient `discovered_ago()` helper from Station version

## Summary of Merges Needed

1. **SignalHistoryComponent**: Add `is_playing()`, `current_play_duration()`, `update_last_heard()` methods
2. **SignalDiscoveryComponent**: Add `discovered_ago()` method
3. **SignalInfoComponent**: No changes needed (already superior)
4. **SignalPlaybackComponent**: No changes needed (already superior)