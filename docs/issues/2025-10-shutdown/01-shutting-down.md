# SDRplay Device Shutdown Issue - First Pass Analysis

**Date**: October 2025
**Status**: Research complete

## Problem Statement

The scanner leaves the SDRplay device in a bad state after exiting, requiring manual intervention to recover:
- Sometimes requires `sudo systemctl restart sdrplay`
- Sometimes requires starting the scanner twice (first attempt fails, second works)
- Occurs in both TUI and headless modes
- Affects both normal exits and crashes

## Symptoms

When the device is in a bad state:
1. Next scanner startup may fail to initialize SDR
2. Device may be unresponsive or locked
3. SoapySDR may report device access errors
4. Mutex locks may be held from previous run
5. `sdrplay_api_Init()` may report "Failed to lock mutex" errors
6. Device may report `SoapySDR::Device::readStream timeout!` errors

## Root Cause Analysis

### SoapySDR Stream Lifecycle

According to [SoapySDR Device documentation](https://pothosware.github.io/SoapySDR/doxygen/latest/classSoapySDR_1_1Device.html) and the [DriverGuide](https://github.com/pothosware/SoapySDR/wiki/DriverGuide), proper stream management requires:

1. **setupStream()** - Allocates resources for streaming (may be lengthy)
2. **activateStream()** - Starts data flow (lightweight on/off switch)
3. **deactivateStream()** - Stops data flow (lightweight off switch)
4. **closeStream()** - Deallocates resources, may power down components (may be lengthy)

**Critical design principle**: setup/close handle lengthy allocation and cleanup procedures, while activate/deactivate are lightweight switches that may be called multiple times. The proper implementation separates resource allocation from stream control.

**Proper shutdown sequence:**
```
deactivateStream(stream)  // Stop data flow first
closeStream(stream)       // Then clean up resources
```

**Why this order matters**: Community reports indicate that calling `closeStream()` on a still-active stream can cause crashes. One report states: "the closeStream call segfaulted if it didn't first stop the streaming threads in deactivateStream." This is especially problematic when clients are killed (e.g., kill -9) without proper cleanup.

### SDRplay-Specific Issues

The SDRplay API and service layer add complexity beyond standard SoapySDR stream management:

**Mutex Lock Failures**:
- Error: `sdrplay_api_Init(): line 583: Failed to lock mutex`
- Occurs when the API service holds locks from a previous run
- Common after improper shutdown or crashes

**Recovery Methods** (in order of invasiveness):
1. **Service restart**: `sudo systemctl restart sdrplay` (Linux) or restart via SDRuno/Start menu (Windows)
2. **USB cable unplug/replug**: Physical hardware reset - unplug USB, wait several seconds, reconnect
3. **System reboot**: Required in severe cases, especially on macOS

**Global Mutex and Callbacks**:
- SDRplay API uses a global mutex that can block callbacks
- If software waits for callbacks without releasing the global lock, timeouts occur
- Community fix: "unlocking global mutex during sleep, to allow callbacks to execute"
- This addresses "random timeouts and failures to set various RSP settings (such as LNA state)"

**Service Management**:
- Windows: SDRplay API Service must be running (check in Services)
- Some system optimization tools (e.g., AVG Tuneup) may disable the service
- macOS: Service auto-restarts when stopped via `sudo launchctl stop com.sdrplay.sdrplay_service`

### Current Implementation Issues

1. **rust-soapysdr's RxStream** (`/home/walker/.cargo/registry/src/.../soapysdr-0.4.2/src/device.rs:931-940`)
   - **Has proper Drop implementation**:
     ```rust
     impl<E: StreamSample> Drop for RxStream<E> {
         fn drop(&mut self) {
             unsafe {
                 if self.active {
                     self.deactivate(None).ok();  // ✓ Deactivates before closing
                 }
                 SoapySDRDevice_closeStream(self.device.inner.ptr, self.handle);
             }
         }
     }
     ```
   - This properly follows the deactivate → close sequence
   - Assumes it gets a chance to run during normal Drop

2. **rustradio's SoapySdrSource** (`/home/walker/code/rustradio/src/soapysdr_source.rs:287-299`)
   - **No Drop implementation**
   - Stores `stream: soapysdr::RxStream<Complex>` as a struct field
   - When dropped, fields are dropped in declaration order
   - Relies on RxStream's Drop to cleanup, which should work in normal circumstances

3. **Scanner's signal handler** (`/home/walker/code/emwalker/scanner/bin/scanner.rs:400-409`)
   - Catches SIGINT/SIGTERM with ctrlc handler
   - Triggers graceful shutdown signal
   - Sleeps 500ms to allow cleanup
   - Calls `soapy::cleanup_soapysdr_state()`
   - **Forces `std::process::exit(0)` immediately after**
   - **Problem**: `std::process::exit()` terminates process without running Drop implementations
   - This prevents RxStream::drop() from deactivating/closing the stream

4. **Scanner's TUI normal exit path** (`/home/walker/code/emwalker/scanner/bin/scanner.rs:475-492`)
   - **Critical race condition**:
     ```rust
     let _ = tui_handle.join();           // Line 476: Wait for TUI
     shutdown_trigger.trigger();          // Line 479: Signal main thread
     std::thread::sleep(...::from_millis(100));  // Line 483: Inadequate sleep
     let _result = std::thread::spawn(move || {  // Lines 487-489: Detached thread!
         let _ = main_handle.join();
     });
     Ok(())  // Line 492: Returns immediately
     ```
   - **Problem**: Spawns detached thread to join main thread, then returns immediately
   - When `Ok(())` returns → propagates to main() → main() returns → process exits
   - Main thread might still be running Drop implementations when process exits
   - If Drop completes before process exit: ✓ Device clean
   - If process exits during Drop: ✗ Device corrupted
   - **This explains intermittent failures even without Ctrl+C**

5. **Scanner's headless normal exit path** (`/home/walker/code/emwalker/scanner/bin/scanner.rs:511-522`)
   - **Works correctly**:
     ```rust
     let main_handle = thread::spawn(...);
     let result = main_handle.join()  // Line 515-517: Properly waits!
         .map_err(...)?;
     shutdown_trigger.trigger();
     result
     ```
   - Actually waits for main thread to complete before returning
   - Drop implementations run to completion
   - Device cleaned up properly
   - No race condition in headless mode normal exit

### The Issue Chain

**Headless Normal Exit Path (works correctly)**:
```
1. Scanner completes normally in headless mode
2. main_handle.join() waits for main thread (line 515-517)
3. Main thread completes, starts dropping
4. SoapySdrManager::drop() called
5. Graph cancelled, thread joined
6. SoapySdrSource dropped (no custom Drop)
7. RxStream field dropped (Drop impl runs)
8. ✓ RxStream::drop() calls deactivate(), then closeStream()
9. Device left in clean state
10. Process exits
```

**TUI Normal Exit Path (RACE CONDITION)**:
```
1. User presses 'q' or scanner completes in TUI mode
2. TUI thread exits
3. main(): tui_handle.join() completes (line 476)
4. main(): shutdown_trigger.trigger() (line 479)
5. main(): sleep 100ms (line 483) - inadequate!
6. main(): spawns DETACHED thread to join main_handle (line 487-489)
7. main(): returns Ok(()) immediately (line 492)
8. main() returns to OS
9. Process begins exiting
10. Main thread might still be running:
    a. Detecting shutdown
    b. Exiting run() loop
    c. Starting Drop of SoapySdrManager
    d. Calling stop() to cancel graph
    e. Joining graph thread
    f. Dropping SoapySdrSource
    g. Dropping RxStream
    h. Calling deactivate/close
11. ❌ If process exits during steps 10a-10h: Device corrupted
12. ✓ If Drop completes before process exits: Device clean
13. This race explains intermittent failures!
```

**Signal Handler Path (problematic)**:
```
1. User presses Ctrl+C (SIGINT/SIGTERM)
2. ctrlc handler catches signal
3. Triggers shutdown via triggered::trigger()
4. Sleeps 500ms for "graceful" shutdown
5. Calls soapy::cleanup_soapysdr_state()
   ❌ This unloads SoapySDR modules via SoapySDR_unloadModules()
   ❌ Now deactivateStream/closeStream functions are gone!
6. ❌ Calls std::process::exit(0) - terminates immediately
7. ❌ Drop implementations never run (or crash if they do run)
8. ❌ Stream never deactivated or closed (functions unloaded)
9. ❌ Device/API left holding locks and resources
10. Next run: mutex lock failures or timeouts
```

**Multiple Core Problems**:

1. **std::process::exit(0) bypasses Drop**: Terminates the process immediately without running destructors:
   - The sleep happens in the signal handler thread, doesn't wait for main thread
   - Drop implementations need to run on the owning thread
   - `std::process::exit()` bypasses all Drop implementations regardless

2. **cleanup_soapysdr_state() timing is backwards**: Called BEFORE Drop tries to run:
   - Unloads SoapySDR plugin modules
   - Drop implementations try to call deactivateStream/closeStream
   - Those functions are in the unloaded modules
   - Results in crashes or no-ops
   - Device cleanup never happens

3. **No join timeout**: If Drop did run and graph thread was stuck, it would hang forever in `SoapySdrManager::stop()`

## Research Findings

### SoapySDR Documentation

From the [SoapySDR DriverGuide](https://github.com/pothosware/SoapySDR/wiki/DriverGuide):
- **setup/close**: Handle lengthy allocation and cleanup procedures
- **activate/deactivate**: Lightweight on/off switches, may be called multiple times
- Proper implementation separates resource allocation from stream control
- Calling operations in the wrong order can crash the driver

From [SoapySDR Device API](https://pothosware.github.io/SoapySDR/doxygen/latest/classSoapySDR_1_1Device.html):
- `deactivateStream()`: "Deactivate a stream. Call deactivate when not using read/write()."
- `closeStream()`: "Close an open stream created by setupStream. The implementation may change switches or power-down components."

### Community-Reported Issues

**readStream Timeout Errors**:
- Multiple users report `SoapySDR::Device::readStream timeout!` errors
- Particularly common with SDRplay RSP1 and RSPduo devices
- Can occur when stream is in inconsistent state from previous run
- Timeout parameter behavior varies by driver implementation

**Device Busy/Lock Errors**:
- "Timeout expired/failed to establish connection with the device"
- Occurs when device or API service holds locks from crashed/killed processes
- Most common after kill -9 or crashes during streaming
- Can persist across application restarts

**Cleanup on Kill**:
- Issue report: "when clients were killed, the server was trying to clean up the stream"
- Problem: "the closeStream call segfaulted if it didn't first stop the streaming threads in deactivateStream"
- SoapySDRServer can die with "terminate called without an active exception" when clients are kill -9'd
- Drivers need to handle cleanup gracefully even if activate/deactivate wasn't properly called

**API Version Mismatches**:
- API version mismatches between SoapySDR core and device plugins can cause crashes in destructors
- Particularly problematic when SoapySDR and device-specific modules are compiled against different library versions
- Can manifest as segfaults during stream cleanup

**std::process::exit() and Drop**:
- Rust documentation clearly states: "This function will never return and will immediately terminate the current process in a platform specific manner"
- Unlike normal return from main(), `std::process::exit()` does not:
  - Run destructors for any stack-allocated objects
  - Call Drop implementations
  - Flush I/O buffers (stdout, stderr, files)
  - Run `atexit` handlers
- This is a fundamental limitation, not a bug

### Code Analysis Summary

**rust-soapysdr library** (`/home/walker/.cargo/registry/src/.../soapysdr-0.4.2/src/device.rs`):
- Lines 931-940: Proper `Drop` implementation for `RxStream<E>`
- Deactivates stream if active, then calls closeStream
- ✓ Correctly implements the deactivate → close sequence
- Only works if Drop gets a chance to run

**rustradio** (`/home/walker/code/rustradio/src/soapysdr_source.rs`):
- Lines 266-267: Creates and activates stream in builder
- Lines 287-299: `SoapySdrSource` struct definition
- Has `stream: soapysdr::RxStream<Complex>` as field
- No custom Drop implementation
- ✓ Relies on automatic field drop (which calls RxStream::drop())
- This is correct design - the RxStream Drop handles cleanup

**scanner binary** (`/home/walker/code/emwalker/scanner/bin/scanner.rs`):
- Line 386: `soapy::reset_soapysdr_state();` at startup only
- Lines 400-409: Signal handler implementation
  - Triggers shutdown signal
  - Sleeps 500ms
  - Calls `soapy::cleanup_soapysdr_state()`
  - ❌ Calls `std::process::exit(0)` - bypasses all Drop
- **Missing**: No cleanup at end of normal `handle_scan_command()` execution
- **Missing**: No explicit stream deactivation before exit

**scanner soapy module** (`/home/walker/code/emwalker/scanner/src/soapy.rs`):
- Lines 219-236: `impl Drop for SoapySdrManager`
- Properly cancels graph and joins thread
- Graph cancellation should cause SoapySdrSource to drop
- Which should cause RxStream to drop
- Which should deactivate and close stream
- ✓ Correct design, but only works if Drop runs
- ❌ `stop()` method (lines 122-133) has no join timeout
  ```rust
  if let Some(handle) = self.graph_handle.take() {
      let _ = handle.join();  // Can hang forever if graph thread stuck
  }
  ```
- If graph thread is stuck in SoapySDR readStream(), join() never returns
- Entire shutdown process hangs indefinitely

## Key Insights

1. **rust-soapysdr is correctly implemented**: The library properly deactivates streams before closing them in the Drop implementation. This is not the source of the problem.

2. **Headless mode normal exit works correctly**: When the scanner exits normally in headless mode (not via signal), the Drop chain executes correctly: SoapySdrManager → rustradio graph cleanup → SoapySdrSource → RxStream → deactivate + close. The code properly joins the main thread before returning.

3. **TUI mode has a critical race condition** (`bin/scanner.rs:487-492`):
   ```rust
   // After TUI exits and shutdown is triggered:
   let _result = std::thread::spawn(move || {
       let _ = main_handle.join();  // Join happens in DETACHED thread
   });
   Ok(())  // Returns immediately - RACE!
   ```
   When `Ok(())` returns, it propagates to `main()`, which returns, **terminating the process**. Meanwhile, the main scanning thread is still running Drop implementations. If process exit happens before Drop completes, the device is left corrupted. This explains why the problem happens **intermittently even without Ctrl+C**.

4. **Signal handler bypasses Drop**: The `std::process::exit(0)` call in the signal handler completely bypasses Rust's Drop system, leaving the device with:
   - Active streams that were never deactivated
   - Resources that were never closed
   - Locks that were never released
   - API state that was never cleaned up

5. **500ms sleep is ineffective**: The sleep in the signal handler doesn't help because:
   - Drop needs to run on the owning thread (main thread)
   - Sleep happens in signal handler thread
   - `std::process::exit()` still bypasses Drop regardless

6. **cleanup_soapysdr_state() timing is wrong**: It's called in the signal handler (line 406) **before** Drop tries to run. This unloads the SoapySDR modules, so when `RxStream::drop()` tries to call `deactivateStream()` and `closeStream()`, the plugin code is already unloaded, causing crashes or preventing cleanup.

7. **No join timeout in SoapySdrManager::stop()**: The `handle.join()` call can hang forever if the graph thread is stuck, blocking the entire shutdown process.

8. **Multiple paths to device corruption**:
   - **Path 1**: User presses Ctrl+C → signal handler → `std::process::exit(0)` → no cleanup
   - **Path 2**: TUI normal exit → spawns detached join thread → returns immediately → main() returns → process exits → Drop gets cut off
   - **Path 3**: cleanup_soapysdr_state() unloads modules → Drop crashes trying to use unloaded code

9. **SDRplay API compounds the issue**: The SDRplay API service maintains global mutex state that persists across process lifetimes. When a process dies without proper cleanup, these locks remain held, requiring service restart or hardware reset.

## References

**SoapySDR Documentation**:
- [SoapySDR Device API](https://pothosware.github.io/SoapySDR/doxygen/latest/classSoapySDR_1_1Device.html)
- [SoapySDR Driver Guide](https://github.com/pothosware/SoapySDR/wiki/DriverGuide)

**Code Locations**:
- rust-soapysdr RxStream Drop: `/home/walker/.cargo/registry/src/.../soapysdr-0.4.2/src/device.rs:931-940`
- rustradio SoapySdrSource: `/home/walker/code/rustradio/src/soapysdr_source.rs:266-299`
- scanner soapy module: `/home/walker/code/emwalker/scanner/src/soapy.rs:219-236`
- scanner binary signal handler: `/home/walker/code/emwalker/scanner/bin/scanner.rs:400-409`

**Community Resources**:
- GitHub Issues: SoapySDR timeout and cleanup issues
- SDRplay forum: Mutex lock failures and recovery methods
- SoapySDRPlay3 PR #59: Global mutex and callback blocking fixes

## Summary

The SDRplay device shutdown issue had **five distinct root causes**:

1. **TUI Race Condition** (bin/scanner.rs:487-492): ✅ **FIXED** - Was spawning detached thread to join main, returning immediately. Fixed by actually waiting for main thread join before returning.

2. **Signal Handler Exit** (bin/scanner.rs:408): ✅ **FIXED** - Was calling `std::process::exit(0)` bypassing Drop. Fixed with double Ctrl+C pattern: first triggers graceful shutdown, second force quits.

3. **Cleanup Timing** (bin/scanner.rs:406): ✅ **FIXED** - Was calling `cleanup_soapysdr_state()` before Drop. Fixed by only calling it on force quit (second Ctrl+C).

4. **Missing Shutdown Check in Pause Loop** (src/main_thread.rs:451-454): ✅ **FIXED** - **This was the critical issue**. When scanner was paused (browse mode), it would sleep and loop without checking shutdown signal. Main thread would never exit. Fixed by adding shutdown check inside the paused loop.

5. **AudioSession Drop Order** (src/audio_session.rs:120-119): ✅ **FIXED** - Was dropping SDR segment before joining audio graph thread, causing use-after-free. Fixed by cancelling and joining audio graph thread BEFORE dropping SDR segment.

**The Key Discovery**:
The main hang wasn't in window processing or thread joins - it was in the **pause loop**. When the scanner was in browse mode (user had pressed pause to listen to a station), the scan loop would:
```rust
if paused {
    std::thread::sleep(100ms);
    continue;  // ← Goes back to top of loop
}
// Shutdown check was HERE ← Never reached when paused!
if self.shutdown_listener.is_triggered() {
    break;
}
```

When shutdown was triggered while paused, the loop would continue forever because the shutdown check was AFTER the paused block.

**Why All Attempts Failed Initially**:
We were looking at window processing, thread joins, and AudioSession drop order, but the real issue was simpler: the main thread wasn't detecting shutdown at all when paused. Once we added the shutdown check inside the pause loop, everything else worked.

**Final Solution**:
- Fixed signal handler to allow graceful shutdown (double Ctrl+C pattern)
- Fixed TUI exit to wait for main thread completion
- Added shutdown check in pause loop (CRITICAL FIX)
- Fixed AudioSession drop order (cancel/join audio graph before dropping SDR segment)
- Added shutdown checks in window processing for faster response

**Status**: All shutdown paths now work correctly. Device is left in clean state.

## Note: AudioSession Drop Order (Resolved)

During debugging, we initially thought the AudioSession drop order was causing the hang:

### AudioSession Drop Order Investigation

**Location**: `src/audio_session.rs:97-119`

**Initial Hypothesis**: When the scanner enters browse/pause mode (user presses pause in TUI), an `AudioSession` is created that plays audio continuously. This AudioSession has two critical resources:

1. **SDR segment** (`Box<dyn Segment>`): Contains the SoapySDR device and SDR graph thread
2. **Audio graph handle** (`GraphHandle`): Contains the audio processing graph thread

**Current Drop Sequence (WRONG)**:
```
1. AudioSession::drop() called
2. stop_current_station() called
3. Takes current_segment, drops it immediately
   a. SoapySdrManager::drop() called
   b. SDR graph cancelled and joined
   c. RxStream::drop() → deactivateStream() + closeStream()
   d. SoapySDR device closed and potentially freed
4. Takes current_graph (GraphHandle), drops it
   a. GraphHandle::drop() calls cancel()
   b. GraphHandle::drop() calls join() on audio graph thread
   c. ❌ Audio graph thread might still be processing SDR data!
   d. ❌ Audio graph might access freed SoapySDR device
   e. ❌ SIGSEGV - use-after-free
```

**Why This Happens**:
- The audio graph thread is a separate thread running `audio_graph.run()`
- It reads from the SDR segment's broadcast channel
- It processes samples through FM demodulation and audio resampling
- When we drop the SDR segment first, we:
  - Close the SoapySDR device (deactivateStream + closeStream)
  - Potentially free device memory
  - Close the broadcast channel
- **But**: The audio graph thread is still running!
- It might be:
  - In the middle of reading a sample
  - Processing data in the FM demodulator
  - Holding pointers to SDR-sourced data
- When the audio graph thread tries to access anything related to the SDR segment → SIGSEGV

**Attempted Fixes That Failed**:
1. **Dropping SDR segment first** (current code): Causes SIGSEGV as described above
2. **Adding GraphHandle newtype with Drop**: Doesn't change the drop order, still SIGSEGV
3. **Passing shutdown_listener to AudioSession**: Field never used, doesn't affect drop order

**The Chicken-and-Egg Problem**:
- Can't drop SDR segment first: Audio graph thread might still use it → SIGSEGV
- Can't drop GraphHandle first: Audio graph thread won't exit without broadcast channel closing → hang
- Can't join audio graph while SDR segment alive: Thread keeps reading forever → hang
- Can't close SDR device while audio graph alive: Thread might access freed memory → SIGSEGV

**Why This Manifests During Shutdown**:
1. User presses 'q' or Ctrl+C while scanner is in browse mode
2. Scanner has an active AudioSession playing a station
3. Shutdown is triggered
4. Main thread starts exiting, dropping MainThread
5. MainThread::drop() drops `audio_session: Option<AudioSession>`
6. AudioSession::drop() calls stop_current_station()
7. **stop_current_station() has the wrong drop order**
8. SIGSEGV or hang, depending on exact timing

**Evidence from Logs**:
```
playing 88.9 MHz [moderate audio]          ← AudioSession active
AudioSession: Tuning to station            ← Audio graph being created
Window complete, advancing to next         ← Window finished
AudioSession: Audio graph thread started   ← Audio graph running
[User presses 'q']
>>> TUI finished
>>> Triggered shutdown for main thread
>>> Waiting for main thread to finish...
[HANG or SIGSEGV]
```

**What We Thought Was Happening**:
We believed the AudioSession drop order was causing SIGSEGV because we were dropping the SDR segment before joining the audio graph thread, leading to use-after-free.

**What Was Actually Happening**:
The AudioSession never got a chance to drop! The main thread was stuck in the pause loop and never reached the point where AudioSession would be dropped. The real issue was the missing shutdown check in the pause loop (root cause #4).

**The Fix We Applied Anyway**:
Even though it wasn't the immediate blocker, we fixed the AudioSession drop order as a defensive measure:
```rust
// Cancel and join audio graph thread FIRST
if let Some((cancel_token, handle)) = self.current_graph.take() {
    cancel_token.cancel();
    let _ = handle.join();
}
// THEN drop SDR segment (after audio graph is definitely gone)
if let Some(segment) = self.current_segment.take() {
    drop(segment);
}
```

This ensures that if/when AudioSession does drop (after the pause loop issue is fixed), it won't cause use-after-free errors.

**Impact**:
- Initially thought this was the main cause of hangs/crashes
- Actually was a secondary issue that would have manifested once the pause loop was fixed
- Fixed proactively to prevent future issues
- Good defensive programming even though it wasn't the immediate blocker
