# SDRplay Hot-Plug Removal

Fixed SDRplay devices not disappearing from TUI when unplugged during active scans.

## Challenges

### Challenge: Device removal events not reaching TUI

**Goal**: Have SDRplay devices disappear from the TUI list immediately when unplugged, matching RTL-SDR behavior.

**Failure Mode**: RTL-SDR devices disappeared correctly when unplugged, but SDRplay devices remained in the TUI list indefinitely.

**Attempts**:
- Added logging to trace event flow through discovery system - confirmed udev events were detected and enumeration tasks were being submitted
- Investigated discovery forwarder shutdown timing - initially thought shutdown check was dropping events prematurely, but user clarified shutdown only happened after waiting for UI update
- Attempted to make ScanStationsTask yield semaphore between windows - added window_index field and TaskContinuation::Resubmit logic, but this was the wrong task (scan used ScanBandTask, not ScanStationsTask)

**Solution**: Changed TaskScheduler to use async `acquire_owned()` with FIFO ordering instead of `try_acquire_owned()` busy-waiting. Created per-thread Tokio runtime using `tokio::runtime::Builder::new_current_thread()` to enable `runtime.block_on(semaphore.acquire_owned())` from sync context.

**Key Insight**: The scheduler was using `try_acquire_owned()` which doesn't provide fair queuing. When ScanBandTask yielded the backend semaphore after processing a window, it would immediately reacquire it before the waiting enumeration task could grab it. The async `acquire_owned()` method provides FIFO fairness, ensuring enumeration tasks get their turn when the scan yields.

### Challenge: Understanding semaphore yielding behavior

**Goal**: Verify that ScanBandTask was actually yielding the backend semaphore between windows to allow enumeration.

**Failure Mode**: Initial investigation showed no yielding logs, suggesting the task was holding the semaphore for the entire scan duration.

**Attempts**:
- Added extensive logging to track task execution - revealed `run()` was only called twice during entire scan
- Investigated state machine transitions - found no mode transition logs despite state machine being initialized
- Checked for errors that might prevent yielding - no scan errors found

**Solution**: Discovered that each scanning window processes synchronously for several seconds while holding the backend semaphore. ScanBandTask does yield via TaskContinuation::Resubmit after each window completes, but enumeration tasks couldn't acquire the semaphore due to lack of FIFO ordering.

**Key Insight**: Each window took 443 log lines (~several seconds) to complete. The scan task was yielding correctly after each window, but the non-FIFO semaphore acquisition meant resubmitting tasks would grab the permit before waiting tasks could. Tokio's Semaphore documentation confirms: "This Semaphore is fair, which means that permits are given out in the order they were requested" - but only for the async `acquire_owned()`, not `try_acquire_owned()`.
