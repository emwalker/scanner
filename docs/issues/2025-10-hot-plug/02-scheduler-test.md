# Scheduler FIFO Regression Test

Created a regression test to verify FIFO semaphore ordering when tasks yield via TaskContinuation::Resubmit.

## Challenges

### Challenge: Ensuring tasks queue on semaphore before yielding task reacquires

**Goal**: Create a test that would fail with `try_acquire_owned()` and pass with async `acquire_owned()`, demonstrating the FIFO ordering requirement.

**Failure Mode**: Tasks 2 & 3 weren't reaching the `acquire_owned()` call before task 1 yielded and immediately reacquired. Test showed `[1, 1, 1, 2, 3]` even with the fix in place, indicating tasks 2 & 3 weren't queued yet when task 1 yielded.

**Attempts**:
- Used barriers and multiple yielding tasks to orchestrate timing - tasks completed before others started due to thread scheduling non-determinism
- Tried event-driven approach without any sleeps - couldn't guarantee tasks 2 & 3 would reach `acquire_owned()` before task 1 finished its first run
- Added 100ms sleep in task 1's first run - still too short, tasks 2 & 3 weren't consistently queued before task 1 reacquired

**Solution**: Made task 1 sleep for 200ms during its first run to simulate real-world scan windows. After receiving task 1's first "run" event, wait 10ms to submit tasks 2 & 3, then wait an additional 150ms before collecting results. This ensures tasks 2 & 3 have 150ms to start their threads, create Tokio runtimes, and reach `acquire_owned()` while task 1 still holds the permit for another 40ms.

**Key Insight**: Testing semaphore FIFO ordering requires ensuring queued tasks are actually blocked on `acquire_owned()` before the holding task releases. This requires accounting for thread startup time, runtime creation, and scheduler non-determinism. A sleep duration that gives queued tasks ample time to reach the semaphore while the holding task is still active is more reliable than purely event-driven coordination.
