# Retuning Deadlock Fix

Fixed deadlock and IPC communication issues preventing subprocess-based scanning from completing window transitions in TUI mode.

## Challenges

### Challenge: Subprocess Premature Exit

**Goal**: Launch TUI scan using `--use-subprocesses` flag without errors.

**Failure Mode**: Scan didn't start and exited with "Io(Error { kind: UnexpectedEof, message: 'failed to fill whole buffer' })" error.

**Solution**: Changed worker's `main_loop()` to join threads instead of dropping them. The worker was spawning control and data threads then immediately exiting, causing the IPC connection to close before the parent could communicate.

**Key Insight**: Thread handles must be explicitly joined for the process to wait for thread completion. Dropping thread handles allows the main thread to continue immediately.

---

### Challenge: Deadlock During Window Transitions

**Goal**: Complete scanning transitions from Window 1 to Window 2+ without hanging.

**Failure Mode**: Window 2 never completed. Parent process stopped reading from data socket (normal during window transition), causing worker's data thread to block indefinitely in `dat_sender.send()`. The blocked data thread couldn't process the `StopStream` command from the control thread, creating a circular deadlock.

**Attempts**:
- Reduced read timeout from 100ms to 10ms: Rejected because SoapySDRPlay has bugs where timeout parameters aren't respected by the underlying driver.
- Made data socket completely non-blocking: Introduced major audio skipping because sends would immediately fail with WouldBlock and drop samples.

**Solution**: Combined approach in `src/cli/worker.rs`:
1. Added 100ms write timeout to data socket (line 166) so sends don't block indefinitely
2. Check for pending commands before each send (lines 426-438) to detect StopStream even during backpressure
3. Handle timeout/WouldBlock errors gracefully by dropping packets and breaking to check commands (lines 459-470)
4. Track whether any work was done and sleep 100ms when inactive to prevent CPU busy loop (lines 492-493)

**Key Insight**: When implementing producer-consumer patterns with blocking I/O, the producer must handle backpressure without blocking command processing. Write timeouts combined with command checking before blocking operations prevent deadlock.

---

### Challenge: Device Retuning vs Recreation

**Goal**: Efficiently switch between scanning windows by retuning the existing device.

**Failure Mode**: Retuning appeared to leave SoapySDRPlay in a bad state, contributing to timeout and deadlock issues.

**Solution**: Adopted rustradio's pattern of tearing down and recreating the device for each window transition. Changed `ConfigureAndStart` to recreate the device (lines 330-346) and `StopStream` to explicitly drop the device (lines 375-395). This matches the working direct path behavior.

**Key Insight**: Some SDR drivers don't handle retuning reliably. Recreating the device is more robust than attempting to retune, even though it involves additional overhead.
