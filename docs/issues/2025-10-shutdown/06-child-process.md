# Subprocess Shutdown Hang When Pressing 'q' in TUI

Fixed subprocess-based scanning hanging indefinitely after audio playback completed when user pressed 'q' to exit.

## Challenges

### Challenge: Identifying the Hang Location

**Goal**: Determine where in the shutdown sequence the hang was occurring.

**Failure Mode**: Application hung after printing "Audio stream statistics", never returning control to shell. Only occurred with `--use-subprocesses` flag.

**Attempts**:
- Initially suspected device enumeration conflicts based on earlier issues
- Investigated retuning vs device recreation approaches - both had same hang
- Ruled out worker subprocess (it exited cleanly per logs)

**Solution**: Added debug logging to trace shutdown sequence. Found hang in `main_handle.join()` waiting for main thread, which was blocked in `Segment::drop()` waiting for graph thread to finish.

**Key Insight**: The worker subprocess was exiting correctly. The hang was in the main process's graph thread that reads from the subprocess via Unix socket.

### Challenge: Graph Thread Blocked in Socket Read

**Goal**: Allow graph thread to exit cleanly when shutdown is triggered.

**Failure Mode**: Graph thread blocked forever in `SubprocessSource::work()` calling `receiver.recv()`, which performed blocking `read_exact()` on Unix socket. When subprocess stopped sending data during shutdown, no more bytes would arrive, causing infinite wait.

**Attempts**:
- Verified read timeout was already set to 100ms on data socket (src/hardware/pool/subprocess.rs:110)
- Found timeout errors were being treated as fatal errors in SubprocessSource

**Solution**: Updated error handling in src/hardware/pool/subprocess_source.rs:73-80 to treat timeout errors as non-fatal:

```rust
Err(e) => {
    let err_str = e.to_string();
    if err_str.contains("would block") || err_str.contains("timed out") {
        Ok(BlockRet::Again)  // Returns control, allowing cancellation check
    } else {
        Err(rustradio::Error::msg(format!("Data receiver error: {}", e)))
    }
}
```

**Key Insight**: Unix socket read timeouts generate different error messages than `WouldBlock` errors. The existing code only checked for "would block" but not "timed out", so timeout errors were being treated as fatal instead of allowing the graph to check its cancellation token.
