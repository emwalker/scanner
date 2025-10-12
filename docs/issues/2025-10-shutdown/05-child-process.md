# Graceful Shutdown of Child Process Over IPC

Implemented proper graceful shutdown for SDR worker subprocesses communicating over Unix domain sockets, eliminating the need for SIGTERM/SIGKILL in normal operation.

## Challenges

### Challenge: Adding Shutdown Acknowledgment Message

**Goal**: Confirm that the child worker subprocess received the Shutdown message before it exited.

**Failure Mode**: After adding a `ShutdownAck` message, the parent process would hang waiting for the acknowledgment that never arrived. The worker subprocess would send the ShutdownAck but then immediately break from its control loop, dropping the `ctl_channel` (Unix socket) before the parent could read the message from the socket buffer.

**Attempts**:
- Added explicit `drop(ctl_channel)` after sending ShutdownAck - this didn't help because dropping earlier doesn't keep the socket open longer
- Added inter-thread coordination with `done_tx/done_rx` channels to keep control thread alive until data thread finished - this created a deadlock where the parent blocked on `control.recv()` waiting for ShutdownAck while the worker blocked waiting for internal thread coordination
- Considered using `thread::sleep()` to give parent time to read from socket buffer - rejected as a race condition fix

**Solution**: Removed the ShutdownAck message entirely. The correct pattern is to send the Shutdown message and then wait for the process to exit using `process.wait()`. Process exit serves as the implicit acknowledgment that shutdown completed successfully.

**Key Insight**: Rust's graceful shutdown pattern (documented in The Rust Book) does not use explicit acknowledgment messages in shutdown paths. The parent sends a signal (shutdown message, or dropping a sender), then waits for thread joins or process exit. Attempting to send acknowledgments during cleanup creates circular dependencies and deadlocks because shutdown involves tearing down the communication channels themselves.

### Challenge: Thread Joining During Shutdown

**Goal**: Ensure both control and data threads complete their cleanup before the worker process exits.

**Failure Mode**: Initial implementation called `.join()` on both threads in the main loop after spawning them. Combined with waiting for ShutdownAck, this created deadlock scenarios.

**Attempts**:
- Kept thread joins while trying to coordinate shutdown with additional channels - this maintained the deadlock because the main thread would wait for control thread, which waited for data thread, which waited for internal commands

**Solution**: Removed thread joins entirely - let both threads finish naturally by breaking from their loops. When both threads exit, the process terminates automatically due to RAII cleanup. The parent's `process.wait()` returns when this happens.

**Key Insight**: In subprocess shutdown, joining threads in the main function is unnecessary. The process exit itself provides the synchronization point. Joining threads during shutdown can create deadlocks when combined with other coordination mechanisms.

### Challenge: Timeout Configuration

**Goal**: Choose an appropriate timeout for graceful shutdown before escalating to SIGTERM.

**Failure Mode**: Initial 5-second timeout was conservative but unnecessarily long given that testing showed the worker subprocess exited in under 1 second.

**Solution**: Reduced graceful shutdown timeout to 2 seconds based on industry standards research:
- Docker default: 10 seconds
- Kubernetes default: 30 seconds
- systemd default: 90 seconds
- Our worker subprocess: exits in < 1 second in testing

For a simple SDR worker that only needs to stop streaming and clean up, 2 seconds provides ample margin while keeping shutdown responsive.

**Key Insight**: Timeout values should be based on empirical testing of your specific application's shutdown time, not just copying framework defaults. Industry defaults (10-90 seconds) are designed for complex applications with database connections, network drains, and in-flight request handling.

---

## Implementation Summary

Final shutdown flow:
1. Parent sends `ControlMessage::Shutdown` via Unix socket
2. Parent waits for `process.wait()` to return (2 second timeout)
3. Worker control thread receives Shutdown, sends `InternalCommand::Shutdown` to data thread, breaks from loop
4. Worker data thread receives internal Shutdown, cleans up active streams, exits loop
5. Both threads exit naturally, process terminates
6. Parent's `process.wait()` returns successfully

Escalation if timeout exceeded:
1. Send SIGTERM, wait 1 second
2. Send SIGKILL if still running

Files modified:
- `src/ipc/protocol.rs`: No ShutdownAck variant needed
- `src/cli/worker.rs:264-268`: Control thread handles Shutdown by signaling data thread and breaking
- `src/cli/worker.rs:413-420`: Removed thread joins
- `src/hardware/pool/subprocess.rs:242-253`: Removed ShutdownAck waiting
- `src/hardware/pool/subprocess.rs:255`: Reduced timeout from 5s to 2s
