# IPC Protocol Design Guidelines

## Architecture

The IPC system uses traits to enable multiple implementations (Unix sockets, TCP/IP, mocks):

- **Traits**: `ControlChannel`, `DataReceiver`, `DataSender` define interfaces
- **Unix Socket Implementation**: `UnixControlChannel`, `UnixDataReceiver`, `UnixDataSender`
- **Mock Implementation**: `MockControlChannel`, `MockDataReceiver`, `MockDataSender` for testing

This design allows swapping implementations (e.g., Unix sockets for local, TCP for remote) and facilitates testing with mock implementations.

### Process Isolation

The subprocess architecture provides process-level isolation for device operations. See `docs/research/2025-10-process-safety.md` for research on SoapySDR/USB enumeration process safety, including:
- Concurrent enumeration across multiple processes
- Enumeration while streaming in separate processes
- SDRplay driver limitations and workarounds

## Command Design Philosophy

Commands in this IPC protocol follow an RPC/GraphQL mutation style rather than fine-grained setter patterns.

### Atomic Operations

Each command should represent a complete, atomic operation that either fully succeeds or fully fails. Avoid exposing low-level hardware configuration steps as separate commands.

**Good (atomic):**
```rust
ConfigureAndStart {
    channel: usize,
    freq_hz: f64,
    gain_db: f64,
    sample_rate: f64,
}
```

**Bad (fine-grained):**
```rust
SetFrequency { channel: usize, freq_hz: f64 }
SetGain { channel: usize, gain_db: f64 }
SetSampleRate { channel: usize, rate: f64 }
StartStream { channel: usize }
```

### RPC-Style Responses

Responses should be structured result types that include all relevant data from the operation. Always return actual hardware values, not just success/failure, since hardware may not support exact requested values.

**Good (structured response):**
```rust
StreamStarted {
    channel: usize,
    actual_freq: f64,
    actual_gain: f64,
    actual_sample_rate: f64,
}
```

**Bad (boolean response):**
```rust
Success { channel: usize }
```

### When to Add New Commands

Only add commands that represent operations the system will actually use. Don't add commands speculatively or for "completeness."

If you need to change a configuration:
- Stop the stream
- Start a new stream with different parameters

This keeps the state machine simple and makes the protocol easier to reason about.

### Error Handling

Errors should include context about what failed. Use `Option<usize>` for channel when the error might be device-level rather than channel-specific.

```rust
Error {
    channel: Option<usize>,  // None for device-level errors
    message: String,
}
```
