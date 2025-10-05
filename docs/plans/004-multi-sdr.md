# Plan 004: Multi-SDR Architecture

**Date**: October 2025
**Status**: Planning Complete - Split into Smaller Plans
**Related Plans**: `001-async.md` (Hybrid Architecture), `003-structured-concurrency-shutdown.md`

## Executive Summary

Transform the scanner from single-SDR operation to multi-SDR orchestration with dynamic device discovery.

## Goals

### Primary
1. **Parallel Operations**: Scan on SDR #1 while streaming audio on SDR #2
2. **Dynamic Discovery**: Detect devices plugged/unplugged at runtime
3. **Task-Based**: Operations (scanning, audio, P25 control) as independent tasks
4. **Capability-Aware**: Match tasks to suitable devices automatically

### Future Capabilities
- P25 trunked radio: control channel monitoring + dynamic voice channel following
- Direction finding: multiple SDRs working together
- Network SDRs: remote devices over network

## Current Limitations

**Single-device operation**:
- Operations are mutually exclusive (scan OR listen, not both)
- Devices enumerated once at startup
- No hot-plug support
- Hard-coded device selection

**After implementation**:
- Multiple simultaneous operations (scan AND listen)
- Hot-plug: devices appear/disappear at runtime
- Automatic device assignment based on task requirements
- Graceful degradation (works with 1 device, scales to N)

## Implementation Plans

This plan is split into 6 smaller plans to be implemented in dependency order:

### 1. [Plan 005: Backend Abstraction](005-backend-abstraction.md)
**Time**: 4-6 hours | **Dependencies**: None

Isolate SoapySDR dependency behind trait abstraction. Enables future migration to native Rust drivers (Seify, rtl-sdr-rs).

### 2. [Plan 006: Device Discovery](006-device-discovery.md)
**Time**: 5-6 hours | **Dependencies**: Plan 005

Platform-optimized hot-plug detection (udev on Linux, polling elsewhere). Emits device add/remove events.

### 3. [Plan 007: Device Pool](007-device-pool.md)
**Time**: 7-8 hours | **Dependencies**: Plans 005, 006

RAII-based device management with automatic return-to-pool. Capability-based allocation matches devices to task requirements.

### 4. [Plan 008: Subprocess IPC](008-subprocess-ipc.md)
**Time**: 9-10 hours | **Dependencies**: Plans 005, 007

Universal subprocess isolation for ALL devices (not just SDRplay). Custom Unix socket IPC provides crash isolation and memory safety.

### 5. [Plan 009: Task Abstraction](009-task-abstraction.md)
**Time**: 12-13 hours | **Dependencies**: Plans 005, 007

Extract operations (scanning, audio) into independent tasks. TaskScheduler automatically assigns tasks to available devices.

### 6. [Plan 010: Multi-SDR Orchestration](010-multi-sdr-orchestration.md)
**Time**: 14-15 hours | **Dependencies**: All previous plans

Integrate all components. Update TUI to show real-time device and task status. End-to-end testing and examples.

## Key Architectural Decisions

Based on comprehensive research (October 2025):

1. **Custom device pool** (not SoapyMultiSDR)
   - SoapyMultiSDR lacks hot-plug support and task abstraction

2. **Backend abstraction layer**
   - Keeps SoapySDR at a distance
   - Enables migration to native Rust drivers when ready

3. **Universal subprocess isolation**
   - ALL devices run in isolated subprocesses (simple, consistent)
   - Custom Unix socket IPC (not SoapySDRServer)
   - ~25μs latency overhead (acceptable)

4. **RAII resource management**
   - Devices automatically return to pool (impossible to leak)
   - Rust ownership guarantees correctness

## Expected Behaviors

### With 1 Device
- Backward compatible with current behavior
- Scan and audio are mutually exclusive
- No changes to user experience

### With 2+ Devices
- **New**: Scan while listening to audio simultaneously
- **New**: Hot-plug devices during operation
- **New**: Automatic device assignment

### Example TUI Output
```
╭─ SDRplay RSPduo ───────────────────────────╮
│ Scanning: FM Band (88-108 MHz)             │
│ Status: Window 15/20 • Running 45.2s       │
╰────────────────────────────────────────────╯

╭─ RTL-SDR (00000001) ───────────────────────╮
│ Audio: 88.9 MHz FM                         │
│ Status: Playing • Running 12.5s            │
╰────────────────────────────────────────────╯

╭─ HackRF One ───────────────────────────────╮
│ Idle • Available                           │
╰────────────────────────────────────────────╯
```

## Implementation Order

Each plan builds on previous ones:

```
005 (Backend Abstraction)
  ↓
006 (Discovery) + 007 (Pool)
  ↓
008 (Subprocess IPC) + 009 (Tasks)
  ↓
010 (Orchestration)
```

## References

- Research findings: See individual plans for details
- Plan 003: Structured Concurrency (shutdown coordination)
- Plan 001: Hybrid Architecture (future async coordination)
