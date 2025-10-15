# Device Pool Architecture

This module manages the allocation and lifecycle of SDR tuners for scanning and listening operations.

## Key Architectural Principle

**The device pool is largely unrelated to the tuners shown in the TUI UI.**

The pool serves a specific, narrow purpose: determining which tuners are allocated to tasks and tracking their activity state (Scanning or Listening). The pool does NOT control what devices appear in the TUI.

## Pool Responsibilities

### What the Pool Does

- **Tuner Allocation**: Provides tuners to tasks that need them (scan, listen)
- **State Tracking**: Tracks whether allocated tuners are Available or Allocated
- **Activity Tracking**: Records whether allocated tuners are Scanning, Listening, or Other
- **Filter Application**: Enforces pool filters to limit which tuners can be allocated
- **Subprocess Management**: Manages device worker subprocesses for allocated tuners

### What the Pool Does NOT Do

- **Device Discovery**: Pool does not enumerate or discover devices (handled by DeviceEnumerationTask)
- **TUI Display**: Pool does not control which devices appear in the TUI (handled by TUI Model via discovery events)
- **Hot-plug Events**: Pool does not send device add/remove events to TUI (handled by discovery service)

## TUI Display vs Pool Allocation

The TUI displays ALL discovered tuners, regardless of pool filter configuration:

```
TUI Display:
- SDRplay Ch0 ✓ (matches pool filter, can be allocated)
- SDRplay Ch1 ✗ (filtered out, cannot be allocated)
- SDRplay Ch2 ✗ (filtered out, cannot be allocated)
- SDRplay Ch3 ✗ (filtered out, cannot be allocated)
- RTL-SDR Ch0 ✗ (different driver, filtered out)

All 5 tuners appear in TUI, but only SDRplay Ch0 can be allocated from pool.
```

The pool filter affects:
- Which tuners can be allocated via `pool.acquire()`
- Which tuners show "Scanning" or "Listening" status

The pool filter does NOT affect:
- Which devices appear in the TUI
- Hot-plug add/remove behavior
- Device enumeration

## Data Flow

```
Hardware Change (plug/unplug)
    ↓
udev/polling detects change
    ↓
DeviceEnumerationTask runs
    ↓
Discovery Event (Added/Removed) sent
    ↓
TUI Model updates (add_device/remove_device)
    ↓
TUI displays updated device list

Separate flow for allocation:

Task needs tuner
    ↓
pool.acquire() with filter
    ↓
Returns tuner if filter allows
    ↓
Task updates tuner activity (Scanning/Listening)
    ↓
TUI queries pool_info to display activity state
```

## Common Confusion Points

### "Why do filtered tuners still appear in TUI?"

Because TUI displays all discovered devices. The filter only affects allocation, not display.

### "Why doesn't removing a device from the pool remove it from TUI?"

The pool doesn't control TUI display. Discovery events control TUI display. When a device is physically removed, the discovery service detects it and sends a removal event to TUI.

### "Why do I see 4 SDRplay tuners but can only use 1?"

Pool filter is restricting allocation to specific tuners. All 4 tuners are valid and discovered, but your pool filter configuration only allows allocation of tuner 0.

## Filter Configuration

Pool filters are set at pool creation:

```rust
let filter = PoolFilter::new()
    .with_driver("sdrplay")
    .with_mode(TuningMode::SingleTuner);

let pool = Pool::new(filter, None);
```

This filter:
- Allows allocation of SDRplay tuners only
- Restricts to single-tuner mode
- Does NOT hide other devices from TUI
- Does NOT prevent other devices from being enumerated

## Hot-plug and the Pool

When a device is physically removed:

1. Discovery service detects the removal
2. DeviceTracker identifies which devices are gone
3. Discovery event sent: `Event::Removed(device_id)`
4. Pool receives the event and removes device metadata
5. TUI receives the event and removes from display
6. Any allocated tuners from that device are automatically returned (via Drop)

The pool is just one recipient of discovery events. It does not orchestrate hot-plug.

## Testing Considerations

When writing tests involving the pool:

- Mock discovery events to test TUI display
- Use pool filters to test allocation behavior
- Don't assume pool state affects TUI display
- Test hot-plug by simulating discovery events, not pool operations
