# Steps for Simplifying Pool Handling

Refactored device and tuner management to separate cached (pre-enumerated) devices from dynamically discovered devices, and fixed TUI to display all available tuners regardless of pool allocation status.

## Challenges

### Challenge: Duplicate tuner labels with inconsistent formatting

**Goal**: Display tuners with consistent labels in the TUI.

**Failure Mode**: Two "Dev0" tuners appeared in the UI with slightly different labels (one with trailing dash), both showing the same state.

**Solution**: Added `DeviceInfo::tuner(&TunerId)` method to look up tuners consistently. Updated pool status handler to use tuner labels from DeviceInfo rather than constructing them.

**Key Insight**: Labels were being constructed differently in the discovery path (using `tuner.label`) versus the pool status path (using `device.label`). Centralizing label lookup eliminated the inconsistency.

---

### Challenge: Redundant fields in TunerStatus

**Goal**: Simplify TunerStatus structure by removing duplicate data.

**Failure Mode**: TunerStatus contained `model`, `backend`, and `channel_index` fields that were already available via the TunerId.

**Solution**: Removed the redundant fields from TunerStatus. Updated `collect_tuner_statuses()` to only populate `id`, `state`, and `activity`. Removed fallback label construction that relied on these fields.

**Key Insight**: The TunerId already contains the device_id (which includes backend and serial) and channel_index. These fields were only used for fallback label construction, which was unnecessary since device info is available through the cached_devices/devices collections.

---

### Challenge: Mixed device lifecycle management

**Goal**: Separate pre-enumerated devices (that should persist) from dynamically discovered devices (that can be removed).

**Failure Mode**: Both cached and dynamic devices were stored in the same collection, making it unclear which devices should be removed on hot-unplug events.

**Solution**: Added separate `devices: HashMap<DeviceId, DeviceInfo>` collection parallel to `cached_devices`. Updated `add_device()` to check if device is in cached_devices (do nothing) or add to devices (dynamic). Updated `remove_device()` to only remove from devices, never touching cached_devices.

**Key Insight**: Cached devices from initial enumeration (SDRplay) should never be removed, while dynamically discovered devices (RTL-SDR hotplug) need removable lifecycle management.

---

### Challenge: TUI only showing tuners present in pool

**Goal**: Display all available tuners from all devices, not just tuners currently in the pool.

**Failure Mode**: Only one SDRplay tuner appeared in UI, even though the RSPduo has 4 modes (ST, DT, MA, MA8). RTL-SDR tuners discovered via hotplug didn't appear at all.

**Attempts**:
- Initially thought cached_devices weren't being populated, added debug logging to verify
- Discovered cached_devices were being set but tuners still weren't showing

**Solution**: Changed `ActiveTunersUpdated` handler to iterate over all tuners in all devices (both cached_devices and devices) rather than iterating over pool_tuner entries. Pool status is now only used to build pool_info HashMap for state lookups.

**Key Insight**: The pool may not expose all available tuners (e.g., it only allocates one SDRplay tuner at a time), but the TUI should show all tuners from device enumeration. Pool status provides state information (Available/Scanning/Listening), but device collections are the authoritative source for what tuners exist.
