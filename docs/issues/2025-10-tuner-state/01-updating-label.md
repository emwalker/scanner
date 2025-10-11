# Fix for Scanning/Listening Label

Fixed regression where TUI tuner status labels weren't updating to show "Scanning" or "Listening" states.

## Challenges

### Challenge: DeviceId Mismatch Between Discovery and Pool

**Goal**: Ensure discovered devices and pool-opened devices have identical DeviceIds so TUI can match tuner states.

**Failure Mode**: Discovery produced `DeviceId { backend: "sdrplay", serial: "2301034E34:ST" }` while Pool produced `DeviceId { backend: "SDRplay", serial: "sdrplay_api_api_version=3.150000..." }`. TUI couldn't match devices, so status updates never applied to the right tuner.

**Attempts**:
- Tried modifying only the Mock backend to use consistent DeviceId creation - didn't fix the real bug with SoapySDR
- Created helpers in test code only - didn't affect production behavior

**Solution**: Added `DeviceId::normalize_driver()` that lowercases driver names in `DeviceId::from_serial()`. Updated both `Soapy::open_device()` and `Mock::open_device()` to pass driver/serial parameters through to device constructors, ensuring all backends use the same DeviceId creation logic.

**Key Insight**: SoapySDR's enumeration API returns different driver capitalization ("sdrplay") than the opened device metadata ("SDRplay"). The solution required normalizing at the point of DeviceId creation, not trying to match inconsistent strings later.

### Challenge: ActiveTunersUpdated Events Not Sent at Right Time

**Goal**: Send `ActiveTunersUpdated` events to TUI when tuners are actually acquired or released, not before/after.

**Failure Mode**: "Listening" status worked but "Scanning" status never appeared. Events were being sent from MainThread before tuners were acquired from the Pool.

**Attempts**:
- Added `send_active_tuners_update()` calls at start of scanning loops with `if station_idx == 0` conditional - rejected as code smell, doesn't fire when tuner is actually acquired
- Added `send_active_tuners_update()` calls with `if window_num == 1` conditional - rejected as workaround instead of architectural fix

**Solution**: Implemented observer pattern with callbacks in Pool. Added `on_state_change: Arc<Mutex<Vec<Box<dyn Fn() + Send + Sync>>>>` to Pool struct, invoked callbacks in `try_acquire()` after allocation and in `Tuner::drop()` after return. MainThread registers callback in `with_tui_event_sender()` that sends `ActiveTunersUpdated` events. Removed all manual event sending from command handlers.

**Key Insight**: The bug wasn't in the TUI event handling - it was in when events were sent. Events must be sent from inside Pool when tuner state actually changes, not from outside code that guesses when changes happen. Callbacks are the right pattern for decoupling Pool lifecycle from UI updates.

### Challenge: Passing Callbacks Through Tuner RAII Wrapper

**Goal**: Invoke callbacks when Tuner is dropped and returned to pool.

**Failure Mode**: Pool could invoke callbacks on acquire, but Tuner::drop() had no reference to the callbacks to invoke on release.

**Attempts**: None - the solution was straightforward once the problem was identified.

**Solution**: Added `on_state_change` field to Tuner struct, passed from Pool during allocation in `allocate_tuner()`. Modified `PoolInner::return_tuner()` to return bool indicating success. Tuner::drop() checks return value and invokes callbacks if tuner was successfully returned.

**Key Insight**: RAII wrappers need access to notification mechanisms if they're responsible for state changes. The Tuner already held references to Pool internals, adding callback reference followed the same pattern.
