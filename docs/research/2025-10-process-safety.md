# Process Safety Research: SoapySDR and USB Device Enumeration

## SoapySDR Thread and Process Safety

### Enumeration
- `Device::enumerate()` is thread-safe within a single process (protected by internal mutex)
- Multiple processes can safely call `enumerate()` concurrently
- Each process has isolated memory space - no shared global state in data segments
- Enumeration is a read-only operation querying already-enumerated USB devices via kernel interfaces

### Device Operations
- Device methods intended for single-thread access per device
- Most implementations use mutex locks around control transactions (USB/network)
- Streams should not be accessed by multiple threads concurrently

## SDRplay Driver Limitations

The mirsdrapi-rsp.h driver used by SoapySDRPlay has architectural limitations:
- Uses internal global handle instead of per-device handles
- Only supports one open device per process
- Calling `SoapySDRDevice_enumerate` while another device is streaming causes the streaming device to error and output zeros
- Opening an SDRplay device prevents subsequent enumerations in the same process from seeing any SDRplay devices

Industry workaround: Spawn SoapySDRServer per device for process isolation.

### Observed Behavior (2025-10-11)

When testing with RSPduo hardware, we observed:
1. First enumeration in a fresh worker subprocess correctly finds all 4 RSPduo modes (ST, DT, MA, MA8) plus RTL-SDR
2. After opening one RSPduo device via `SoapySDRDevice_make()` in the parent process
3. Subsequent enumerations in new worker subprocesses only find RTL-SDR devices, not SDRplay devices

This suggests the SDRplay driver holds system-level state (likely via shared library or kernel driver state) that persists across process boundaries and prevents enumeration when a device is open in another process.

### Workaround Strategy

For discovery service to work correctly with SDRplay devices:
- Enumerate all devices once at startup before opening any devices
- Cache the device list and provide it to the discovery service
- Discovery service monitors for hardware changes (USB hotplug) but doesn't re-enumerate SDRplay devices
- Alternatively: Don't open devices until needed, keep discovery service running first

## USB/libusb Behavior

### Enumeration
- `libusb_get_device_list()` queries kernel interfaces for already-enumerated devices
- Reference counting (via `libusb_ref_device()`/`libusb_unref_device()`) manages device lifecycle
- Enumeration operates separately from I/O operations via device handles
- Multiple processes can enumerate concurrently - operations are read-only from kernel perspective

### Low-Level Constraints
- Port resets serialized via "enumeration lock" per host controller
- Only one device can use USB address 0 during initial plug-in enumeration
- This serialization occurs at kernel level during device hotplug, not during `enumerate()` API calls

## Subprocess Architecture Safety

### Enumeration in Separate Process
- Safe to enumerate devices in one subprocess while streaming in another subprocess
- Process isolation prevents SDRplay global handle interference
- Each subprocess operates independently with isolated memory

### Per-Device Worker Subprocess
- One subprocess per device eliminates SDRplay's one-handle-per-process limitation
- Enumeration subprocess exits before device workers start
- No temporal overlap between enumeration and streaming in same process

## Sources
- SoapySDR GitHub Issue #111 (thread safety documentation)
- libusb documentation (device enumeration and multi-threading)
- Pothos Users mailing list (SDRplay multi-device issues)
- USB specification (enumeration lock and address 0 constraint)
