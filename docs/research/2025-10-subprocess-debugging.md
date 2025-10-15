# Subprocess Debugging

This document describes how to debug and troubleshoot subprocess workers (enumeration and device workers).

## Architecture Overview

The scanner uses subprocess isolation for SDR device operations:

- **Enumeration workers**: Short-lived processes that enumerate devices for a specific backend
- **Device workers**: Long-lived processes that stream I/Q data from devices

Each subprocess communicates with the parent via Unix domain sockets (control and data channels).

## Worker Logging

Worker processes log to files only when the parent process is logging to a file. Worker log paths are automatically derived from the parent log file path.

### Log Path Derivation

If parent logs to `/path/to/scanner.log`:
- Enumeration workers: `/path/to/scanner-enum-{backend}.log`
- Device workers: `/path/to/scanner-worker-{device_id}-{timestamp}.log`

Example:
```bash
# Parent logs to /tmp/scanner.log
scanner scan --band fm --log-file /tmp/scanner.log

# Worker logs appear as:
# /tmp/scanner-enum-soapy.log          (enumeration worker)
# /tmp/scanner-worker-sdrplay-123-456.log  (device worker)
```

### Enabling Worker Logging

Simply pass `--log-file` to the scanner:

```bash
# TUI mode with worker logging
scanner scan --band fm --log-file /tmp/scanner.log

# Headless mode with worker logging
scanner scan --stations 88.9e6 --duration 3 --headless --log-file /tmp/scanner.log
```

## Manual Testing

The `worker` command is hidden from normal help output but can be invoked directly for debugging.

### Test Enumeration Worker

```bash
scanner worker enumerate --backend soapy --socket-path /tmp/test.sock
```

### Test Device Worker

```bash
scanner worker device \
  --device-id-str '{"Driver":{"driver":"sdrplay","serial":"123456"}}' \
  --control-socket-path /tmp/ctl.sock \
  --data-socket-path /tmp/dat.sock \
  --log-file /tmp/worker.log
```

## Troubleshooting

### Worker Won't Start

Check:
- Device permissions (SDR hardware access)
- SoapySDR driver installed
- Socket directory writable
- Review worker logs for startup errors
- Test worker command manually

### Worker Crashes

Check:
- Worker logs for panic/error messages
- Verify device args are correct
- Check for resource conflicts (device already open)
- Verify driver version compatibility

### IPC Communication Issues

Check:
- Socket files exist in `/tmp/scanner-*.sock`
- Process is still running: `ps aux | grep "scanner worker"`
- Worker logs show message traffic
- Parent logs show IPC errors at warn level

### Zombie Processes

Check for zombies after shutdown:
```bash
scanner scan --band fm
# Ctrl-C
sleep 2
ps aux | grep "scanner worker" | grep defunct  # Should be empty
```

### Socket Cleanup

Check for stale sockets:
```bash
ls -la /tmp/scanner-*.sock
```

Sockets should be cleaned up after shutdown. If not, indicates a shutdown bug.

## Debug Build Features

In debug builds (`cargo build`), subprocess state validation is enabled:

- Socket existence checks before operations
- Process liveness checks
- Panic on unexpected subprocess death
- Assert socket paths exist

This validation is disabled in release builds for performance.

## Common Issues

### Issue: Enumeration worker fails silently

Solution:
1. Check exit status manually: `scanner worker enumerate; echo $?`
2. Enable logging to see errors
3. Verify backend driver is installed

### Issue: Device worker not responding

Solution:
1. Check if subprocess is running: `ps aux | grep "scanner worker device"`
2. Check worker logs for errors
3. Verify socket files exist
4. Check for device access permission issues

### Issue: Worker logs not appearing

Check:
1. Parent process is using `--log-file` flag
2. Log directory is writable
3. Check parent log file exists

## Logging Levels

Worker processes respect the `RUST_LOG` environment variable:

```bash
# Enable debug logging for workers
RUST_LOG=debug scanner scan --band fm --log-file /tmp/scanner.log

# Trace-level logging (very verbose)
RUST_LOG=trace scanner scan --stations 88.9e6 --duration 3 --log-file /tmp/scanner.log
```

## Related Documentation

- Subprocess IPC design: `docs/plans/008-subprocess-ipc.md`
- Process safety research: `docs/research/2025-10-process-safety.md`
