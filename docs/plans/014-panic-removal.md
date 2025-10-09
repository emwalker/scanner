# 014: Panic Removal Plan

**Status**: Proposed
**Date**: 2025-10-08

## Overview

Systematic plan to eliminate all panics from production code. The goal is that production code should **never panic**, even on programmer errors - instead, it should return `Result` to allow graceful degradation, better error reporting, and clean shutdown.

## Analysis Summary

- **Total files with `.unwrap()`**: 31 files
- **Total files with `.expect()`**: 14 files
- **Total files with explicit `panic!()`**: 4 files
- **Total files with `unreachable!()`**: 0 files

**Categorization:**
- Test code panics: Acceptable (13 files)
- Mutex lock unwraps: Needs poisoned lock handling (8 files)
- Critical production panics: Must fix immediately (4 locations)
- Floating point comparisons: Needs NaN handling (multiple locations)

---

## Categories of Panics

### 1. Test Code (Acceptable - No Changes Needed)

These files contain test assertions that are expected to panic on failure:

- `src/ui/tui/model/tests.rs` - Test assertions
- `src/hardware/pool/tests.rs` - Test expectations
- `src/hardware/types.rs:265` - Test panic in match arm
- `src/hardware/pool/types.rs:42` - Test panic
- `src/ui/mod.rs:189,252,256,260,323,368,372` - Test panics for event type validation
- `src/testing/benchmark_datasets.rs`
- `src/testing/variance_measurement.rs`
- `src/testing/detection_regression_tests.rs`

**Rationale**: Test code should panic to clearly indicate test failures.

---

### 2. Mutex Lock Unwraps (Needs Poisoned Lock Handling)

**Problem**: Mutex locks can be poisoned if a thread panics while holding the lock. Current code uses `.unwrap()` which will panic if the mutex is poisoned.

#### Files with mutex lock unwraps:

**`src/hardware/pool/tuner.rs:78,96,114`**
```rust
// Current:
let device = self.device.lock().unwrap();
let mut device = self.device.lock().unwrap();

// Issue: Panics if mutex is poisoned
```

**`src/signal/mod.rs:21`**
```rust
// Current:
let mut processed = PROCESSED_FREQUENCIES.lock().unwrap();

// Issue: Global mutex can be poisoned
```

**`src/pipeline/mod.rs:133,327`**
```rust
// Current:
let processed = crate::signal::PROCESSED_FREQUENCIES.lock().unwrap();
let mut processed = crate::signal::PROCESSED_FREQUENCIES.lock().unwrap();
```

**`src/main_thread.rs:689,697,704,723`** (test helpers - acceptable)

#### Proposed solution:

**Option 1: Handle poisoned mutexes explicitly**
```rust
let device = self.device.lock()
    .map_err(|e| {
        error!("Mutex poisoned - recovering");
        ScannerError::Custom("Internal error: mutex poisoned".to_string())
    })?;
```

**Option 2: Use into_inner() to recover poisoned data**
```rust
let device = match self.device.lock() {
    Ok(guard) => guard,
    Err(poisoned) => {
        warn!("Mutex poisoned - recovering data");
        poisoned.into_inner()
    }
};
```

**Option 3: Use parking_lot::Mutex** (doesn't poison)
```rust
// parking_lot mutexes never poison, they just unlock on panic
use parking_lot::Mutex;
```

**Recommendation**: Use Option 3 (parking_lot) for new code, Option 2 for existing code to avoid breaking changes.

---

### 3. Critical Production Panics (Must Fix)

These are the highest priority fixes:

#### **`src/hardware/pool/lifecycle.rs:214,244,245,301`**

**Current code:**
```rust
let device_entry = inner.devices.get(&entry.device_id).unwrap();
```

**Why this is critical:**
- Assumes device always exists for tuner
- Can panic if device removed between operations
- No recovery path

**Fixed version:**
```rust
let device_entry = inner.devices.get(&entry.device_id)
    .ok_or_else(|| {
        error!(
            device_id = ?entry.device_id,
            tuner_id = ?tuner_id,
            "Critical: device not found for tuner - data structure inconsistency"
        );
        ScannerError::InternalInconsistency {
            message: format!("Device {} not found for tuner", entry.device_id)
        }
    })?;
```

**Line 244 specifically:**
```rust
// Current:
let entry = inner.available_tuners.remove(&tuner_id).unwrap();

// Fixed:
let entry = inner.available_tuners.remove(&tuner_id)
    .ok_or_else(|| {
        error!(tuner_id = ?tuner_id, "Tuner disappeared during acquisition");
        ScannerError::TunerNotFound { tuner_id }
    })?;
```

#### **`src/hardware/device.rs:99`**

**Current code (in documentation example):**
```rust
/// let device_args = raw.downcast::<String>().unwrap();
```

**Fixed version:**
```rust
/// let device_args = raw.downcast::<String>()
///     .map_err(|_| ScannerError::InvalidDeviceArgs)?;
```

**Note**: This is in a doc comment example, but should still show proper error handling.

---

### 4. Floating Point Comparisons (Needs NaN Handling)

**Files affected:**
- `src/signal/mod.rs:128,133,143,252`
- `src/signal/peaks/averaging.rs`
- `src/signal/peaks/multi_frame.rs`

**Current pattern:**
```rust
sorted_peaks.sort_by(|a, b| a.frequency_hz.partial_cmp(&b).unwrap());
.max_by(|a, b| a.partial_cmp(b).unwrap())
```

**Problem**: `partial_cmp` returns `None` for NaN values, causing panic.

**Fixed version:**
```rust
// Sort with NaN handling
sorted_peaks.sort_by(|a, b| {
    a.frequency_hz.partial_cmp(&b.frequency_hz)
        .unwrap_or(std::cmp::Ordering::Equal)
});

// Or better - assert no NaN in debug builds
sorted_peaks.sort_by(|a, b| {
    debug_assert!(!a.frequency_hz.is_nan() && !b.frequency_hz.is_nan());
    a.frequency_hz.partial_cmp(&b.frequency_hz)
        .unwrap_or(std::cmp::Ordering::Equal)
});

// Max with NaN handling
peaks.iter()
    .max_by(|a, b| {
        a.magnitude.partial_cmp(&b.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
```

**Alternative - use total_cmp for f64 (Rust 1.62+):**
```rust
sorted_peaks.sort_by(|a, b| a.frequency_hz.total_cmp(&b.frequency_hz));
```

**Note**: `total_cmp` treats NaN as equal and orders it at the end.

---

### 5. I/O Operation Unwraps

**`src/signal/mod.rs:307`**
```rust
// Current:
std::io::Write::flush(&mut std::io::stdout()).unwrap();

// Fixed:
if let Err(e) = std::io::Write::flush(&mut std::io::stdout()) {
    warn!("Failed to flush stdout: {}", e);
}
```

**Rationale**: Flushing stdout can fail (e.g., broken pipe), but this is not critical.

---

## New Error Variants Needed

To support proper error handling, add these variants to `ScannerError`:

```rust
pub enum ScannerError {
    // Existing variants...

    // New variants for panic removal:

    /// Internal data structure inconsistency (e.g., device missing for tuner)
    InternalInconsistency {
        message: String,
    },

    /// Tuner not found when expected
    TunerNotFound {
        tuner_id: TunerId,
    },

    /// Device not found when expected
    DeviceNotFound {
        device_id: String,
    },

    /// Mutex poisoned (thread panicked while holding lock)
    MutexPoisoned {
        context: String,
    },

    /// Invalid device arguments during downcast
    InvalidDeviceArgs,
}
```

---

## Implementation Plan

### Phase 1: Critical Safety Fixes

**1. Fix HashMap unwraps in `src/hardware/pool/lifecycle.rs`**
   - Lines 214, 244, 245, 301
   - Add new error variants
   - Add defensive logging
   - Test error paths

**2. Fix floating point comparisons**
   - Use `total_cmp()` for f64
   - Add debug assertions for NaN detection
   - Review all `partial_cmp().unwrap()` calls

**3. Add new error variants to `ScannerError`**
   - Define in `src/core/types.rs`
   - Add Display implementations
   - Update From conversions

### Phase 2: Mutex Robustness

**4. Replace mutex lock unwraps**
   - `src/hardware/pool/tuner.rs` - use `into_inner()` recovery
   - `src/signal/mod.rs` - handle poisoned PROCESSED_FREQUENCIES
   - `src/pipeline/mod.rs` - handle poisoned locks
   - Consider migrating to `parking_lot::Mutex` long-term

### Phase 3: Polish

**5. Fix I/O operation unwraps**
   - `src/signal/mod.rs:307` - flush errors

**6. Update documentation examples**
   - `src/hardware/device.rs:99` - show proper error handling

**7. Add comprehensive tests**
   - Test all new error paths
   - Test mutex poison recovery
   - Integration tests for error propagation

---

## Testing Strategy

### Unit Tests

**Test poisoned mutex recovery:**
```rust
#[test]
fn test_poisoned_mutex_recovery() {
    let mutex = Arc::new(Mutex::new(0));

    // Poison the mutex
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::panic::catch_unwind(|| {
        let _guard = mutex_clone.lock().unwrap();
        panic!("Intentional panic to poison mutex");
    });

    // Test recovery
    let result = match mutex.lock() {
        Ok(_) => panic!("Mutex should be poisoned"),
        Err(poisoned) => {
            let _data = poisoned.into_inner();
            Ok(())
        }
    };

    assert!(result.is_ok());
}
```

**Test HashMap lookup errors:**
```rust
#[test]
fn test_device_not_found_error() {
    // Setup pool with missing device
    // ...

    let result = pool.try_acquire(&requirements);

    match result {
        Err(ScannerError::DeviceNotFound { device_id }) => {
            // Expected error
        }
        _ => panic!("Expected DeviceNotFound error"),
    }
}
```

### Integration Tests

**Test error propagation:**
- Verify errors propagate up the stack
- Verify logging occurs at appropriate levels
- Verify application continues or shuts down gracefully

---

## Acceptance Criteria

### Must Have

- [ ] No `.unwrap()` in production code paths (test code excluded)
- [ ] No `.expect()` in production code paths (test code excluded)
- [ ] No explicit `panic!()` in production code paths (test code excluded)
- [ ] All mutex locks handle poisoned state
- [ ] All floating point comparisons handle NaN
- [ ] All new error variants documented
- [ ] All error paths tested

### Should Have

- [ ] Migration plan to `parking_lot::Mutex` for cleaner API
- [ ] Comprehensive error logging at appropriate levels
- [ ] Debug assertions for "impossible" states
- [ ] Documentation of error recovery strategies

### Nice to Have

- [ ] Fuzzing tests for error paths
- [ ] Chaos engineering tests (inject failures)
- [ ] Metrics for error rates in production

---

## Risks and Mitigation

### Risk: Breaking existing callers

**Mitigation**:
- Most functions already return `Result`
- New error variants are additive
- Comprehensive testing before merge

### Risk: Performance impact from error checking

**Mitigation**:
- Error path overhead is minimal (single branch)
- Most "hot paths" don't involve unwraps
- Profile before/after to verify

### Risk: Missing some unwraps

**Mitigation**:
- Comprehensive grep audit completed
- CI should include `#![deny(clippy::unwrap_used)]` eventually
- Regular audits as part of code review

---

## Future Work

### Clippy Lints

Consider adding these lints to prevent regressions:

```rust
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
```

**Note**: Start with `warn` to identify issues, upgrade to `deny` once codebase is clean.

### Error Context

Consider using a crate like `anyhow` or `thiserror` for richer error context:

```rust
use anyhow::Context;

device.tune(freq)
    .context("Failed to tune device to {freq} Hz")?;
```

---

## References

**Rust Error Handling Best Practices:**
- https://doc.rust-lang.org/book/ch09-00-error-handling.html
- https://doc.rust-lang.org/std/result/
- https://doc.rust-lang.org/std/sync/struct.PoisonError.html

**Mutex Poisoning:**
- https://doc.rust-lang.org/std/sync/struct.Mutex.html#poisoning
- https://docs.rs/parking_lot/ (alternative without poisoning)

**Floating Point Comparisons:**
- https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp
- https://doc.rust-lang.org/std/cmp/enum.Ordering.html

---

## Notes

- Analysis conducted: 2025-10-08
- Total panics identified: ~50+ across 31 files
- Production panics (critical): 4 locations
- Test panics (acceptable): 13 files
- Estimated effort: 2-3 focused sessions
