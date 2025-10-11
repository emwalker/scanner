# Passing State to Callback

Modified the Pool state change callback interface to pass PoolStatus directly to listeners instead of requiring them to query pool.status() themselves.

## Challenges

### Challenge: Circular Dependency with Arc<Pool>

**Goal**: Pass PoolStatus to callbacks when tuners are returned to the pool.

**Failure Mode**: Initially tried making Pool methods use `self: &Arc<Self>` so that Tuner could hold `Arc<Pool>` and call `pool.notify_state_change()` on drop. This created architectural issues:
- The `&Arc<Self>` receiver pattern is unconventional and harder to reason about
- Required wrapping all Pool instances in Arc throughout the test suite
- Made the TunerProvider trait implementation problematic

**Attempts**:
- Implemented TunerProvider for `Arc<Pool>` instead of `Pool` - failed because Rust's unsized coercion requires traits to be implemented on the concrete type, not the Arc wrapper
- Tried explicit type casting from `Arc<Pool>` to `Arc<dyn TunerProvider>` - failed for the same reason
- Considered removing TunerProvider abstraction entirely - rejected to preserve dependency injection capability

**Solution**: Changed Tuner to hold an `on_return` closure instead of `Arc<Pool>`. The closure is created in `allocate_tuner` and captures:
- The state change callbacks (`Arc<Mutex<Vec<...>>>`)
- The pool inner reference (`Arc<Mutex<PoolInner>>`)
- The shutdown mode flag (`Arc<AtomicBool>`)

When Tuner drops, it calls the closure which computes the current PoolStatus from PoolInner and invokes all callbacks with the status.

**Key Insight**: Instead of having Tuner call back into Pool (which requires Arc<Pool>), extract the notification logic into a closure that captures only what's needed. This decouples Tuner from Pool while maintaining the callback mechanism.

### Challenge: TunerProvider Implementation for Unsized Coercion

**Goal**: Allow `Arc<Pool>` to automatically coerce to `Arc<dyn TunerProvider>` when passed to Window.

**Failure Mode**: After implementing TunerProvider for `Arc<Pool>`, got compilation errors: "the trait `TunerProvider` is not satisfied" when trying to convert `Arc<Pool>` to `Arc<dyn TunerProvider>`.

**Attempts**:
- Implemented trait for `Arc<Pool>` - failed because unsized coercion doesn't work with traits implemented on Arc wrappers
- Tried implementing for both `Pool` and `Arc<Pool>` - still failed
- Added explicit type casts with `as Arc<dyn TunerProvider>` - failed for same reason

**Solution**: Implement TunerProvider directly for Pool (not Arc<Pool>). Rust's unsized coercion automatically handles converting `Arc<Pool>` to `Arc<dyn TunerProvider>` when the trait is implemented on the concrete type.

```rust
impl TunerProvider for crate::hardware::pool::Pool {
    fn acquire(&self, requirements: &TaskRequirements, activity: TunerActivity) -> Result<Tuner> {
        self.acquire(requirements, activity)
    }
    // ...
}
```

**Key Insight**: Rust's unsized coercion for trait objects (`Arc<T>` → `Arc<dyn Trait>`) only works when the trait is implemented on `T`, not on `Arc<T>`. The Deref coercion happens automatically at call sites.
