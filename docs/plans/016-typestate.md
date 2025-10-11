# Plan 016: Additional Typestate Pattern Opportunities

## Guidance for Updates

When updating this plan as work progresses, avoid adding:
- Lists of accomplishments or completion summaries
- Self-aggrandizement or subjective quality assessments
- Rationales and benefits sections (unless specifically requested)
- Speculation about future improvements or possibilities
- Time estimates or risk assessments

Keep updates matter-of-fact and focused on concrete technical details. Simply check off completed tasks and add technical notes as needed.

## Context

The codebase already uses typestate in `scanner_state.rs` with the hybrid enum wrapper pattern. This plan identifies additional areas where typestate could provide compile-time safety.

## Proposal 1: ShutdownCoordinator Typestate

Apply typestate to `src/shutdown.rs` to enforce lifecycle phases and prevent operations in wrong states.

### Implementation

```rust
struct Active { ... }
struct ShuttingDown { ... }
struct Terminated { ... }

struct ShutdownCoordinator<State> {
    token: CancellationToken,
    thread_handles: Mutex<Vec<JoinHandle<()>>>,
    _state: PhantomData<State>,
}

impl ShutdownCoordinator<Active> {
    fn new() -> Self
    fn spawn_sdr_thread(...) -> Result<()>
    fn shutdown(self) -> ShutdownCoordinator<ShuttingDown>
}

impl ShutdownCoordinator<ShuttingDown> {
    fn wait(self) -> Result<ShutdownCoordinator<Terminated>>
}
```

### Tasks
- [x] Define state marker structs (Active, ShuttingDown, Terminated)
- [x] Add state enum to ShutdownCoordinator (using interior mutability pattern)
- [x] Implement state-specific methods on appropriate impl blocks
- [x] Update all ShutdownCoordinator usage sites (backward compatible)
- [x] Update tests to use typed coordinator
- [x] Handle Arc-wrapped coordinator (using Mutex for interior mutability)

## Proposal 2: Pool Lifecycle Typestate

Apply typestate to `src/hardware/pool/state.rs` for clearer shutdown semantics.

### Implementation

```rust
struct Active;
struct ShuttingDown;

struct Pool<State> {
    pool_ref: Arc<Mutex<PoolInner>>,
    filter: PoolFilter,
    _state: PhantomData<State>,
}

impl Pool<Active> {
    fn acquire(&self, ...) -> Result<Tuner>
    fn add_device(&self, ...) -> AddDeviceResult
    fn initiate_shutdown(self) -> Pool<ShuttingDown>
}

impl Pool<ShuttingDown> {
    fn status(&self) -> PoolStatus
}
```

### Tasks
- [x] Define Active and ShuttingDown marker structs
- [x] Add state enum to Pool (using interior mutability pattern)
- [x] Enforce state checks in acquire/add_device methods
- [x] Create shutdown transition method
- [x] Update Pool construction sites (backward compatible)
- [x] Update all Pool usage to use typed version (no changes needed)
- [x] Handle Arc<Pool> sharing (using Mutex for interior mutability)

## Proposal 3: AudioCaptureSink Typestate

Apply typestate to `src/file/iq.rs` to enforce capture workflow.

### Implementation

```rust
struct Buffering;
struct Recording;
struct Completed;

struct AudioCaptureSink<State> {
    // State-specific fields
}

impl AudioCaptureSink<Buffering> {
    fn new(...) -> Result<Self>
    fn add_samples(&mut self, ...)
    fn start_recording(self, squelch_passed: bool) -> Result<AudioCaptureSink<Recording>>
}

impl AudioCaptureSink<Recording> {
    fn write_samples(&mut self, ...)
    fn finalize(self) -> AudioCaptureSink<Completed>
}
```

### Tasks
- [x] Define state marker structs (Buffering, Recording, Completed)
- [x] Separate state-specific fields into each state
- [x] Implement state transition methods
- [x] Update AudioCaptureSink usage sites
- [x] Update tests

## Areas Not Suitable for Typestate

The following were evaluated but are not good candidates:

- Simple discriminated unions (`CandidateStatus`, `TunerState`, `TunerActivity`) - appropriately modeled as enums
- Configuration structs (`ScanningConfig`, `AudioConfig`) - pure data with no state transitions
- Runtime-determined states - many transitions depend on SDR responses
- Collections with heterogeneous states - windows and candidates need different states simultaneously
- `Segment` lifecycle - current channel-based synchronization already solves initialization
- `DiscoveryMode` enum - simple discriminated union is correct model
- Pipeline builder - rustradio's blockchain! macro provides sufficient structure
