# Device Disconnect/Reconnect State Machine Research

## Problem Statement

When an SDRplay device is unplugged during operation (listening to a station or processing a scanning window) and then plugged back in, the discovery system successfully detects it, but the interrupted operation does not resume. This requires modeling and testing a complex state machine with fault recovery capabilities.

## Research Overview

This research examines patterns for modeling complex state machines with fault tolerance, state preservation, and resume capabilities, along with strategies for testing such systems.

## Modeling Approaches

### Hierarchical State Machines (HSM / UML Statecharts)

Hierarchical state machines extend finite state machines with nested states, avoiding state explosion.

#### Key Concepts

The most important innovation of UML state machines over traditional FSMs is hierarchically nested states. The value of state nesting lies in avoiding repetitions that are inevitable in traditional "flat" FSM formalism and are the main reason for state-transition explosion in FSMs.

If a system is in a nested state (substate), it also implicitly is in the surrounding state (superstate). If the substate does not prescribe how to handle an event, it is automatically handled at the higher level context of the superstate.

#### Practical Application

Hierarchical State Machines are widely used in spacecraft flight software and embedded systems design. The traditional approach involves graphical languages (UML statecharts) from which implementation code is generated.

#### Structure Example

```
DeviceState
├── Disconnected
│   ├── WaitingForDevice
│   └── DeviceDetected
└── Connected
    ├── Idle
    ├── Scanning
    │   └── ProcessingWindow
    └── Listening
        ├── Tuning
        └── Playing
```

When device disconnects from ProcessingWindow state, the system remembers it was in Scanning→ProcessingWindow. Upon reconnection, it can return to that state.

### Typestate Pattern (Rust-Specific)

The typestate pattern moves properties of state into the type level that the compiler can check ahead-of-time, moving certain types of errors from run-time to compile-time.

#### Implementation Characteristics

Rust's strict type system can lock down possible state transitions at compile-time. Ideally:
- State machines should be in one state at a time
- Each state should have its own associated values
- Transitioning between states should have well-defined semantics
- Only explicitly defined transitions should be permitted
- Changing states should consume the previous state

#### Rust Libraries

**statig**: Hierarchical state machines for event-driven systems. The typestate pattern is useful for API design to enforce operation validity at compile time, but statig is designed to model dynamic systems where events originate externally and operation order is determined at runtime.

**rust-bakery/machine**: Type-checked state machine library providing macros for defining state machines with compile-time transition checking.

### Event Sourcing with Snapshots

Event sourcing records all state transitions as events. State is reconstructed by replaying events from the beginning or from a checkpoint snapshot.

#### Snapshot Strategy

Snapshots solve performance issues of replaying many events by recording aggregate state at a point in time, which serves as a checkpoint to replay only events since the snapshot. A stateful actor is recovered by replaying stored events, either from full history or from a snapshot checkpoint, dramatically reducing recovery times.

#### Implementation Considerations

Event sourcing implementations often incorporate periodic snapshots as checkpoints, combining both patterns to balance performance and auditability. Snapshots should be implemented only when metrics confirm that command processing time is unsatisfactory and related to event replay time.

Sources generally recommend starting without snapshots and adding them only when performance metrics indicate necessity.

### State Machine Replication (SMR)

State machine replication is a general method for implementing fault-tolerant services by replicating servers and coordinating client interactions with server replicas. Systems based on SMR typically use finite-state machines to simplify error recovery.

#### Fault Tolerance Requirements

For fault tolerance supporting F failures, a system must have 2F+1 copies (replicas). Determinism is an ideal characteristic for providing fault tolerance.

#### Failure Models

The state machine approach describes protocols for Byzantine and fail-stop failure models.

### Saga Pattern

A saga is a sequence of local transactions where each transaction updates state and publishes a message to trigger the next transaction. If a transaction fails, the saga executes compensating transactions that undo changes made by preceding transactions.

#### Error Handling

The Saga pattern is a fail-over and compensating handling pattern that executes corresponding compensating actions to return to initial state when failures occur. When exceptions occur, the engine reverses execution of compensation nodes for successful nodes to roll back the transaction.

Key considerations:
- Not all exceptions require rollback; custom handling methods may exist
- Both requests and compensating requests must be idempotent
- State machines implement error handling mechanisms to trigger compensating transactions

#### State Machine Implementation

The saga gains eventual consistency by calling compensation steps in reverse order using an orchestration approach with a centralized state machine. State machines make distributed transactions more resilient by providing persistent state for recovery and resume processing during outages.

### Actor Model with FSM

The actor model provides natural fault tolerance through its separation approach to concurrency, where actors operate in isolation, maintaining their own state and communicating only through asynchronous messages.

#### Combining Patterns

FSMs are described in Erlang design principles, and frameworks like Akka provide support for building finite state machine actors. FSMs are effective for tracking long-running workflows with predetermined action sequences.

Actor Oriented Design Patterns utilize the Actor Model's characteristics to solve problems in concurrency, distributed computing, and fault tolerance, offering strategies to achieve scalable, maintainable, and resilient systems.

## Device Connection State Patterns

### Connection State Phases

Phase transitions can move back to Disconnected at all times but will only advance a single step forward:
- **Disconnected**: Initial state before connection
- **Connected**: After establishing connection
- **Synchronized/Acknowledged**: After handshake and validation
- **Ready**: Fully operational state

### Hierarchical State Machine for BLE

The Hierarchical State Machine pattern is suitable for handling communication, broadly split into Disconnected and Connected states. If errors occur in either major state, it switches to Error state.

### Resume After Reconnection

While a client is disconnected, it is safe to assume that when connection is reestablished, all channels remain attached and messages published whilst disconnected will be delivered upon reconnection. This is possible because systems can hold connection state for a period (e.g., 2 minutes) following abrupt disconnection.

For MQTT-based systems, to resume a session and retain messages, set clean_session = false and reconnect with the same identity.

### USB Device State Management

To resume a device, the driver's attach entry point is called with DDI_RESUME. When handling DDI_SUSPEND, clean up device state and driver state as much as necessary for clean resume later.

## Testing Strategies

### Property-Based Testing

Property-based testing generates random inputs for functions to check certain properties and automatically shrinks inputs to minimal failing cases.

#### Rust Tools

**QuickCheck**: Generates and shrinks values based on type alone. Can only define one generator and shrinker per type. Custom generation strategies require wrapping in newtypes and implementing traits manually.

**Proptest**: Uses explicit Strategy objects. Can define arbitrarily many different strategies for the same type with plenty built-in. Inspired by Python's Hypothesis framework.

#### State Machine Testing

The proptest-state-machine crate automates checking properties of a system under test (SUT) against an abstract reference state machine definition. Heavily inspired by Erlang's eqc_statem (see paper "Finding Race Conditions in Erlang with QuickCheck and PULSE").

State machine testing is very effective for modeling effectful code that performs state changes too hard to test thoroughly with manual approaches or too complex to verify formally.

### Model Checking

Model checking involves systems modeled by finite state machines, properties written in propositional temporal logic, and verification procedure as exhaustive search of the state space.

#### Process

A model checker takes an automaton of a system and a temporal logic property to be satisfied, then visits all reachable states of the automaton and verifies for each state that the temporal logic property is satisfied.

In finite-state systems, model checking is theoretically always applicable since exhaustive traversal through reachable state space can effectively provide enough information to solve verification problems. Verification is performed as exhaustive state space search guaranteed to terminate if the model is finite.

#### Temporal Logic

Main temporal logics used are Linear Temporal Logic (LTL), Computation Tree Logic (CTL), and CTL*. Properties can express safety, liveness, or fairness properties.

A test generation strategy can verify that systems satisfy LTL safety conditions exhaustively - any system violating at least one formula will fail at least one test case.

### TLA+ and PlusCal

TLA+ is a formal specification language developed by Leslie Lamport for designing, modeling, documenting, and verifying programs, especially concurrent and distributed systems. Such specifications essentially describe state machines.

#### PlusCal

PlusCal was developed to lower TLA+ threshold. Similar to a programming language, it describes program logic and translates into TLA+ by borrowing provided tools. Introduced in 2009.

#### Verification Capabilities

TLA+ specifications are amenable to finite model checking. The model checker finds all possible system behaviors up to some number of execution steps and examines them for violations of desired invariance properties like safety and liveness.

The TLC model checker builds a finite state model of TLA+ specifications for checking invariance properties. TLC generates initial states satisfying the spec, then performs breadth-first search over all defined state transitions.

#### Industrial Applications

Amazon Web Services has used TLA+ since 2011. Model checking uncovered bugs in DynamoDB, S3, EBS, and an internal distributed lock manager; some bugs required state traces of 35 steps. TLA+ usage has surged since 2015, especially in cloud industry, mainly applied during early system design and debugging. It effectively uncovers deep bugs, enhances system design, and improves overall understanding.

### Assertion-Based Verification

Adding assertions to FSM RTL code describing expected behavior is a powerful technique that can detect design errors and improve test quality.

### Code Coverage Analysis

Measures the extent to which FSM design is exercised by test vectors at different levels such as statement, branch, and path coverage, helping identify untested parts.

## Architectural Options for Device Reconnect Resume

### Option 1: Checkpoint/Restore Pattern

**Concept**: Save current operation state when device disconnected, restore when reconnected.

**Implementation**:
- Scanning state: Save current window index, frequency, progress within window
- Listening state: Save station frequency, audio session state
- On device removal event: Serialize state to checkpoint
- On device addition event: Deserialize checkpoint and resume operation

**Advantages**:
- Clear separation of concerns
- State can be persisted to disk for crash recovery
- Easy to understand and debug

**Disadvantages**:
- Requires serialization/deserialization logic
- Need to determine what state is "checkpointable"
- May have race conditions during checkpoint creation

### Option 2: Hierarchical State Machine with State Preservation

**Concept**: Use nested states where parent state preserves context during device disconnect.

**Implementation**:
```
OperationalState
├── DeviceAvailable
│   ├── Idle
│   ├── Scanning(window_index, progress)
│   └── Listening(frequency, audio_state)
└── DeviceUnavailable
    ├── ScanningPaused(window_index, progress)
    └── ListeningPaused(frequency, audio_state)
```

**State Transitions**:
- DeviceAvailable::Scanning → DeviceUnavailable::ScanningPaused (preserve window_index)
- DeviceUnavailable::ScanningPaused → DeviceAvailable::Scanning (resume)

**Advantages**:
- State and transitions are explicit
- Clear visual representation
- Compiler can enforce valid transitions (with typestate)

**Disadvantages**:
- State explosion if many operation types
- Need to duplicate state data in paused variants

### Option 3: Event Sourcing with Replay

**Concept**: Record all state transitions as events, replay to recover state.

**Implementation**:
- Events: ScanStarted, WindowProcessed, StationTuned, DeviceDisconnected, DeviceConnected
- Event log persisted during operation
- On reconnect: Replay events to determine current operation
- Resume from last incomplete operation

**Advantages**:
- Complete audit trail
- Can replay for debugging
- Natural fit for event-driven architecture

**Disadvantages**:
- Performance overhead of event logging
- Complexity in replay logic
- Need snapshot strategy for long-running operations

### Option 4: Saga Pattern with Compensation

**Concept**: Each operation is a saga step with compensation logic.

**Implementation**:
- Scan window = saga step with compensating "abort scan window" action
- Listen to station = saga step with compensating "stop listening" action
- Device disconnect triggers compensation chain
- Device reconnect triggers retry of interrupted step

**Advantages**:
- Well-established distributed systems pattern
- Clear error handling semantics
- Compensation actions are explicit

**Disadvantages**:
- Overkill for single-node application
- Complexity of saga coordination
- Not all operations have meaningful compensation

### Option 5: Supervisor Pattern (Actor Model)

**Concept**: Supervisor actor manages device lifecycle, worker actors handle operations.

**Implementation**:
- DeviceSupervisor actor owns device connection
- ScanWorker and ListenWorker actors are children
- On device disconnect: Suspend workers but don't destroy
- On device reconnect: Resume workers

**Advantages**:
- Clear separation of concerns
- Supervisor handles all recovery logic
- Workers don't need to know about device lifecycle

**Disadvantages**:
- Requires actor framework
- More complex threading model
- Overkill if not using actors elsewhere

### Option 6: Resumable State with Explicit Pause

**Concept**: Add pause/resume capability to existing states.

**Implementation**:
- Add `paused: bool` flag to Scanning and Listening states
- Add `pause_context: Option<PauseContext>` to save state
- Device disconnect sets paused=true, saves context
- Device reconnect checks paused, restores context, sets paused=false

**Advantages**:
- Minimal changes to existing state machine
- No state explosion
- Simple to implement

**Disadvantages**:
- Pause flag can be in inconsistent state
- Less type-safe than explicit states
- Context structure needs careful design

## Testing Strategies for Each Option

### Checkpoint/Restore Testing

**Unit Tests**:
- Verify checkpoint serialization/deserialization
- Test checkpoint creation at various operation points
- Verify restore from corrupted checkpoint

**Property-Based Tests**:
- Generate random operation sequences
- Create checkpoint at random points
- Verify restored state matches pre-checkpoint state

**Integration Tests**:
- Simulate device disconnect during scanning
- Verify checkpoint created
- Simulate device reconnect
- Verify operation resumes from correct point

### HSM Testing

**State Transition Tests**:
- Enumerate all valid state transitions
- Verify invalid transitions are rejected
- Test transitions with all possible payloads

**Property Tests with proptest-state-machine**:
- Define reference model of HSM
- Generate random event sequences
- Verify SUT matches reference model
- Automatically finds minimal failing sequences

**Model Checking with TLA+**:
- Formally specify HSM in TLA+/PlusCal
- Define invariants (e.g., "never lose operation context")
- Run TLC model checker to verify all reachable states

### Event Sourcing Testing

**Event Replay Tests**:
- Record event sequence during operation
- Disconnect device mid-operation
- Replay events to verify state reconstruction
- Compare reconstructed state to actual state

**Property Tests**:
- Generate random event sequences
- Verify replay produces same final state
- Test with event sequence truncation (simulating incomplete log)

**Snapshot Consistency Tests**:
- Create snapshots at random points
- Verify replay from snapshot matches replay from beginning
- Test snapshot + events matches full event replay

### Integration Testing Strategy (All Options)

**Fault Injection**:
- Use mock device that can disconnect on command
- Inject disconnects at random operation points
- Verify operation resumes correctly on reconnect

**Concurrency Testing with Loom**:
- Test race conditions between device events and operation events
- Verify no data races in state updates
- Test all possible thread interleavings

**Long-Running Chaos Tests**:
- Run scanner for extended period
- Randomly disconnect/reconnect device
- Verify no memory leaks or state corruption
- Monitor for gradual degradation

## References

- Hierarchical State Machines: https://www.state-machine.com/doc/Hierarchical_State_Machines.pdf
- Typestate Pattern in Rust: https://cliffle.com/blog/rust-typestate/
- Property-Based Testing: https://www.lpalmieri.com/posts/an-introduction-to-property-based-testing-in-rust/
- State Machine Testing with Proptest: https://tzemanovic.gitlab.io/posts/state-machine-testing-with-proptest/
- TLA+ Tutorial: https://lamport.azurewebsites.net/pubs/spec-and-verifying.pdf
- Event Sourcing Snapshots: https://codeopinion.com/snapshots-in-event-sourcing-for-rehydrating-aggregates/
- Saga Pattern: https://microservices.io/patterns/data/saga.html
- Fault-Tolerant State Machines: https://www.cs.cornell.edu/fbs/publications/SMSurvey.pdf
