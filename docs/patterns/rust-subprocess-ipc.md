# Rust Subprocess IPC Design Patterns

This document catalogs design patterns specific to Rust subprocess inter-process communication. These patterns leverage Rust's ownership system, type safety, and ecosystem to build robust IPC systems.

## RAII Guards for Resource Management

RAII (Resource Acquisition Is Initialization) guards ensure resources are automatically cleaned up when they go out of scope. In Rust, this pattern is enforced through the ownership system and Drop trait.

The pattern uses two types: a resource holder that doesn't allow direct access, and an RAII guard that mediates access until dropped. Mutex guards are the canonical example, ensuring locks release automatically. For subprocess handles, this means file descriptors, sockets, and process handles are cleaned up deterministically.

Variables are dropped in reverse order of creation. The Drop trait requires implementing one method that takes a mutable reference to self. Best practices avoid panicking in drop implementations since drop may be called during unwinding.

**When to use**: Managing subprocess handles, file descriptors, Unix sockets, or any resource requiring deterministic cleanup. Essential when resource leaks would cause system-level issues.

**When NOT to use**: Simple value types without cleanup requirements. Situations where manual control over cleanup timing is required. Avoid when drop implementations would be complex enough to hide important cleanup logic.

## Typestate Pattern for Subprocess Lifecycle

The typestate pattern uses separate types to represent each state in a subprocess lifecycle, with Rust's ownership system preventing invalid state transitions at compile time.

Each subprocess state (Created, Running, Stopped, Terminated) becomes a distinct type. Methods consuming the old state return a new state, making invalid transitions impossible. Properties like "cannot send commands to a terminated subprocess" become compile-time guarantees.

The pattern enforces that subprocesses must go through states in specific orders. For example, a subprocess must be spawned before being configured, configured before starting streaming, and stopped before cleanup.

**When to use**: Complex subprocess lifecycles with strict state requirements. APIs where invalid state transitions would cause subtle bugs. Systems requiring compile-time guarantees about subprocess state validity.

**When NOT to use**: Simple fire-and-forget subprocess launches. Dynamic scenarios where state isn't known at compile time. Prototyping where type-level complexity slows development. Situations requiring runtime state inspection.

## Zero-Copy Shared Memory

Zero-copy IPC uses memory-mapped files (mmap) to eliminate data copying between processes. Ring buffers with lock-free synchronization primitives provide wait-free access patterns.

Libraries like iceoryx2 implement publish/subscribe and request/response patterns over shared memory. The mmap-sync pattern uses rkyv for zero-copy deserialization, directly referencing bytes in serialized form. The approach draws from Linux kernel's RCU and Left-Right concurrency control techniques.

Shared memory achieves ping-pong in under 200 nanoseconds (~1000 processor cycles), dramatically reducing context switches and cache thrashing for consistently lower latencies and higher throughput.

**When to use**: High-frequency communication where latency matters (microsecond scale). Large data transfers where copying would be prohibitive. Systems requiring maximum throughput between processes on the same machine.

**When NOT to use**: Cross-network communication. Simple request-response where overhead is negligible. Situations requiring strong security boundaries between processes. Systems needing dynamic process discovery rather than fixed shared memory regions.

## Unix Domain Socket Streams

Unix domain sockets provide bidirectional IPC with stream semantics. Rust's std::os::unix::net module offers UnixStream, UnixListener, and UnixDatagram types.

UnixStream::pair() creates connected socket pairs suitable for subprocess communication. UnixListener::bind() creates listening sockets at filesystem paths. Sockets provide message boundaries (with SOCK_SEQPACKET) or stream semantics (SOCK_STREAM).

File descriptor passing enables sharing handles between processes. Unix sockets offer better performance than TCP for local communication while providing familiar socket APIs.

**When to use**: Reliable bidirectional communication between local processes. Situations needing file descriptor passing. Systems requiring message boundaries or datagram semantics. APIs matching network socket patterns for code reuse.

**When NOT to use**: Cross-machine communication. Situations where shared memory would provide better performance. Windows-only applications (though named pipes provide similar functionality). Simple parent-child stdio communication.

## Async Subprocess Management with Tokio

Tokio provides async subprocess management through tokio::process, offering Command::spawn() returning a future. This enables non-blocking subprocess I/O integrated with async/await.

The tokio::process::ChildStdin, ChildStdout, and ChildStderr types implement AsyncRead/AsyncWrite. Select combinators enable racing between subprocess operations and timeouts or cancellation. Tokio's runtime handles I/O readiness notifications.

**When to use**: Applications already using Tokio or async Rust. Managing multiple subprocesses concurrently. Situations requiring timeouts, cancellation, or orchestration of subprocess I/O with other async operations.

**When NOT to use**: Simple synchronous subprocess launches. Applications without async runtime overhead justification. Embedded systems with resource constraints. Blocking operations better suited to thread pools.

## Serde-based Message Passing

Serde provides compile-time serialization code generation supporting JSON, Bincode, MessagePack, CBOR, and more formats. Types derive Serialize and Deserialize traits for automatic implementation.

Different formats offer different trade-offs. JSON provides human readability and wide compatibility. Bincode offers compact binary encoding with minimal overhead. MessagePack balances size and compatibility. CBOR provides schema evolution support.

Serde integrates with std::io::Write for serializing to files, sockets, or pipes. Zero-copy deserialization is possible with specialized formats like rkyv.

**When to use**: Type-safe message passing between Rust processes. Situations needing human-readable messages (JSON) or compact binary (Bincode). Systems requiring schema evolution. Cross-language communication with standard formats.

**When NOT to use**: Maximum performance scenarios where even Bincode overhead matters. Simple fixed-format messages where manual serialization is trivial. Legacy protocols requiring specific binary layouts. Real-time systems where serialization latency is prohibitive.

## Channel-based Worker Pools

The std::sync::mpsc module provides multi-producer, single-consumer channels. Worker pools wrap receivers in Arc<Mutex<Receiver>> for multiple consumers competing for jobs.

An alternative pattern gives each worker its own channel, maintaining a queue of sender clones. Work distributes by taking a sender from the queue front, sending work, and returning it to the queue back. This avoids lock contention on the receiver.

The pattern decouples producers from consumers, enables dynamic worker scaling, and provides backpressure through bounded channels.

**When to use**: Distributing work across multiple subprocess workers. Situations requiring load balancing. Systems needing backpressure control. Applications with bursty workloads benefiting from worker pools.

**When NOT to use**: Single subprocess scenarios. Real-time systems where channel overhead matters. Situations requiring ordered processing rather than concurrent execution. When direct subprocess communication is simpler.

## Communicate Pattern for Deadlock-Free I/O

The communicate pattern handles simultaneous subprocess reading and writing without deadlock, inspired by Python's subprocess.communicate(). It reads and writes concurrently, avoiding deadlocks when subprocesses write output before consuming input.

The subprocess crate provides communicate methods capturing subprocess output/error to memory while feeding stdin. This prevents the common deadlock where both parent and child wait for EOF.

For protocols without EOF, use length-prefixed messages: send 4-byte length followed by payload, enabling readers to consume exactly the expected bytes without waiting for EOF.

**When to use**: Bidirectional subprocess communication. Capturing subprocess output while providing input. Situations where subprocess may produce output before consuming all input. Avoiding manual buffer management complexity.

**When NOT to use**: One-way communication where deadlock isn't possible. Streaming scenarios where complete output capture isn't needed. Real-time applications requiring incremental processing. Complex protocols better served by custom logic.

## Graceful Shutdown Coordination

Graceful shutdown uses signal handling (SIGINT, SIGTERM) to notify processes of shutdown requests. Tokio provides tokio::signal::ctrl_c and Unix signal handling.

The pattern involves three parts: detecting shutdown requests, notifying all subsystems, and waiting for cleanup. Cancellation tokens or broadcast channels propagate shutdown signals. Tasks receive shutdown notices and complete in-flight work before terminating.

The Drop trait on thread pool types joins worker threads during cleanup. Subprocesses receive termination signals and have timeouts before forced kills.

**When to use**: Long-running services requiring clean shutdown. Systems with in-flight work needing completion. Applications managing resources requiring cleanup (databases, files, sockets). Multi-process systems needing coordinated shutdown.

**When NOT to use**: Short-lived command-line tools. Situations where immediate termination is acceptable. Systems without cleanup requirements. Embedded systems where signals aren't available.

## Arc<Mutex<T>> for Shared Subprocess State

Arc (Atomic Reference Counting) enables sharing ownership across threads. Mutex provides mutual exclusion for interior mutability. Combined as Arc<Mutex<T>>, they enable safe shared mutable state.

Arc clones increase the reference count, allowing multiple thread ownership. Mutex::lock() returns a guard that provides access while holding the lock. The guard automatically releases the lock when dropped.

For async contexts, tokio::sync::Mutex provides an await-friendly mutex avoiding blocking the async runtime.

**When to use**: Sharing subprocess state (status, configuration) across multiple threads. Coordinating access to subprocess handles. Managing pools of subprocess resources. Thread-safe caching of subprocess results.

**When NOT to use**: Single-threaded subprocess management. Situations where message passing would be clearer. Performance-critical paths where lock contention matters. When subprocesses could use isolated per-thread state instead.

## Protocol Layering for IPC Messages

Protocol layering builds structured communication on top of raw I/O primitives. Common patterns include length-prefixed messages (4-byte length + payload) and framed protocols (magic bytes + type + length + payload).

Higher-level protocols use encoders/decoders implementing Tokio's Encoder/Decoder traits. Parsers handle partial reads and buffer management. Message types use enums with Serde serialization.

**When to use**: Custom IPC protocols beyond simple request-response. Multiplexing multiple logical channels over one connection. Implementing compatibility with existing protocols. Systems requiring message framing and validation.

**When NOT to use**: Standard protocols where libraries exist (HTTP, gRPC). Simple line-based protocols. Performance-critical paths where protocol overhead matters. Prototyping where protocol may change significantly.

## Capability-based IPC with File Descriptor Passing

Unix sockets support file descriptor passing via SCM_RIGHTS, enabling capability-based security. One process grants another access to resources by passing file descriptors rather than relying on filesystem permissions or authentication.

The pattern separates privilege (one process opens privileged resources) from use (unprivileged processes receive descriptors). Subprocesses receive only the capabilities they need. This follows the principle of least privilege.

**When to use**: Security-sensitive subprocess architectures. Systems requiring privilege separation. Sandboxed subprocesses needing limited resource access. Platforms supporting file descriptor passing (Unix-like systems).

**When NOT to use**: Windows (lacks native FD passing). Simple subprocess launches without privilege requirements. Systems where filesystem permissions suffice. Cross-network communication.

## Supervision Trees for Subprocess Management

Supervision trees organize subprocesses hierarchically with supervisors monitoring workers. Supervisors detect failures and restart failed subprocesses according to restart strategies (one-for-one, all-for-one, rest-for-one).

The pattern provides fault isolation where failures don't cascade. Restart budgets limit retry storms. Supervision strategies match business requirements (transient vs permanent failures).

**When to use**: Long-running subprocess architectures. Systems requiring automatic failure recovery. Applications with multiple cooperating subprocesses. Distributed systems needing resilience.

**When NOT to use**: Short-lived subprocesses. Single subprocess scenarios. Systems where failures should terminate the application. Situations requiring manual failure handling.

## Serialization Format Selection

Different formats offer different trade-offs for subprocess IPC:

**Bincode**: Minimal overhead, compact size, no schema evolution. Best for trusted Rust-to-Rust communication where all types compile into both processes.

**JSON**: Human-readable, widely compatible, larger size, slower parsing. Good for debugging, tools, cross-language communication where performance isn't critical.

**Protocol Buffers (prost)**: Schema evolution, cross-language support, copy-heavy deserialization. Suitable for systems needing backward compatibility and polyglot support.

**Cap'n Proto**: Zero-copy deserialization, excellent performance, unergonomic API. Best for performance-critical IPC willing to trade ergonomics for speed.

**rkyv**: Fastest serialization/deserialization, zero-copy via validation, requires careful memory layout. Ideal for maximum performance with trusted data.

**When to use specific formats**: See individual format characteristics above. The choice depends on trust boundaries, performance requirements, debugging needs, and cross-language requirements.

**When NOT to use**: Avoid overengineering simple protocols. Don't pick formats mismatched to requirements (JSON for performance-critical paths, Bincode for cross-language).

## Pattern Interactions

Real Rust subprocess systems layer multiple patterns:

A subprocess pool uses RAII guards wrapping Unix domain sockets. Workers receive jobs via channel-based worker pools. Messages use Serde with Bincode serialization. Graceful shutdown uses broadcast channels coordinating worker cleanup.

Typestate patterns ensure subprocesses transition through lifecycle states correctly. Arc<Mutex<>> shares pool state across threads. The communicate pattern prevents deadlocks during subprocess I/O.

For maximum performance, zero-copy shared memory replaces sockets. Protocol layering adds message framing. Supervision trees provide automatic restart. File descriptor passing implements capability-based security.

Async Rust with Tokio coordinates multiple subprocesses concurrently. Signal handling initiates graceful shutdown. Drop implementations ensure resource cleanup even during unwinding.

## References

### RAII and Resource Management
- Rust Design Patterns: RAII Guards - https://rust-unofficial.github.io/patterns/patterns/behavioural/RAII.html
- Effective Rust: Item 11 - Implement the Drop trait for RAII patterns - https://effective-rust.com/raii.html
- The Rustonomicon: Ownership Based Resource Management - https://doc.rust-lang.org/stable/nomicon/obrm.html

### Typestate Pattern
- The Typestate Pattern in Rust - Cliffle - https://cliffle.com/blog/rust-typestate/
- Build with Naz: Rust typestate pattern - https://developerlife.com/2024/05/28/typestate-pattern-rust/
- Typestate - Type-Driven API Design in Rust - https://willcrichton.net/rust-api-type-patterns/typestate.html

### IPC Libraries and Implementations
- servo/ipc-channel - Multiprocess drop-in replacement for Rust channels - https://github.com/servo/ipc-channel
- subprocess crate - External process execution and interaction - https://docs.rs/subprocess/latest/subprocess/
- interprocess crate - Cross-platform IPC toolkit - https://github.com/kotauskas/interprocess
- rust-subprocess crate documentation - https://docs.rs/subprocess/latest/subprocess/

### Zero-Copy and Shared Memory
- Eclipse iceoryx2 - True zero-copy IPC - https://github.com/eclipse-iceoryx/iceoryx2
- Cloudflare mmap-sync - https://github.com/cloudflare/mmap-sync
- I Tried Zero-Copy IPC in Rust - https://levelup.gitconnected.com/i-tried-zero-copy-ipc-in-rust-and-blew-my-mind-heres-how-you-can-too-953fa0817d10
- rkyv is faster than alternatives - https://david.kolo.ski/blog/rkyv-is-faster-than/

### Unix Domain Sockets
- Rust RFC 1479: Unix Socket Support - https://rust-lang.github.io/rfcs/1479-unix-socket.html
- std::os::unix::net::UnixStream documentation - https://doc.rust-lang.org/std/os/unix/net/struct.UnixStream.html
- Example: IPC with Unix domain sockets - https://gist.github.com/tesaguri/b27d0d35d1a45465ddc9cb32a3ebe9ae

### Async Subprocess Management
- tokio::process documentation - https://docs.rs/tokio/latest/tokio/process/index.html
- Tokio Tutorial - https://tokio.rs/

### Serialization
- Serde - https://serde.rs/
- Rust Serialization Benchmarks - https://github.com/djkoloski/rust_serialization_benchmark
- Comparing serialization formats - https://users.rust-lang.org/t/overwhelmed-by-the-vast-variety-of-serialization-formats-which-to-use-when/88440

### Concurrency and Coordination
- Rust Book: Shared-State Concurrency - https://doc.rust-lang.org/book/ch16-03-shared-state.html
- Tokio: Shared State - https://tokio.rs/tokio/tutorial/shared-state
- std::sync::mpsc documentation - https://doc.rust-lang.org/std/sync/mpsc/index.html

### Graceful Shutdown
- Tokio: Graceful Shutdown - https://tokio.rs/tokio/topics/shutdown
- Rust Book: Graceful Shutdown and Cleanup - https://doc.rust-lang.org/book/ch20-03-graceful-shutdown-and-cleanup.html
- Command Line Apps: Signal Handling - https://rust-cli.github.io/book/in-depth/signals.html

### Deadlock Prevention
- subprocess crate communicate methods - https://docs.rs/subprocess/latest/subprocess/
- rust-subprocess-communicate - https://github.com/dropbox/rust-subprocess-communicate

### Performance Comparisons
- IPC in Rust - a Ping Pong Comparison - https://3tilley.github.io/posts/simple-ipc-ping-pong/
- Yet Another IPC in Rust Experiment - https://vadosware.io/post/yet-another-ipc-in-rust-experiment
- Cap'n Proto vs gRPC in Rust - https://medium.com/@learnwithshobhit/comparing-capn-proto-and-grpc-in-rust-a-performance-and-feature-analysis-61d2da815d18
