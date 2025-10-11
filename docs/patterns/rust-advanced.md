# Advanced Rust Patterns

This document covers advanced Rust patterns for specialized use cases, complex type system features, and less commonly used techniques. For core patterns used in everyday Rust code, see `idiomatic-rust.md`.

**Organization:**
- **idiomatic-rust.md**: Core patterns every Rust developer should know
- **rust-advanced.md** (this file): Advanced type system, async, FFI, compile-time computation, and specialized patterns

## Advanced Type System Patterns

### Typestate Pattern

Encodes object state in the type system, making invalid state transitions impossible at compile time.

**How it works:**
- Each state is a distinct type
- State transitions consume the object and return the new state
- Uses generic type parameters or phantom data to track state
- Compiler prevents calling methods on wrong state

```rust
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> { /* ... */ }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> { /* ... */ }
    fn open(&mut self) { /* only available when unlocked */ }
}
```

**When to use:**
- State machines where invalid transitions should be impossible
- Builder patterns with required initialization steps
- Resource management with distinct lifecycle phases
- API designs preventing misuse (file must be open before reading)
- Protocol implementations with strict ordering requirements

**When NOT to use:**
- Simple state machines where runtime validation is clearer
- When state transitions are highly dynamic and data-driven
- Types that need to be stored in collections with mixed states
- When type complexity outweighs safety benefits
- Deeply nested state hierarchies (combinatorial explosion)

**References:** Cliffle's blog, Will Crichton's type-level programming, developerlife.com

---

### PhantomData and Zero-Sized Types

Marker types that carry compile-time information without runtime cost.

**PhantomData<T>**: Tells the compiler the struct acts like it owns a T
- Zero-sized, no runtime overhead
- Affects variance and drop check
- Used for unused type parameters
- Critical for unsafe code with ownership-like relationships

```rust
struct Slice<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<&'a T>,  // Indicates we "borrow" T for 'a
}
```

**Zero-Sized Types (ZSTs)**: Types with no data
- Unit struct: `struct Marker;`
- Empty tuple: `()`
- Arrays of size 0: `[T; 0]`
- Take no memory but can carry type information

**When to use PhantomData:**
- Unused lifetime or type parameters
- Unsafe code with ownership-like relationships
- Variance annotations
- Type-level programming and marker types

**When to use ZSTs:**
- Marker traits and type-level flags
- Stateless trait implementations
- Type-level computations
- Eliminating allocation for certain state machines

**When NOT to use:**
- When actual data should be stored
- Cases where runtime size matters (FFI, serialization)
- When it adds confusion without benefit

**References:** The Rust Programming Language book, Rustonomicon, std::marker

---

### Generic Type Parameters

Uses generic type parameters to create flexible, reusable code that works across different types while maintaining type safety.

**How it works:**
- Define functions, structs, enums, or traits with type parameters
- Monomorphization generates specialized code for each type
- Trait bounds constrain acceptable types
- Associated types allow specifying related types

```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T { /* ... */ }

struct Container<T> {
    value: T,
}

impl<T: Display> Container<T> {
    fn print(&self) {
        println!("{}", self.value);
    }
}
```

**When to use:**
- Building reusable data structures
- Writing generic algorithms
- Abstracting over types with shared behavior
- Library code that works with many types

**When NOT to use:**
- When concrete types would be clearer
- Excessive generics leading to poor error messages
- When trait objects would be more appropriate

---

### Const Generics

Permits values of integral types to be used as parameters to generic types, traits, and functions.

**How it works:**
- Generic parameters can be const values (integers, bools, chars)
- Enables compile-time array sizes and other numeric constraints
- Type-safe abstractions over sizes and counts

```rust
struct Array<T, const N: usize> {
    data: [T; N],
}

fn combine<const M: usize, const N: usize>(
    a: [i32; M],
    b: [i32; N]
) -> [i32; M + N] {
    // ...
}
```

**When to use:**
- Fixed-size arrays with type-safe lengths
- Compile-time computations over sizes
- Generic code parameterized by numeric values
- Eliminating runtime bounds checking

**When NOT to use:**
- Runtime-determined sizes (use Vec)
- Complex const expressions (limited support)
- When dynamic sizing is needed

---

### Trait Objects (dyn Trait)

Represents a value of any type that implements a specific trait, using dynamic dispatch.

**How it works:**
- Typically used via references (`&dyn Trait`) or smart pointers (`Box<dyn Trait>`)
- Trait objects are dynamically sized types (DSTs)
- Store two pointers: one to data and one to a vtable
- Methods dispatched via vtable lookup at runtime

```rust
trait Draw {
    fn draw(&self);
}

fn draw_all(shapes: &[Box<dyn Draw>]) {
    for shape in shapes {
        shape.draw();  // Dynamic dispatch
    }
}
```

**When to use:**
- Heterogeneous collections of different types
- Plugin systems and dynamic loading
- When types are determined at runtime
- Reducing binary size (less monomorphization)

**When NOT to use:**
- Performance-critical paths (use static dispatch)
- When all types are known at compile time
- Traits with generic methods (not object-safe)
- When trait has associated types with multiple implementations

**Comparison with Static Dispatch:**
- **Static (generics)**: Faster, larger binary, compile-time polymorphism
- **Dynamic (trait objects)**: Slower, smaller binary, runtime polymorphism

**References:** The Rust Programming Language book, Rust docs

---

### Extension Traits

Add methods to foreign types by defining and implementing your own trait.

**How it works:**
- Define a trait with the methods you want to add
- Implement the trait for the foreign type
- Import the trait to make methods available
- Works around the orphan rule

```rust
trait StrExt {
    fn word_count(&self) -> usize;
}

impl StrExt for str {
    fn word_count(&self) -> usize {
        self.split_whitespace().count()
    }
}
```

**When to use:**
- Adding domain-specific methods to standard types
- Creating convenience methods for foreign types
- Building DSLs (domain-specific languages)
- Providing compatibility layers
- Extending types from dependencies

**When NOT to use:**
- When the method should be on the type itself (if you own it)
- Methods that conflict with existing or likely future methods
- Adding methods that are too generic or likely to clash
- When free functions would be clearer

**References:** Rust API Guidelines, Mastering Backend

---

### Sealed Traits

Traits that cannot be implemented outside the defining crate.

**How it works:**
- Create a private module with a public trait
- Make the main trait require this private trait as a supertrait
- External crates cannot name the private trait, so cannot implement the main trait

```rust
mod private {
    pub trait Sealed {}
}

pub trait PublicTrait: private::Sealed {
    fn method(&self);
}

// Only internal impls can implement both traits
impl private::Sealed for MyType {}
impl PublicTrait for MyType {
    fn method(&self) { /* ... */ }
}
```

**When to use:**
- Traits where external implementations would break invariants
- Future-proofing APIs (can add methods without breaking changes)
- Traits closely tied to specific implementations
- Enumerating all valid implementations
- API evolution without semver breakage

**When NOT to use:**
- Traits intended for external implementation
- Simple marker traits without invariants
- When extensibility is a core feature
- Traits that define clear contracts for external use

**References:** Rust API Guidelines (Future Proofing), Rust Internals, Predr.ag blog

---

## Advanced Ownership and Lifetime Patterns

### Lifetime Annotations

Explicit markers that tell the compiler how long references should remain valid.

**How it works:**
- Generic lifetime parameters denoted with `'a`, `'b`, etc.
- Express relationships between input and output lifetimes
- Prevent dangling references at compile time
- Required in function signatures, structs, and enums with references

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

struct ImportantExcerpt<'a> {
    part: &'a str,
}
```

**When to use:**
- Functions returning references
- Structs/enums storing references
- When compiler cannot infer lifetimes
- Complex borrowing relationships

**When NOT to use:**
- When you can return owned types
- When lifetime elision applies
- Over-constraining lifetimes unnecessarily

---

### Lifetime Elision

Rules that allow the compiler to infer lifetimes in common cases, reducing boilerplate.

**Elision Rules:**
1. Each elided input lifetime gets a distinct parameter
2. If exactly one input lifetime, it's assigned to all output lifetimes
3. If multiple input lifetimes and one is `&self` or `&mut self`, self's lifetime assigned to outputs

**When elision works:**
```rust
fn first_word(s: &str) -> &str { /* ... */ }
// Equivalent to:
fn first_word<'a>(s: &'a str) -> &'a str { /* ... */ }
```

---

### Pin and Unpin

Guarantees that self-referential types won't be moved in memory.

**How it works:**
- `Pin<P>` wraps a pointer and prevents moving the pointed-to value
- `Unpin` is an auto trait indicating a type is safe to move when pinned
- Required by Future::poll which takes `Pin<&mut Self>`
- Most types implement Unpin automatically

**When to use:**
- Implementing Future types with self-references
- Creating async primitives
- Unsafe code with pointer stability requirements
- Types that cannot be moved after initialization

**When NOT to use:**
- Regular synchronous code
- Types that don't have self-references
- When Box<T> or &mut T suffices
- Beginners should use `Box::pin` and not worry about details

**References:** std::pin docs, Cloudflare blog, Async Rust book

---

### Advanced Smart Pointers

**RefCell<T>**: Interior mutability with runtime borrow checking
- Panics if borrow rules violated
- Single-threaded only
- Often used with Rc<RefCell<T>>

**Mutex<T>** / **RwLock<T>**: Thread-safe interior mutability
- Mutex: Exclusive access
- RwLock: Multiple readers or one writer
- Blocking synchronization primitives
- Often used with Arc<Mutex<T>>

**When to use Mutex:**
- Shared mutable state across threads
- Exclusive access to resources
- When reads and writes are equally common

**When to use RwLock:**
- Many concurrent readers, few writers
- Read-heavy workloads
- When read performance matters

**When NOT to use:**
- Single-threaded code (use RefCell)
- Lock-free alternatives exist and are suitable
- When contention would be high

---

## Behavioral Patterns

### Command Pattern

Encapsulates actions as objects, allowing parameterization of clients with different requests, queuing of requests, and logging of operations.

**How it works:**
- Define a trait for commands
- Implement trait for concrete command types
- Store and execute commands dynamically

**When to use:**
- Undo/redo systems
- Task queues
- Macro recording
- Transactional systems

**When NOT to use:**
- Simple function calls suffice
- When closures provide the same benefit
- Over-engineering simple operations

---

### Interpreter Pattern

Defines a representation for a grammar and an interpreter that uses the representation to interpret sentences in the language.

**When to use:**
- Implementing DSLs
- Expression evaluators
- Configuration languages
- Query languages

**When NOT to use:**
- Complex languages (use parser generators)
- Performance-critical parsing
- When existing parsers/interpreters suffice

---

### Strategy Pattern

Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Often implemented using trait objects or generics.

**How it works:**
- Define a trait for the strategy
- Implement different algorithms as types implementing the trait
- Accept strategy via trait object or generic parameter

**When to use:**
- Multiple algorithms for the same task
- Runtime algorithm selection
- Dependency injection
- Testing with mock implementations

**When NOT to use:**
- Only one algorithm
- When function pointers suffice
- Compile-time selection is adequate (use generics)

---

### Visitor Pattern

Separates algorithms from the objects on which they operate, allowing new operations to be added without modifying the objects.

**How it works:**
- Define Visitor trait with visit methods
- Data types accept visitors
- Visitor implements operations
- Double dispatch enables type-safe operations

**When to use:**
- Operating on complex object structures
- Adding new operations without changing types
- When operations are more volatile than data structure
- Abstract syntax trees and compilers

**When NOT to use:**
- Simple data transformations (use iterators/map)
- When data structure changes frequently
- Single operation on data

**References:** Rust Design Patterns book, Refactoring Guru

---

## Async Patterns

### Async/Await and Futures

Futures represent values that will be available later. The `async`/`await` syntax provides ergonomic asynchronous programming.

**How it works:**
- `async fn` returns a Future
- `.await` yields control to executor
- Compiler transforms async functions into state machines
- Runtime schedules futures across threads

**When to use:**
- I/O-bound workloads
- High-concurrency applications
- Microservices and web servers
- Applications with many concurrent operations

**When NOT to use:**
- CPU-bound workloads (use thread pools)
- Simple scripts where async overhead isn't justified
- FFI or blocking operations (use spawn_blocking)
- When synchronous code is simpler and adequate

---

### Stream Pattern

Asynchronous equivalent of iterators, yielding values over time rather than all at once.

**How it works:**
- Stream trait similar to Iterator but async
- `poll_next()` instead of `next()`
- Combinators like map, filter, fold for async
- Used for async sequences (websocket messages, file chunks)

**When to use:**
- Processing async sequences
- Network protocol implementations
- Async iteration over collections
- Real-time data feeds

**When NOT to use:**
- Synchronous iteration (use Iterator)
- Single values (use Future)
- When buffering entire sequence is acceptable

---

### Async Patterns with Tokio

Common patterns for Tokio-based applications.

**Spawning Tasks:**
```rust
tokio::spawn(async {
    // Background task
});
```

**Select over multiple futures:**
```rust
tokio::select! {
    result1 = future1 => { /* ... */ },
    result2 = future2 => { /* ... */ },
}
```

**Timeouts:**
```rust
tokio::time::timeout(Duration::from_secs(5), operation).await?;
```

**When to use:**
- Concurrent async operations
- Timeouts and cancellation
- Racing multiple operations
- Complex async control flow

**References:** Tokio docs, Async Rust book

---

## Compile-Time Computation

### Const Functions (const fn)

Functions that can be evaluated at compile time, allowing their results to be used in constant contexts.

**How it works:**
- `const fn` can be called in const contexts
- Enables compile-time computation
- Results can be used for array sizes, const declarations
- Restrictions on what operations are allowed

```rust
const fn fibonacci(n: usize) -> usize {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

const FIB_10: usize = fibonacci(10);
```

**When to use:**
- Compile-time computations
- Deriving constants from other constants
- Zero-runtime-cost computed values
- Generic const parameters

**When NOT to use:**
- Complex computations increasing compile time
- When runtime computation is acceptable
- Operations not allowed in const fn

---

### Const Evaluation

The process of computing expression results during compilation.

**Benefits:**
- Zero runtime cost
- Compile-time validation
- Embedded systems with limited runtime
- Type-level programming

**When to use:**
- Array sizes and bounds
- Lookup tables
- Configuration constants
- Static assertions

---

## Structural Patterns

### Compose Structs

Builds complex types by composing smaller, focused structs rather than creating monolithic types.

**How it works:**
- Break large structs into smaller components
- Compose via fields
- Each component has single responsibility
- Easier to test and maintain

**When to use:**
- Large structs with distinct concerns
- Reusable components
- Clear separation of concerns
- Testing individual components

**When NOT to use:**
- Tiny structs where composition adds complexity
- Tightly coupled data
- When flat structure is clearer

---

### Prefer Small Crates

Organizational pattern that favors breaking projects into smaller, focused crates.

**Benefits:**
- Faster compilation (parallel compilation)
- Clear module boundaries
- Reusable components
- Easier testing

**When to use:**
- Large projects
- Reusable functionality
- Clear module boundaries
- Long compilation times

**When NOT to use:**
- Small projects
- Premature modularization
- Highly coupled code

---

### Contain Unsafety in Small Modules

Isolates `unsafe` code into small, well-documented modules with safe public APIs.

**How it works:**
- Encapsulate unsafe in small modules
- Provide safe wrappers
- Document safety invariants
- Make unsafe code easy to audit

**When to use:**
- All unsafe code
- FFI boundaries
- Performance-critical unsafe optimizations
- Low-level primitives

**When NOT to use:**
- Unsafe is avoidable
- Safe alternatives exist
- Without thorough documentation

---

## Foreign Function Interface (FFI) Patterns

### Object-Based APIs

Provides stable, C-compatible APIs by hiding Rust implementation details behind opaque pointers.

**How it works:**
- Expose opaque pointer types to C
- Provide constructor/destructor functions
- Methods take opaque pointer as first argument
- Hide Rust types behind void pointers

```rust
#[repr(C)]
pub struct MyObject {
    _private: [u8; 0],
}

#[no_mangle]
pub extern "C" fn myobject_new() -> *mut MyObject { /* ... */ }

#[no_mangle]
pub extern "C" fn myobject_free(ptr: *mut MyObject) { /* ... */ }
```

**When to use:**
- C API design
- Stable ABI
- Hiding Rust details from C
- Object-oriented C APIs

---

### Type Consolidation into Wrappers

Groups related FFI types and functions into wrapper types to provide more idiomatic Rust interfaces over C APIs.

**When to use:**
- Wrapping C libraries
- Providing safe Rust APIs
- Hiding unsafe C details
- Idiomatic Rust over C APIs

---

### FFI Error Handling

Converts between Rust's `Result` types and C-style error codes.

**How it works:**
- Return int error codes to C
- Use out parameters for success values
- Convert Rust errors to error codes
- Preserve error information

```rust
#[no_mangle]
pub extern "C" fn do_something(out: *mut i32) -> i32 {
    match internal_operation() {
        Ok(value) => {
            unsafe { *out = value; }
            0  // Success
        }
        Err(_) => -1  // Error code
    }
}
```

---

### FFI String Handling

**Accepting Strings:**
- Use `*const c_char` for C strings
- Convert with `CStr::from_ptr()`
- Validate UTF-8 if needed

**Returning Strings:**
- Allocate with `CString::new()`
- Use `into_raw()` to transfer ownership
- Document who frees the string

**When to use:**
- All FFI string operations
- C library interop
- System APIs

---

## Advanced Rust Idioms

### Concatenating Strings with format!

Use the `format!` macro for readable string concatenation.

```rust
let s = format!("{} {}", first, last);
```

**When to use:**
- Building strings from multiple parts
- String templates
- Readable concatenation

**When NOT to use:**
- Simple cases where `+` is clearer
- Performance-critical tight loops (use String::push_str)

---

### Iterating Over Option

Use `.iter()`, `.iter_mut()`, or `.into_iter()` on `Option` to iterate zero or one time.

```rust
for value in maybe_value.iter() {
    // Executes 0 or 1 times
}
```

**When to use:**
- Composing with iterator chains
- Optional processing in pipelines
- Uniform handling of Option and Iterator

---

### Pass Variables to Closure

Use `move` keyword to transfer ownership into closures.

```rust
let data = vec![1, 2, 3];
tokio::spawn(async move {
    process(data).await
});
```

**When to use:**
- Spawning threads or async tasks
- Closure outlives original scope
- Transferring ownership to closure

---

### Privacy for Extensibility

Make fields private and provide controlled access through methods.

**Benefits:**
- Change internals without breaking API
- Enforce invariants
- Future-proofing

**When to use:**
- Public APIs
- Types with invariants
- Library development

---

### Temporary Mutability

Use scopes or shadowing to create temporary mutable access.

```rust
let data = {
    let mut data = Vec::new();
    data.push(1);
    data.push(2);
    data  // Now immutable
};
```

**When to use:**
- Initialization requiring mutation
- Limiting mutability scope
- Enforcing immutability after setup

---

### On-Stack Dynamic Dispatch

Use trait objects with stack allocation (e.g., `&dyn Trait`) when possible.

**When to use:**
- Short-lived trait objects
- Avoiding heap allocation
- Local polymorphism

**When NOT to use:**
- Long-lived objects (use Box<dyn Trait>)
- When ownership transfer is needed
- Returning trait objects from functions

---

## References

### Official Documentation
- **The Rust Programming Language**: https://doc.rust-lang.org/book/
- **The Rustonomicon**: https://doc.rust-lang.org/nomicon/
- **Rust By Example**: https://doc.rust-lang.org/rust-by-example/
- **Async Book**: https://rust-lang.github.io/async-book/

### Advanced Pattern Resources
- **Rust Design Patterns**: https://rust-unofficial.github.io/patterns/
- **Cliffle's Typestate**: https://cliffle.com/blog/rust-typestate/
- **Will Crichton's Type-Level Programming**: https://willcrichton.net/notes/type-level-programming/
- **Sealed Traits Guide**: https://predr.ag/blog/definitive-guide-to-sealed-traits-in-rust/

### Async Resources
- **Tokio Documentation**: https://tokio.rs
- **Pin and Unpin (Cloudflare)**: https://blog.cloudflare.com/pin-and-unpin-in-rust/

### Related Documentation
- See **idiomatic-rust.md** for core patterns, error handling, iterators, and everyday Rust idioms
