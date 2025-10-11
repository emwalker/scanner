# Idiomatic Rust Patterns

This document catalogs core patterns that every Rust developer should know for writing idiomatic Rust code. For advanced type system patterns, async patterns, and specialized techniques, see `rust-advanced.md`.

**Organization:**
- **rust-idiomatic.md** (this file): Everyday patterns and idioms used in most Rust code
- **rust-advanced.md**: Advanced type system features, compile-time computation, FFI, and specialized patterns

## Core Ownership Patterns

### Borrowed Types for Arguments

Use borrowed types (`&str`, `&[T]`, `&Path`) instead of owned types (`String`, `Vec<T>`, `PathBuf`) for function parameters.

**How it works:**
- Accept the most general borrowed form of a type
- `&str` accepts `&String`, `&str`, and string literals
- `&[T]` accepts `&Vec<T>`, `&[T]`, and arrays
- Enables callers to pass any compatible type without cloning

```rust
// Good
fn process_data(s: &str) { }

// Not idiomatic
fn process_data(s: &String) { }
```

**When to use:**
- Function parameters that only need to read data
- Public APIs where maximum flexibility is desired
- Functions that don't need ownership
- Library APIs and public interfaces

**When NOT to use:**
- Function needs to take ownership or store the value
- Function needs to mutate and return owned data
- Performance-critical code where coercion overhead matters (rare)

**References:** Rust API Guidelines, Pascal Hertleif's "Elegant Library APIs"

---

### RAII (Resource Acquisition Is Initialization)

Pattern where resource lifetime is tied to object lifetime, with automatic cleanup via the Drop trait.

**How it works:**
- Constructor acquires the resource
- Drop trait implementation releases the resource
- Rust guarantees Drop is called when value goes out of scope
- Cannot manually call drop() - use `std::mem::drop()` for early cleanup

**When to use:**
- Managing any resource requiring cleanup (files, sockets, locks, database connections)
- Implementing guard types (MutexGuard, File handles)
- Ensuring cleanup happens even on early returns or panics
- Creating scoped behaviors (timing blocks, temporary state changes)
- All types holding OS or external resources

**When NOT to use:**
- Types that don't own resources requiring cleanup
- When resource lifetime is not tied to a single owner
- Resources managed by reference counting with complex lifetimes (may still need Drop)

**References:** Rust By Example, Effective Rust Item 11, Rust Design Patterns book

---

### Smart Pointers (Box, Rc, Arc)

Pointer types with additional capabilities beyond regular references.

**Box<T>**: Single ownership, heap allocation
- Zero-cost abstraction for heap allocation
- Enables recursive types
- Fixed size on stack pointing to heap data

**Rc<T>**: Reference counted shared ownership (single-threaded)
- Multiple owners with shared immutable access
- Reference counting overhead
- Not thread-safe

**Arc<T>**: Atomic reference counted (multi-threaded)
- Thread-safe shared ownership
- Atomic operations have higher overhead than Rc
- Use with `Mutex` or `RwLock` for shared mutable state

**When to use Box:**
- Large values you want on the heap
- Recursive data structures
- Trait objects with unknown size
- Transferring ownership of heap data

**When to use Rc:**
- Multiple ownership in single-threaded contexts
- Tree structures with parent/child references
- Caches and memoization
- Immutable shared data

**When to use Arc:**
- Sharing data across threads
- Multi-threaded caches
- Read-heavy concurrent data structures
- Immutable configuration shared across threads

**When NOT to use:**
- When simple borrowing suffices
- When `&'static` lifetime works
- For small `Copy` types (pass by value)
- Arc in single-threaded code (use Rc instead)

**References:** The Rust Programming Language book ch15

---

### Interior Mutability (RefCell/Cell)

Pattern for mutating data through shared references, moving borrow checking from compile-time to runtime.

**How it works:**
- `Cell<T>` for `Copy` types - replaces entire value
- `RefCell<T>` for non-`Copy` types - provides runtime borrow checking
- Panics if borrow rules violated at runtime
- Often combined with `Rc<RefCell<T>>` for shared mutable state

**When to use:**
- Implementing data structures with complex sharing (graphs, caches)
- Mocking and testing when mutation is needed through shared references
- Implementing logical immutability with internal caching
- When compiler's borrow checker is too conservative for known-safe patterns

**When NOT to use:**
- In most code - prefer compile-time checking
- Multi-threaded contexts (use `Mutex` or `RwLock` instead)
- When ownership and borrowing can be structured normally
- Performance-critical code where runtime checks are prohibitive

**References:** The Rust Programming Language book ch15-05

---

### Newtype Pattern

A tuple struct with a single field that wraps another type to create a distinct type at compile time.

**How it works:**
- Creates a zero-cost abstraction around an existing type
- Provides type safety by making distinct types from the same underlying data
- Enables implementing traits on foreign types (working around orphan rule)
- Can encode validation logic into the type system

```rust
struct Miles(f64);
struct Kilometers(f64);
struct UserId(u64);  // Can't accidentally use as PostId
```

**When to use:**
- Distinguishing semantically different values with the same representation (units, IDs)
- Implementing foreign traits on foreign types
- Hiding implementation details while controlling the public interface
- Encoding validation guarantees in the type system (validated strings, non-zero numbers)
- Creating semantic types that prevent misuse

**When NOT to use:**
- When the type distinction adds no value or clarity
- For types that genuinely need to be interchangeable
- When runtime validation is more appropriate than compile-time guarantees
- Simple type aliases would suffice (use `type` instead)

**References:** Rust By Example, Effective Rust Item 6, Rust Design Patterns book

---

## Error Handling Patterns

### Result and Option Types

Rust's approach to fallible operations and optional values.

**Result<T, E>**: Represents success (Ok) or failure (Err)
- Forces explicit error handling
- Propagates with `?` operator
- Composable with combinators (map, and_then, etc.)

**Option<T>**: Represents presence (Some) or absence (None)
- Makes null states explicit
- Eliminates null pointer errors
- Composable with Result via `ok_or()`

**When to use Result:**
- Operations that can fail
- I/O operations
- Parsing and validation
- Any fallible computation requiring error context

**When to use Option:**
- Optional fields or parameters
- Collection lookups that might not find items
- Nullable values without error context
- Computations where absence is not an error

**When NOT to use:**
- Unrecoverable errors (use `panic!`)
- Programming errors (assert!/debug_assert!)
- When error type doesn't matter (use `anyhow::Result`)

**References:** The Rust Programming Language book ch09, Rust API Guidelines

---

### Error Libraries: thiserror vs anyhow

Two complementary approaches to error handling.

**thiserror**: For library error types
- Derive macro for implementing Error trait
- Creates concrete, strongly-typed errors
- Automatic Display implementation
- Preserves error type information

**anyhow**: For application error handling
- Type-erased error type
- Easy error propagation
- Context and error chaining
- Simplified error handling in binaries

**When to use thiserror:**
- Library code with public error types
- When callers need to match on error variants
- Defining domain-specific errors
- Recoverable errors requiring specific handling
- APIs where error type is part of contract

**When to use anyhow:**
- Application code (binaries, not libraries)
- Error reporting to users/logs
- When error details don't affect control flow
- Rapid prototyping
- Command-line tools

**When NOT to use:**
- Don't use anyhow in library public APIs
- Don't use thiserror if error details never matter
- Don't over-engineer error types for simple cases

**References:** anyhow and thiserror docs, Comprehensive Rust, Effective Rust

---

## Iterator Patterns

### Iterator Trait and Adapters

Lazy, composable iteration over sequences.

**How it works:**
- Iterator trait defines `next()` returning `Option<Item>`
- Adapters transform iterators without consuming (map, filter, take)
- Consumers drive iteration (collect, for_each, fold)
- Zero-cost abstractions - compiles to efficient code

**Common Adapters:**
- `map`: Transform elements
- `filter`: Select elements
- `flat_map`: Map and flatten
- `take`/`skip`: Limit iteration
- `chain`: Concatenate iterators
- `zip`: Pair elements from two iterators

**When to use:**
- Processing sequences of data
- Transforming collections
- Lazy computation
- Building data pipelines
- Avoiding intermediate allocations

**When NOT to use:**
- When indices are needed (use enumerate() or manual loops)
- Complex state machines where loops are clearer
- When iteration order or mutation patterns are complex
- Micro-optimizations where iterator overhead matters (rare)

**References:** The Rust Programming Language book ch13, std::iter docs

---

### Fold and Reduction

Reduce a sequence to a single value by repeatedly applying a function.

**How it works:**
- Takes an initial accumulator value
- Applies a closure to accumulator and each element
- Returns final accumulator value
- `fold(initial, |acc, item| ...)` or `reduce()`

**Common Use Cases:**
- Summing or multiplying sequences
- Building complex structures from elements
- Custom reductions beyond sum/product
- Combining elements with arbitrary logic

**When to use:**
- Any reduction operation
- Building aggregate values
- Complex accumulations beyond simple sums
- When sum(), product(), or collect() don't fit

**When NOT to use:**
- Use sum() or product() for simple numeric aggregations
- Use collect() for building collections
- When imperative loop is clearer for complex logic

**References:** Rust Design Patterns book, std::iter docs

---

## Builder and Constructor Patterns

### Builder Pattern

Construct complex objects step-by-step with a fluent API.

**How it works:**
- Separate builder struct with methods for each parameter
- Each method returns `self` for chaining
- Final `build()` method constructs the target type
- Handles optional parameters and validation

```rust
let server = ServerBuilder::new()
    .port(8080)
    .timeout(Duration::from_secs(30))
    .workers(4)
    .build()?;
```

**When to use:**
- Types with many optional parameters
- Complex initialization requiring validation
- When you need multiple constructors (Rust lacks overloading)
- APIs where configuration is built incrementally

**When NOT to use:**
- Simple types with few parameters (use `new()` or struct literals)
- When all parameters are required (use a regular constructor)
- Types where field order and naming are obvious
- When added complexity outweighs API ergonomics

**References:** Rust Design Patterns book, Rust API Guidelines

---

### Constructor Conventions

Idiomatic patterns for creating instances.

**new()**: Standard constructor
- Associated function (not method)
- Returns Self or Result<Self>
- Most common constructor name

**default()**: Default trait implementation
- Required for Default trait
- Zero-argument constructor
- Enables struct update syntax

**with_*()**: Named alternatives
- For types with multiple construction modes
- Example: `String::with_capacity()`

**from_*()**: Descriptive constructors
- When multiple "from X" constructors exist
- Example: `PathBuf::from_str()` vs `from_path_buf()`

**When to use:**
- Use `new()` as primary constructor
- Implement Default for sensible zero-value types
- Use `with_*()` for configuration variants
- Use builder pattern for complex construction

**When NOT to use:**
- Don't implement both new() and Default differently
- Avoid too many constructor alternatives (use builder)

**References:** Rust API Guidelines, Rust Design Patterns book

---

## Conversion Patterns

### Conversion Traits (From, Into, AsRef, AsMut)

Standard traits for type conversions.

**From/Into**: Consuming conversions
- `From` is preferred - `Into` is automatic
- Infallible conversions
- Takes ownership

**AsRef/AsMut**: Cheap reference conversions
- Cheap, non-failing conversions to references
- Used in generic APIs for flexibility
- Does not take ownership

**TryFrom/TryInto**: Fallible conversions
- Returns Result for conversions that might fail
- Preferred over custom error-prone methods

**When to use From/Into:**
- Converting between semantically equivalent types
- Constructor alternatives
- Making APIs more flexible
- Type conversions without failure

**When to use AsRef/AsMut:**
- Generic function parameters
- Accepting multiple reference types
- Cheap, non-failing conversions
- Standard library integration

**When NOT to use:**
- From for fallible conversions (use TryFrom)
- AsRef for expensive operations (use explicit methods)
- Into in signatures (use From, Into is inferred)

**References:** Rust API Guidelines, std::convert docs

---

### Deref Coercion

Implicit conversion from one reference type to another.

**How it works:**
- Implement `Deref` trait with associated `Target` type
- Compiler automatically inserts deref calls
- Enables smart pointers to act like references
- `&T` to `&U` when `T: Deref<Target=U>`

**When to use:**
- Smart pointer types (Box, Rc, Arc)
- Wrapper types that are transparent to Target
- Types that have clear "deref to" semantics
- Newtype pattern where inner type is primary interface

**When NOT to use:**
- Types that aren't pointer-like
- When deref behavior would surprise users
- For general "acts like" relationships (use AsRef)
- Creating inheritance-like hierarchies

**Warning:** Don't overuse for polymorphism - this is an anti-pattern called "Deref Polymorphism"

**References:** The Rust Programming Language book ch15, std::ops::Deref

---

## Performance Patterns

### Zero-Cost Abstractions

High-level abstractions that compile to efficient machine code.

**How it works:**
- Generics use monomorphization (static dispatch)
- Compiler generates specialized code for each type
- No virtual dispatch unless using trait objects
- Inlining and optimization produce minimal overhead

**Examples:**
- Iterators compile to loops
- Generics produce specialized functions
- Zero-sized types have no runtime representation
- Smart pointers like Box have no overhead vs raw pointers

**When to use:**
- Always - this is Rust's default
- Building high-level abstractions
- Writing generic, reusable code
- Performance-critical code that needs expressiveness

**When NOT to use:**
- When compile times become problematic (excessive monomorphization)
- Binary size is critical (consider dynamic dispatch)
- When trait objects provide needed flexibility

**References:** Rust Embedded book, Stack Overflow, Medium articles

---

### Clone-on-Write (Cow)

Smart pointer that defers cloning until mutation is needed.

**How it works:**
- Enum with Borrowed and Owned variants
- Starts as borrowed reference
- Clones only when mutation occurs via `to_mut()`
- Useful for functions that might modify input

```rust
fn process(input: Cow<str>) -> Cow<str> {
    if needs_modification {
        let mut s = input.into_owned();
        s.push_str("...");
        Cow::Owned(s)
    } else {
        input
    }
}
```

**When to use:**
- Functions that conditionally modify input
- Read-heavy workloads
- APIs accepting both borrowed and owned data
- Avoiding unnecessary allocations
- Return types that are sometimes borrowed, sometimes owned

**When NOT to use:**
- When data is always modified (just take ownership)
- When data is never modified (use &T)
- Performance-critical code where enum overhead matters
- Simple cases where borrowing or owning is clear

**References:** std::borrow docs, Easy Rust, Sling Academy

---

## Testing Patterns

### Module Organization for Tests

Rust's conventions for organizing test code.

**Unit Tests:**
- In same file as code being tested
- `#[cfg(test)]` module
- Can access private items

**Integration Tests:**
- `tests/` directory
- Each file is separate crate
- Can only use public API

**Doc Tests:**
- In doc comments
- Compiled and executed
- Show examples and verify they work

**When to use unit tests:**
- Testing individual functions
- Testing private implementation details
- Fast, focused tests
- Testing error conditions

**When to use integration tests:**
- Testing public API
- Testing component interactions
- End-to-end scenarios
- External interface validation

**References:** The Rust Programming Language book ch11

---

### Testable Design Principles

Structuring code to be easily testable.

**Key Techniques:**
- Dependency injection via traits
- Small, focused functions
- Pure functions without side effects
- Separate I/O from logic
- Parameterize over types with generics

**When to use:**
- All production code
- Library development
- Complex business logic
- Code requiring mocking
- Integration with external systems

**When NOT to use:**
- Over-engineering simple code
- When tests would be more complex than code
- Sacrificing clarity for testability

**References:** Rust testing documentation, community best practices

---

## Common Rust Idioms

### Collections as Smart Pointers

Understanding that `Vec<T>`, `String`, and other collections manage heap memory and implement `Deref` to their borrowed counterparts.

**How it works:**
- `Vec<T>` derefs to `&[T]`
- `String` derefs to `&str`
- Allows passing owned types to functions expecting borrowed types

**When to use:**
- Understanding method resolution
- Designing APIs with borrowed types
- Working with collection methods

---

### mem::take() and mem::replace()

Swap values out of borrowed context without cloning.

**How it works:**
- `mem::take(&mut T)` replaces with Default::default()
- `mem::replace(&mut T, new_value)` replaces with provided value
- Returns the old value
- Useful for working around borrow checker

**When to use:**
- Moving out of `&mut` reference
- Temporarily taking ownership
- Resetting values to default
- Implementing state transitions

**When NOT to use:**
- When you can restructure to avoid the need
- Type doesn't implement Default (use replace instead)

---

### Return Consumed Arg on Error

When a function takes ownership but fails, return the owned value back to the caller so they can retry or handle it differently.

**When to use:**
- Fallible operations that consume arguments
- Allowing retry after failure
- Giving caller maximum flexibility

**When NOT to use:**
- When partial processing makes returning meaningless
- Errors that make the value unusable

---

## Anti-Patterns to Avoid

### Clone to Satisfy Borrow Checker

Cloning data unnecessarily to avoid borrowing issues.

**Why it's bad:**
- Performance overhead
- Hides design problems
- Suggests misunderstanding of ownership

**Better approaches:**
- Restructure code to satisfy borrow checker
- Use references where possible
- Use Rc/Arc if multiple ownership is genuinely needed
- Understand why the borrow checker complains

---

### Deref Polymorphism

Using Deref to emulate inheritance or add unrelated methods.

**Why it's bad:**
- Breaks user expectations about Deref
- Makes trait resolution confusing
- Violates Deref's intended purpose

**Better approaches:**
- Use traits for polymorphism
- Use extension traits for adding methods
- Explicit wrapper methods

---

### #[deny(warnings)]

Denying all compiler warnings in published code.

**Why it's bad:**
- New compiler versions add warnings
- Breaks builds unnecessarily
- Prevents using newer Rust versions

**Better approaches:**
- Fix warnings instead
- Use `#[allow(specific_warning)]` when justified
- Use clippy for linting without breaking builds

**References:** Rust Design Patterns book (Anti-Patterns chapter)

---

## Pattern Interactions in Real Systems

Idiomatic Rust code typically combines multiple patterns for robust, efficient systems.

### Example: CLI Application
- **Error Handling**: anyhow for application errors, thiserror for library errors
- **Configuration**: Builder pattern with Default trait
- **Parsing**: From/TryFrom conversions
- **I/O**: RAII for file handles, Result for error handling
- **Processing**: Iterator chains for data transformation

### Example: Web Server
- **State**: Arc<RwLock<T>> for shared mutable state (see rust-advanced.md)
- **Errors**: thiserror for domain errors, anyhow in handlers
- **Resources**: RAII guards for database connections
- **API**: Builder pattern for server configuration
- **Type Safety**: Newtype for type-safe IDs and tokens

### Example: Data Processing Library
- **Abstractions**: Zero-cost with generics
- **Memory**: Cow for conditional cloning
- **Safety**: Newtype for validated data
- **API Design**: AsRef/Into for flexible parameters
- **Iterators**: Custom Iterator implementations for lazy evaluation

---

## References

### Official Documentation
- **Rust Book**: https://doc.rust-lang.org/book/
- **Rust by Example**: https://doc.rust-lang.org/rust-by-example/
- **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/
- **Standard Library Docs**: https://doc.rust-lang.org/std/

### Design Pattern Resources
- **Rust Design Patterns Book**: https://rust-unofficial.github.io/patterns/
- **Effective Rust**: https://effective-rust.com/
- **Pascal Hertleif - Elegant APIs**: https://deterministic.space/elegant-apis-in-rust.html

### Crates Mentioned
- **anyhow**: https://docs.rs/anyhow
- **thiserror**: https://docs.rs/thiserror

### Related Documentation
- See **rust-advanced.md** for advanced type system patterns, lifetimes, async, FFI, and compile-time computation
