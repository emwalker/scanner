# Advanced Rust Patterns Overview

This document provides a high-level enumeration of advanced Rust usage patterns, based on community resources and documentation.

## Type System Patterns

### Typestate Pattern
Encodes the states of a state machine into the type system, making invalid state transitions impossible at compile time. Operations are only available when the object is in certain states, with attempts to use operations in wrong states failing to compile.

### Newtype Pattern
Wraps a type in a single-field tuple struct. Used to encode validation logic into the type system, provide distinct types for values with same underlying representation, or implement external traits on external types.

### Phantom Types
Uses `PhantomData<T>` to mark things that "act like" they own a T, even though they don't store it. Zero-sized type used for:
- Unused lifetime parameters
- Type parameters that aren't directly used in struct fields
- Controlling variance
- Type-level state tracking without runtime cost

### Generic Type Parameters
Uses generic type parameters to create flexible, reusable code that works across different types while maintaining type safety.

### Const Generics
Permits values of integral types to be used as parameters to generic types, traits, and functions. Useful for fixed-size arrays, compile-time computations, and type-safe abstractions.

### Trait Objects (dyn Trait)
Represents a value of any type that implements a specific trait, using dynamic dispatch. Typically used via references (`&dyn Trait`) or smart pointers (`Box<dyn Trait>`, `Arc<dyn Trait>`). Trait objects are dynamically sized types (DSTs) and store two pointers: one to data and one to a vtable.

## Ownership and Lifetime Patterns

### Lifetime Annotations
Explicit markers that tell the compiler how long references should remain valid. Used in function signatures, structs, and enums to express relationships between references.

### Lifetime Elision
Rules that allow the compiler to infer lifetimes in common cases, reducing boilerplate.

### Smart Pointers
- **`Box<T>`**: Heap allocation with single ownership
- **`Rc<T>`**: Reference counting for shared ownership (single-threaded)
- **`Arc<T>`**: Atomic reference counting for shared ownership (thread-safe)
- **`RefCell<T>`**: Interior mutability with runtime borrow checking
- **`Mutex<T>` / `RwLock<T>`**: Thread-safe interior mutability

### Pin and Unpin
`Pin<P>` wraps a pointer and prevents its value from moving, critical for self-referential types in async code. `Unpin` is an auto trait indicating it's safe to move a pinned value. Most types implement `Unpin`; async futures often do not.

## Behavioral Patterns

### Command
Encapsulates actions as objects, allowing parameterization of clients with different requests, queuing of requests, and logging of operations.

### Interpreter
Defines a representation for a grammar and an interpreter that uses the representation to interpret sentences in the language.

### RAII Guards
Uses the type system and lifetimes to ensure resources are properly released when they go out of scope. Commonly used for locks, file handles, and other resources.

### Strategy
Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Often implemented using trait objects or generics.

### Visitor
Separates algorithms from the objects on which they operate, allowing new operations to be added without modifying the objects.

### Iterator Pattern
Rust's iterator trait provides a powerful abstraction for sequential traversal. Iterators are lazy, composable, and often optimized away by the compiler.

## Creational Patterns

### Builder Pattern
Constructs complex objects step by step. Often combined with typestate pattern to enforce required fields at compile time.

### Fold
Accumulates values by repeatedly applying a function. Generalization of operations like sum, product, or collecting into a data structure.

### Constructor Functions
Idiomatic way to create instances, typically implemented as associated functions named `new()`. Often combined with `Default` trait.

### Default Trait
Provides default values for types, enabling concise initialization and integration with other Rust patterns.

## Structural Patterns

### Compose Structs
Builds complex types by composing smaller, focused structs rather than creating monolithic types.

### Prefer Small Crates
Organizational pattern that favors breaking projects into smaller, focused crates for better modularity and compilation times.

### Contain Unsafety in Small Modules
Isolates `unsafe` code into small, well-documented modules with safe public APIs, making unsafe code easier to audit and reason about.

## Foreign Function Interface (FFI) Patterns

### Object-Based APIs
Provides stable, C-compatible APIs by hiding Rust implementation details behind opaque pointers and C-style function interfaces.

### Type Consolidation into Wrappers
Groups related FFI types and functions into wrapper types to provide more idiomatic Rust interfaces over C APIs.

### Idiomatic Errors
Converts between Rust's `Result` types and C-style error codes in FFI boundaries.

### String Handling
- **Accepting Strings**: Use `&str` or accept `CStr` when crossing FFI boundaries
- **Passing Strings**: Convert Rust strings to C-compatible null-terminated strings

## Async Patterns

### Futures and Async/Await
Futures represent values that will be available later. The `async`/`await` syntax provides ergonomic asynchronous programming, with the compiler transforming async functions into state machines.

### Pinning Patterns
Self-referential async futures must be pinned before polling. Common patterns include `Box::pin()`, `pin!()` macro, and ensuring futures are pinned before spawning.

### Stream Pattern
Asynchronous equivalent of iterators, yielding values over time rather than all at once.

## Compile-Time Computation

### Const Functions (const fn)
Functions that can be evaluated at compile time, allowing their results to be used in constant contexts like array sizes and constant declarations.

### Const Evaluation
The process of computing expression results during compilation, enabling zero-cost abstractions and eliminating runtime computation.

## Rust Idioms

### Use Borrowed Types for Arguments
Accept `&str` instead of `&String`, `&[T]` instead of `&Vec<T>`, and other borrowed types to make functions more flexible.

### Concatenating Strings with `format!`
Use the `format!` macro for readable string concatenation rather than manual `String` manipulation.

### Collections Are Smart Pointers
Understanding that `Vec<T>`, `String`, and other collections manage heap memory and implement `Deref` to their borrowed counterparts.

### Finalization in Destructors
Implement the `Drop` trait to perform cleanup when values go out of scope, critical for proper resource management.

### `mem::take()` and `mem::replace()`
Swap values out of borrowed context without cloning, useful for working around borrow checker limitations.

### On-Stack Dynamic Dispatch
Use trait objects with stack allocation (e.g., `&dyn Trait`) when possible to avoid heap allocation overhead.

### Iterating Over an Option
Use `.iter()`, `.iter_mut()`, or `.into_iter()` on `Option` to iterate zero or one time, composing well with iterator chains.

### Pass Variables to Closure
Use `move` keyword to transfer ownership into closures when needed, particularly for spawning threads or async tasks.

### Privacy for Extensibility
Make fields private and provide controlled access through methods, allowing internal changes without breaking API.

### Temporary Mutability
Use scopes or shadowing to create temporary mutable access to otherwise immutable values.

### Return Consumed Arg on Error
When a function takes ownership but fails, return the owned value back to the caller so they can retry or handle it differently.

## Conversion Traits

### `From` and `Into`
- Implement `From<T>` to convert from `T` to your type
- `Into<T>` is automatically implemented when `From<T>` is implemented
- Preferred over custom conversion methods

### `AsRef` and `AsMut`
Cheap reference-to-reference conversions, often used for accepting multiple types that can be viewed as the same reference type.

### `Deref` and `DerefMut`
Enables automatic dereferencing. Should be used for smart pointer types, not as a general conversion mechanism (which is an anti-pattern called "Deref Polymorphism").

## Anti-Patterns to Avoid

### Clone to Satisfy the Borrow Checker
Unnecessarily cloning values to work around borrow checker errors. Often indicates a design issue that should be addressed through refactoring.

### `#[deny(warnings)]`
Using this in library code can break downstream users when new compiler versions introduce new warnings. Use in local development, not in published code.

### Deref Polymorphism
Implementing `Deref` on newtypes to automatically convert to the inner type. This is surprising and goes against the purpose of newtypes. Use explicit conversions instead.

## Sources

Information compiled from:
- rust-unofficial.github.io/patterns/ - Rust Design Patterns book
- The Rust Programming Language Book (doc.rust-lang.org/book/)
- The Rustonomicon (doc.rust-lang.org/nomicon/)
- Rust By Example (doc.rust-lang.org/rust-by-example/)
- Various Rust community discussions and articles
- Official Rust documentation on std types and traits
