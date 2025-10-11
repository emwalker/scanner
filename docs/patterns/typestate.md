# Typestate Pattern in Rust

This document summarizes findings from various sources on the internet about the typestate pattern in Rust, including its variations, use cases, and alternatives.

## Overview

The typestate pattern encodes the states of a state machine into the type system. It has operations on an object that are only available when the object is in certain states, encoded at the type level, such that attempts to use operations in the wrong state fail to compile.

## Pattern Variations

### 1. Two-State Typestate

Uses separate types for different states (e.g., "living" and "dead"). Enforces state transitions at compile-time and prevents operations in incorrect states. This is commonly seen in RAII patterns in Rust.

**Advantages:**
- Simplest form of typestate
- Enforces compile-time state transitions
- Prevents operations in incorrect states
- Natural fit for RAII patterns

**Disadvantages:**
- Limited to only two states
- Not suitable for more complex state machines

### 2. Separate Structs for Each State

Each state is its own distinct struct with specific methods. State transitions consume ownership of the object and produce a new object in the next state.

```rust
struct StateA { /* ... */ }
struct StateB { /* ... */ }

impl StateA {
    fn transition(self) -> StateB {
        StateB { /* ... */ }
    }
}
```

**Advantages:**
- Clear, understandable error messages about invalid transitions
- States are part of the type signature
- Old state is consumed during transition, preventing access to stale values
- Everything happens on the stack with lean memory consumption
- Straightforward implementation

**Disadvantages:**
- More verbose implementation
- Can increase code complexity
- Requires more boilerplate code
- Might require implementing `From` or `Into` traits for each transition
- Type signatures can become long
- Requires handling all potential states when manipulating the machine

### 3. Generic Type Parameter Approach

Uses a generic struct with a state type parameter. Allows operations across different states and can define state-specific and cross-state methods. More flexible and concise than separate types.

```rust
struct Machine<State> {
    data: Vec<u8>,
    _marker: PhantomData<State>,
}
```

PhantomData is a zero-sized type that doesn't use any memory but allows satisfying the type checker.

**Advantages:**
- More flexible and concise than separate types
- Allows operations across different states
- Can define state-specific and cross-state methods
- Enables complex state transition logic

**Disadvantages:**
- Usage of generics increases binary size and compilation time
- Requires carrying around PhantomData marker or else Rust complains about unused type parameter
- Adds some runtime overhead with phantom data
- Steeper learning curve for developers unfamiliar with advanced type system techniques

### 4. Trait-Based Typestate

States are defined as traits rather than concrete types. This allows extending typestates over a family of types and provides more flexibility, though it diverges from most documentation on the topic and increases complexity.

**Advantages:**
- Allows extending typestates over a family of types
- More flexible than concrete struct-based states
- Can use multiple type parameters to represent different state transitions
- Enables decoupling of construction logic from other concerns

**Disadvantages:**
- Increases code complexity compared to simple struct-based implementations
- Requires careful design of trait methods and type parameters
- Diverges from most documentation on the topic
- Boxed trait objects don't work well for this pattern
- Runtime errors can still occur if state transitions are not perfectly managed

### 5. Hybrid Enum Wrapper Pattern

Combines typestate structs with an outer enum wrapper. This addresses the problem that each typestate may have a different size while still providing compile-time enforcement of transitions through the typestate API.

```rust
enum State {
    A(TypeStateA),
    B(TypeStateB),
    C(TypeStateC),
}
```

The enum handles runtime representation while the typestate types provide compile-time guarantees.

**Advantages:**
- Solves the problem that each state may have a different size
- Provides compile-time enforcement of transitions through the typestate API
- Can represent collections of items in different states
- Combines benefits of both typestate and enum approaches

**Disadvantages:**
- Requires implementing everything almost twice (both typestate types and enum variants)
- More complex than using either pattern alone
- May be overkill for simpler state machines

## General Guidance on Typestate Pattern

### When to Use

According to various sources, the typestate pattern is recommended when:

- Enforcing strict operation sequences at compile-time
- Preventing invalid state transitions in critical systems
- Creating robust, self-documenting APIs
- Designing state machines with compile-time guarantees
- Working on safety-critical systems requiring the strongest possible compile-time guarantees
- Building APIs with clear state machines
- Systems where runtime checks for state validity are expensive
- Need to enforce properties like "you must not perform any I/O operations on a file handle after it's been closed" or "these messages can only be sent after authentication has succeeded"

### When to Avoid

Sources recommend avoiding or reconsidering typestate when:

- Working in dynamic environments where you can't control the order of operations
- State transitions are determined at runtime based on conditions
- Simple state management is sufficient
- Type system complexity outweighs benefits
- Performance-critical code with minimal state complexity
- Systems with extremely dynamic state transitions
- Need to handle runtime conditional state changes (since Rust eliminates types during compilation)
- Working with collections of items that need to be in different states simultaneously
- Designing a builder with too many states (would be overkill and could have counterproductive side effects)

### Shared Benefits Across Typestate Variations

- Moves certain types of errors from runtime to compile-time, giving programmers faster feedback
- Eliminates runtime state checks, making code faster/smaller
- Interacts nicely with IDEs, which can avoid suggesting operations that are illegal in a certain state
- Creates self-documenting APIs

## Alternative: Enum-Based State Machines

Enum-based state machines use Rust's "fat enums" where all states are mutually exclusive and each state can carry data.

**Advantages:**
- Natural and concise representation of states
- Pattern matching is very ergonomic in Rust
- Compiler will error if you forget to handle a state transition
- Memory efficient - the size of the largest variant (fat enum is only as big as its biggest variant)
- Everything happens on the stack
- Performance optimized by the compiler, often matching direct integer use
- Most standard state management scenarios
- Readable, maintainable code
- Works well with collections of items in different states

**Disadvantages:**
- Invalid transitions can't be enforced at compile time
- Errors are found at runtime rather than compile-time
- Requires match statements that can become verbose
- Can manipulate states internally within a module without enforcement

**Community Recommendation:**

"Use enums whenever you need to represent a set of possible values, like when representing the state of an object. For even stronger guarantees, consider the typestate pattern, especially in safety-critical applications."

"If you want both compile-time and run-time you'll end up implementing everything almost twice. It's more pragmatic to implement the run-time version only and ensure it's correct with unit tests." - Kornel on Rust forums

## Real-World Examples

The highest-profile example of the typestate pattern in the Rust ecosystem is **serde**. The Serializer models a fairly complex state machine using typestates. For instance, the serialize_struct operation produces an object implementing the SerializeStruct trait, and the end method consumes the SerializeStruct and produces a result. You cannot accidentally call serialize_struct twice, or call both serialize_struct and serialize_i32, or add fields to the struct after calling end - attempting any of these will produce a compile error.

## Sources

Information compiled from:
- cliffle.com/blog/rust-typestate/
- willcrichton.net/rust-api-type-patterns/typestate.html
- hoverbear.org/blog/rust-state-machine-pattern/
- corrode.dev/blog/enums/
- developerlife.com/2024/05/28/typestate-pattern-rust/
- depth-first.com/articles/2023/02/28/using-the-typestate-pattern-with-rust-traits/
- Various Stack Overflow and Rust forum discussions
