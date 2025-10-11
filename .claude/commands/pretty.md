---
description: Find Rust files that need cleanup and refactoring
allowed-tools: Bash(cargo:*), Bash(find:*), Bash(wc:*), Bash(grep:*), Read, Grep, Glob
---

You are an expert open-source Rust developer with many years of experience. Can you think hard and identify modifications to the codebase that would be beneficial, so that the resulting codebase is a little more pleasing to the eyes and will be in a good position for future modifications? Analyze the codebase for:

1. **Code Smell Detection**:
   - Functions longer than 20-30 lines
   - Dead code warnings from `cargo check`
   - Overly complex functions
   - Long parameter lists (>4 parameters)
   - Deeply nested code (>3 levels of indentation)

2. **Style Issues**:
   - Functions with `get_` prefix (should follow Rust conventions)
   - Non-idiomatic imports (not following project standards)
   - Departures from idiomatic Rust conventions
   - Cases where a new struct should be created and functions should be turned into associated funcions
   - Cases where a struct has too many responsibilities and should be separated into multiple structs to better separate concerns
   - Cases where we're not using the Iterator trait and we should be using it
   - Cases where we're using a custom error and a dedicated error variant would be beneficial

3. **Structural Issues**:
   - Large files (>500 lines) that could be split
   - Modules with unclear responsibilities
   - Duplicated code patterns
   - Architectural issues
   - Code in one module that would be better moved to another module
   - Cases where an elegant approach to something specific could prepare for future possibilities
   - Cases where a forward-looking approch could make life easier later on
   - Cases where the typestate pattern can be used to enforce compile-time correctness
   - See docs/patterns/ for related information

4. **Performance and Algorithmic Issues**
   - Code paths that could benefit from better algorithms
   - Code paths that are using core collections and data structure inefficiently
   - Code paths that are copying unnecessarily without unduly making the code complex
   - Code paths that are using locks when there is a lock-free alternative
   - Cases where we're re-inventing the wheel and would do better to delegate to an important crate

5. **Safety Issues**
   - Cases where a panic might unexpectedly result
   - Cases where shutdown safety will be impaired

6. **Testing Issues**
   - Cases where a different approach to a certain part of the code would make it easier to test
   - Cases where a trait would make it easier to test a component or area of the code

**Output Format**:

Print out some well-thought-out proposals for what steps to take next.

Prioritize modifications with the most issues or highest impact on code maintainability. Do an internet search to see what the internet thinks of these suggestions and modify them using the results of your search. This slash command should be convergent, in the sense that if you ran it enough times and followed the recommendations, fewer and fewer changes would be suggested, until at some point there are no more recommendations. Omit suggestions that are likely to be found unnecessary on further inspection.
