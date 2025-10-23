# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not be sycophantic. Avoid flattery or agreement for its own sake; challenge the user when logic or evidence demands.

## License Compliance for External Code

**IMPORTANT**: Before examining any external code (libraries, example projects, etc.), always check the license first.

- **If GPL, AGPL, or similar copyleft license**: DO NOT read the code directly
  - Skip over the project source code entirely
  - Instead, read about the project from:
    - Documentation (README, API docs)
    - Forum discussions (Reddit, forums, Stack Overflow)
    - Blog posts and tutorials
    - Issue discussions and release notes
  - You can understand the architecture and patterns without viewing copyleft code

- **If MIT, Apache, BSD, or permissive license**: Safe to read and learn from the code

- **When in doubt**: Treat as copyleft - stick to documentation and discussions only

This ensures we avoid any licensing concerns while still learning from the broader ecosystem.

## Rust code

- This is a greenfield project, so we don't need to maintain backwards compatability.
- Add mimimal comments. Infrequent comments are ok, but in general we don't need them.
- Don't include development-specific comments that wouldn't make sense to people not involved in the development process (e.g., "changed this to that", "increased width of such-and-such", or other references to previous iterations)
- When adding or modifying a `debug!` or `info!` log statement, use the json key value style: `debug!(key1 = value1, key2 = value2, ...)`
- Avoid the `get_` prefix in function names. Follow Rust conventions by omitting `get_` for accessor methods. Use `refined_frequency` instead of `get_refined_frequency`, `name` instead of `get_name`, etc. This differs from languages like Java/C# where `get_`/`set_` prefixes are common.
- Keep methods at or below 20 lines if possible
- Follow idiomatic Rust import style:
  - Import types and functions directly (e.g., `use std::collections::HashMap;` then use `HashMap`, not `collections::HashMap`)
  - Use namespaces as an alternative to "as" for disambiguation (e.g., import `use std::io;` and `use tokio::io;` then use `std::io::Error` vs `tokio::io::Error`)
  - Only use namespaces at call sites for disambiguation or when it improves clarity
  - Avoid long qualified paths like `foo::bar::baz::Thing` - instead import intermediate modules if needed
  - **Avoid wildcard imports** (`use foo::*;`) in production code - use explicit imports to make dependencies clear and improve maintainability
    - Exception: `use super::*;` is acceptable in test modules (`#[cfg(test)]`)
  - Exception: For enum variants, prefer keeping the enum name (e.g., `Status::Success` not `Success`)
- Return results instead of panicking, even in "impossible" code paths
- **Error handling with unwrap/expect:**
  - ✅ **In tests**: `unwrap()` and `expect()` are acceptable and idiomatic - tests should fail fast on unexpected conditions
  - ❌ **In runtime/production code**: Don't use `unwrap()` or `expect()` - use proper error handling with `Result`/`Option` and the `?` operator
- Prefer module-private over `pub(crate)` and `pub(super)`, and prefer `pub(crate)` and `pub(super)` over `pub`. But use `pub` when it makes sense.
- Don't use sleeps in tests to prevent races; use a deterministic approach instead.
- When adding loops, be careful to avoid busy waits

## Shutdown Safety

**CRITICAL**: All code changes must be cognizant of and prioritize shutdown safety to prevent deadlocks and hangs.

Key principles:
- **Never block during shutdown** - Use `try_lock()` instead of `lock()` for mutexes that might be held during teardown
- **Use atomic flags for shutdown state** - Check shutdown mode with lock-free atomics before acquiring locks
- **Make Drop implementations non-blocking** - Drop should use `try_lock()` and gracefully handle lock contention
- **Return early on shutdown** - Operations should check shutdown state and fail fast rather than waiting
- **Test shutdown scenarios** - Add tests for concurrent shutdown, locked resources, and drop-during-shutdown cases

Example patterns:
```rust
// Good: Non-blocking shutdown check
if self.shutdown_mode.load(Ordering::SeqCst) {
    return Err("Shutting down");
}

// Good: Try-lock in Drop
impl Drop for Resource {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pool.try_lock() {
            pool.return_resource(self.id.clone());
        }
    }
}

// Bad: Blocking lock during shutdown (can deadlock!)
impl Drop for Resource {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();  // ❌ Can hang forever
        pool.return_resource(self.id.clone());
    }
}
```

See `src/pool/mod.rs` for reference implementation of shutdown-safe patterns.

## Building and Checking
- `make lint` - Check for syntax errors and basic correctness
- `cargo build` - Build the project
- `cargo run - scan --stations 88.9e6 --duration 1 --headless` - Run tuned to specific frequency (88.9 MHz)
- When checking `--band fm`, use a timeout command with a suitable timeout

## Using rust-analyzer MCP

The rust-analyzer MCP provides semantic understanding of Rust code. **Always prefer rust-analyzer tools over Grep/Read/manual editing for Rust symbol operations.**

**Decision tree for Rust code work:**
- Looking for all usages of a symbol? → `find_references` (not Grep)
- Renaming something? → `rename_symbol` (not find/replace, NEVER manual editing)
- Looking for where something is defined? → `find_definition` (not Grep)
- Searching for types/traits/functions by name? → `workspace_symbols` (not Grep)
- Just made changes? → `get_diagnostics` (faster than cargo build) + `organize_imports` (not manual)
- Refactoring code? → `extract_function`, `inline_function`, `move_items` (not manual editing)
- Checking compilation? → `run_cargo_check` (faster than cargo build)
- Have compiler errors? → `get_diagnostics` (not reading terminal output)

**When to use standard tools (Grep/Read):**
- Searching string literals across multiple file types
- Reading non-Rust files (markdown, logs, etc.)
- Finding TODO comments
- Simple text patterns without semantic meaning

**See `.claude/skills/rust-analyzer-tools/SKILL.md` for the full Pre-Flight Checklist and detailed tool documentation.**

## Debugging and Logs
- **NEVER pipe `cargo run` output to `grep`, `sed`, `awk`, etc.** - The scanner uses subprocesses that write to separate log files
- Log files are written to `/tmp/scanner*.log` (e.g., `/tmp/scanner-worker-*.log`)
- To debug, run the scanner first, then examine the log files in /tmp
- Use `cat /tmp/scanner*.log | grep pattern` to search logs AFTER the run completes
- Use `RUST_LOG=debug` or `RUST_LOG=trace` to increase log verbosity

## Testing and Code Quality
- **Dead code warnings are NOT acceptable** - remove all unused functions, constants, fields, and imports
  - Use `#[allow(dead_code)]` only if the code will be used soon or is intentionally kept for future use
  - When in doubt, remove the dead code rather than suppressing the warning
- When updating tests after a significant change, try to bring failing assertions up to date rather than deleting them. And if that proves challenging, ask for guidance on what to do.

# Documentation and Planning
- When reading or writing plans, design documents, or issue documentation:
  - Do NOT include time estimates (e.g., "1-2 days", "2-3 weeks") as they are often unrealistic and can bias future work
  - Focus on concrete tasks, dependencies, and completion criteria instead
  - If existing documents contain estimates, ignore them and focus on the actual work to be done
- Avoid bold text, all-caps and and emojis
- Lean towards paragraphs over bullet points, although use bullet points when they make sense
- Avoid sections that enumerate rationales, time estimates, benefits, accomplishments, self-aggrandizement, speculation, and future improvements, unless these things are specifically requested.
- Be concise and matter-of-fact

# Committing to Git
- Do not commit to Git. Wait for the user to do the committing to Git.
