# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Common Commands

### Rust code
- Add mimimal comments. Infrequent comments are ok, but in general we don't need them.
- Don't include development-specific comments that wouldn't make sense to people not involved in the development process (e.g., "changed this to that", "increased width of such-and-such", or other references to previous iterations)
- When adding or modifying a debug! log statement, use the json key value style: `debug!(key1 = value1, key2 = value2, ...)`
- When adding debug output, prefer `debug!` to `eprintln!` or `println!`.
- The `info!` macro is used for user-facing output, so don't use it.
- Avoid the `get_` prefix in function names. Follow Rust conventions by omitting `get_` for accessor methods. Use `refined_frequency` instead of `get_refined_frequency`, `name` instead of `get_name`, etc. This differs from languages like Java/C# where `get_`/`set_` prefixes are common.
- Keep methods at or below 20 lines if possible
- Follow idiomatic Rust import style:
  - Import types and functions directly (e.g., `use std::collections::HashMap;` then use `HashMap`, not `collections::HashMap`)
  - Use namespaces as an alternative to "as" for disambiguation (e.g., import `use std::io;` and `use tokio::io;` then use `std::io::Error` vs `tokio::io::Error`)
  - Only use namespaces at call sites for disambiguation or when it improves clarity
  - Avoid long qualified paths like `foo::bar::baz::Thing` - instead import intermediate modules if needed
  - Exception: For enum variants, prefer keeping the enum name (e.g., `Status::Success` not `Success`)

### Building and Checking
- `cargo check` - Check for syntax errors and basic correctness (feel free to use anytime)
- `cargo build` - Build the project
- `cargo run -- scan --stations 88.9e6 --duration 1 --json` - Run tuned to specific frequency (88.9 MHz)
- When checking `--band fm`, use a timeout command with a suitable timeout
- When troubleshooting a test, add `let _ = tracing_subscriber::fmt::try_init();` and use the `debug!` output if it's already available.

### Testing and Code Quality
- The user will use the `/fix` when they want to run the tests and fix linting issues
- Do NOT proactively run `cargo test` or `make lint` - wait for the user to run `/fix` at the appropriate time
- The `/fix` command will handle both testing and linting comprehensively
- **Dead code warnings are NOT acceptable** - remove all unused functions, constants, fields, and imports
  - Use `#[allow(dead_code)]` only if the code will be used soon or is intentionally kept for future use
  - When in doubt, remove the dead code rather than suppressing the warning

# Documentation and Planning
- When reading or writing plans, design documents, or issue documentation:
  - Do NOT include time estimates (e.g., "1-2 days", "2-3 weeks") as they are often unrealistic and can bias future work
  - Focus on concrete tasks, dependencies, and completion criteria instead
  - If existing documents contain estimates, ignore them and focus on the actual work to be done

# Committing to Git
- Do not commit to Git. Wait for the user to do the committing to Git.
