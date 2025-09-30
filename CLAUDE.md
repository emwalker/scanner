# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Rust code
- Add mimimal comments. Infrequent comments are ok, but in general we don't need them.
- Don't include development-specific comments that wouldn't make sense to people not involved in the development process (e.g., "changed this to that", "increased width of such-and-such", or other references to previous iterations)
- When adding or modifying a debug! log statement, use the json key value style: `debug!(key1 = value1, key2 = value2, ...)`
- When adding debug output, prefer `debug!` to `eprintln!` or `println!`.
- The `info!` macro is used for user-facing output, so don't use it.
- Avoid the `get_` prefix in function names. Follow Rust conventions by omitting `get_` for accessor methods. Use `refined_frequency` instead of `get_refined_frequency`, `name` instead of `get_name`, etc. This differs from languages like Java/C# where `get_`/`set_` prefixes are common.
- Keep methods at or below 20 lines if possible
- Be eager to import namespaces, but use namespaces for disambiguation as well. Don't use "as" in an import, and instead use the namespace for disambiguation. But don't reference a long namespace with several parts in the code, either, if you can shorten it with some imports of the submodules.

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

# Committing to Git
- Do not commit to Git. Wait for the user to do the committing to Git.
