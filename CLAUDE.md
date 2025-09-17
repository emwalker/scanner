# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Rust code
- Add mimimal comments. Infrequent comments are ok, but in general we don't need them.
- When adding or modifying a debug! log statement, use the json key value style: `debug!(key1 = value1, key2 = value2, ...)`
- When adding debug output, prefer `debug!` to `eprintln!` or `println!`.
- The `info!` macro is used for user-facing output, so don't use it.
- Avoid associated functions the `get_` prefix. Use `refined_frequency` instead of `get_refined_frequency`, for example.
- Keep methods at or below 20 lines if possible
- After making significant changes, run the tests: `cargo t`

### Building and Checking
- `cargo check` - Check for syntax errors and basic correctness
- `cargo build` - Build the project
- `cargo run -- scan --stations 88.9e6 --duration 1 --json` - Run tuned to specific frequency (88.9 MHz)
- `make lint` - Format code and run clippy with fixes
- When checking `--band fm`, use a timeout command with a suitable timeout
- When troubleshooting a test, add `let _ = tracing_subscriber::fmt::try_init();` and use the `debug!` output if it's already available.

# Committing to Git
- Use one-line commit messages
- Run `make lint` and fix all warnings. We should not create a commit with linter warnings.
- In commit messages, omit the "Generated with" line
- In commit messages, omit the "Co-Authored-By" line
- There is no need at this time to maintain backwards compatability, as this is a greenfield project.
- Don't include comments in generated code