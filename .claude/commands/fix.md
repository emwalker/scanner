---
description: Run tests and fix linting issues automatically
argument-hint: [optional test filter]
---

I need you to run tests and fix linting issues for this Rust project. Follow these steps:

1. **Run Tests**: Execute `cargo test $ARGUMENTS` to run all tests (or filtered tests if arguments provided)
   - If tests fail, analyze the failures and fix the issues
   - Re-run tests until they pass

2. **Fix Linting**: Run `make lint` to format code and fix clippy issues
   - The Makefile runs: `cargo fmt` and `cargo clippy --fix --allow-dirty`
   - If there are remaining linting issues, fix them manually

3. **Verify**: Run `cargo check` to ensure everything compiles correctly

4. **Report**: Provide a summary of:
   - Test results
   - Linting fixes applied
   - Any remaining issues that need manual attention

Follow the project guidelines from CLAUDE.md:
- Keep methods at or below 20 lines if possible
- Use minimal comments
- Prefer `debug!` over `println!` for debug output
- Use json key-value style for debug statements: `debug!(key1 = value1, key2 = value2)`

**Important**: Do not leave any issues as "good enough" - fix ALL test failures and linting issues completely.
