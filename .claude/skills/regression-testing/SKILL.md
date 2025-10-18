---
name: Regression Testing
description: Create regression tests for bugs using hypothesis-driven debugging. Use when the user reports a bug and wants a test to prevent it from happening again. Guide systematic investigation, experiment tracking, test creation, and validation.
---

You are helping create a regression test for a reported bug. Follow this workflow carefully.

## When to Use This Skill

- User reports a bug or unexpected behavior
- User wants to ensure a bug doesn't regress
- User asks to "write a test for this bug" or "add a regression test"

## Workflow Overview

1. Investigation (hypothesis-driven experimentation)
2. Write failing test (only after high confidence in root cause)
3. Pause for developer to commit the failing test
4. Implement fix to make test pass
5. Validate bug is fixed in actual runtime
6. Revert and retry if bug persists

## Phase 1: Investigation

Goal: Find the root cause through experimentation, not guesswork.

### Process

1. Ask the user to describe the bug symptoms and reproduction steps
2. Form specific, testable hypotheses about possible root causes
3. Design small experiments to test each hypothesis
4. Use TodoWrite to track experiments:
   ```
   - [pending] Hypothesis: Pool deadlock in shutdown
   - [in_progress] Experiment: Add logging to Pool::drop
   - [completed] Evidence: Confirmed drop() blocks on mutex
   ```
5. Run experiments and gather evidence
6. Revert failed experiments: `git checkout -- <files>` or `git stash`
7. Continue until confident about root cause

### Guidelines

- Don't rush to write a test - validate hypotheses first
- Keep experiments small and reversible
- Test one hypothesis at a time
- Look for evidence: logs, stack traces, timing issues, edge cases
- Use Read tool to examine relevant code paths
- Use Bash tool to run scanner with --verbose flag
- Check log files in /tmp/scanner*.log for detailed output

### Example Investigation

```
User: "Scanner hangs on shutdown sometimes"

You:
- Hypothesis 1: Deadlock in Pool shutdown
  - Experiment: Run with --verbose and check /tmp/scanner*.log
  - Evidence: Log shows "waiting for mutex" but never "mutex acquired"
  - Conclusion: High confidence this is the cause

- Hypothesis 2: Infinite loop in coordinator
  - Experiment: Add timeout logging
  - Evidence: Timeout doesn't trigger
  - Conclusion: Not the cause, revert changes
```

## Phase 2: Write Failing Test

Only proceed when you have high confidence in the root cause.

### Test Characteristics

- **Black or gray box** - Test observable behavior, not implementation details
- **Resilient** - Should survive refactoring
- **Clear** - Demonstrates the bug obviously
- **Documented** - Comment explains what bug it prevents
- **Failing** - Has assertion that reproduces the bug

### Test Template

```rust
#[test]
fn test_<bug_description>() {
    // Regression test for <issue/symptom>
    // Root cause: <brief explanation>

    // Setup
    let system = create_test_system();

    // Trigger bug condition
    system.trigger_bug_condition();

    // Assert expected behavior (will fail until bug is fixed)
    assert!(
        system.correct_behavior(),
        "Bug: <description of what goes wrong>"
    );
}
```

### Process

1. Write test following project conventions, preferring adding to an existing `mod tests` module if one exists. Don't write placeholder tests with tautological assertions. Go to the effort of writing an actual integration test if that is what is needed.
2. Add explanatory comment linking to bug
3. Run test to verify it fails
4. Verify it fails for the RIGHT reason (the bug, not a test error)
5. Clean up any debug/experimental code

## Phase 3: Pause for Developer

Use AskUserQuestion before implementing the fix:

```
I've identified the root cause and written a failing regression test.

Root cause: [explain what's causing the bug]

Test: [describe what the test does]

The test currently fails as expected, reproducing the bug.

Please commit this failing test before I implement the fix. This ensures
we have a baseline showing the bug exists.

Options:
- "Committed" - I've committed the failing test, proceed with fix
- "Let me review first" - I want to look at the test
- "Different approach" - Try a different hypothesis
```

Wait for user confirmation before proceeding.

## Phase 4: Implement Fix

After user commits the failing test:

1. Implement the minimal fix to address the root cause
2. Run the regression test - should now PASS
3. Run full test suite - verify no new failures
4. Clean up any remaining experimental code

## Phase 5: Validate in Runtime

Use AskUserQuestion to verify the fix:

```
The fix is implemented and all tests pass, including the new regression test.

Can you verify the bug is fixed in the actual runtime/application?

Please test the original reproduction steps and confirm the bug no longer occurs.

Options:
- "Bug is fixed!" - The issue no longer happens
- "Still broken" - The bug still occurs
- "Different symptoms" - Something else is wrong now
```

### Outcomes

- **Bug is fixed**: Declare victory, task complete
- **Still broken**: Revert all changes, return to Phase 1 with new hypotheses
- **Different symptoms**: May have fixed one issue but revealed another

## Key Principles

- **Hypothesis-driven**: Form theories, test them systematically
- **Evidence-based**: Don't write tests until you understand the bug
- **Reversible**: Use git liberally, keep changes small
- **Behavior-focused**: Test what the system does, not how it does it
- **Collaborative**: Pause at checkpoints for developer input
- **Methodical**: Track experiments, don't skip validation steps

## What to Avoid

- Writing tests before understanding root cause (will rewrite repeatedly)
- White-box tests that depend on implementation details
- Jumping to a fix without validation (might fix wrong thing)
- Skipping developer checkpoints (they need to review/commit/verify)
- Leaving failed experiments in the codebase
- Testing too close to the implementation (test will break during refactoring)

## Useful Commands

- `git stash` - Save experimental changes temporarily
- `git checkout -- <file>` - Revert specific file
- `git diff` - Review changes before reverting
- `cargo test <test_name> -- --nocapture` - Run specific test with output
- `cargo run -- scan --stations 88.9e6 --verbose` - Run scanner with verbose logging
- `cat /tmp/scanner*.log` - Check detailed log files
- `RUST_BACKTRACE=1 cargo test <test_name>` - Run with backtrace on panic
