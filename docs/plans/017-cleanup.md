# Plan 017: Code Organization and Testability

## Guidance for Updates

When updating this plan as work progresses, avoid adding:
- Lists of accomplishments or completion summaries
- Self-aggrandizement or subjective quality assessments
- Rationales and benefits sections (unless specifically requested)
- Speculation about future improvements or possibilities
- Time estimates or risk assessments

Keep updates matter-of-fact and focused on concrete technical details. Simply check off completed tasks and add technical notes as needed.

## Proposal 1: Trait Abstraction for Audio Quality Analysis

Extract audio quality analysis in squelch block behind a trait to enable dependency injection for testing.

### Status: Already Implemented

The audio quality analysis already has complete trait abstraction:
- `Classifier` trait defined in src/audio/quality/mod.rs with `analyze()` method
- All classifiers (heuristic1, heuristic2, heuristic3) implement the trait
- `AudioAnalyzer` wraps `Arc<dyn Classifier>` for dependency injection
- `AudioAnalyzer::mock()` and `MockClassifier` available for tests
- SquelchBlock accepts `AudioAnalyzer` with full trait object support

### Tasks
- [x] Define `AudioQualityAnalyzer` trait with `analyze()` method
- [x] Implement trait for existing `AudioAnalyzer` type
- [x] Update `SquelchBlock` to accept trait object instead of concrete type
- [x] Update `SquelchConfig` to use trait-based analyzer
- [x] Create mock analyzer implementation for tests
- [x] Update existing tests to use mock analyzer where appropriate
- [x] Run `make lint` and `make test`, fix any issues

## Proposal 2: Split Large Files

Improve navigability by splitting large files into focused modules.

### Status: Completed (Partial)

scanner_state refactoring completed successfully. squelch.rs analysis determined that splitting would reduce cohesion without clear benefits - the file is well-organized with logical sections.

### Tasks
- [x] Move src/scanner_state.rs tests (lines 425-771) to scanner_state/tests.rs
- [x] Update src/scanner_state.rs to use `mod tests` for test module
- [x] Evaluate squelch.rs split - determined unnecessary (file is well-organized at 652 lines)
- [x] Run `make lint` and `make test`, fix any issues

## Proposal 3: Reduce SquelchBlock Cognitive Complexity

Apply Single Responsibility Principle by extracting helper types from SquelchBlock.

### Tasks
- [ ] Create `AudioCollector` type for buffering samples during learning phase
- [ ] Create `FileCoordinator` type for managing audio capture state transitions
- [ ] Create `SignalReporter` type for handling signal generation and progress events
- [ ] Update `SquelchBlock` to use these helper types
- [ ] Move relevant methods to appropriate helper types
- [ ] Update tests to verify helper types independently
- [ ] Run `make lint` and `make test`, fix any issues
