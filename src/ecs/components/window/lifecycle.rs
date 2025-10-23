//! Window lifecycle component - manages analysis state of a scanning window

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowLifecycleState {
    /// Peak detection is running or completed, waiting for analysis to start
    PeakDetectionComplete,
    /// Analysis threads are running
    AnalyzingSignals(usize), // Number of signals still being analyzed
    /// All analysis complete, window can be cleaned up
    Complete,
}

/// Component tracking the lifecycle and analysis progress of a window
#[derive(Debug, Clone)]
pub struct WindowLifecycleComponent {
    state: WindowLifecycleState,
}

impl WindowLifecycleComponent {
    pub fn new() -> Self {
        Self {
            state: WindowLifecycleState::PeakDetectionComplete,
        }
    }

    /// Get current lifecycle state
    pub fn state(&self) -> WindowLifecycleState {
        self.state
    }

    /// Transition to analyzing signals with given count
    pub fn start_analyzing(&mut self, signal_count: usize) {
        self.state = WindowLifecycleState::AnalyzingSignals(signal_count);
    }

    /// Check if we're currently analyzing
    pub fn is_analyzing(&self) -> bool {
        matches!(self.state, WindowLifecycleState::AnalyzingSignals(_))
    }

    /// Mark a signal as complete, decrement count
    /// Returns true if all signals are now complete
    pub fn complete_signal(&mut self) -> bool {
        match self.state {
            WindowLifecycleState::AnalyzingSignals(count) if count > 1 => {
                self.state = WindowLifecycleState::AnalyzingSignals(count - 1);
                false
            }
            WindowLifecycleState::AnalyzingSignals(1) => {
                self.state = WindowLifecycleState::Complete;
                true
            }
            _ => false,
        }
    }

    /// Check if window analysis is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.state, WindowLifecycleState::Complete)
    }
}

impl Default for WindowLifecycleComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_window_lifecycle() {
        let lifecycle = WindowLifecycleComponent::new();
        assert!(!lifecycle.is_analyzing());
        assert!(!lifecycle.is_complete());
        assert_eq!(
            lifecycle.state(),
            WindowLifecycleState::PeakDetectionComplete
        );
    }

    #[test]
    fn test_transition_to_analyzing() {
        let mut lifecycle = WindowLifecycleComponent::new();
        lifecycle.start_analyzing(3);

        assert!(lifecycle.is_analyzing());
        assert!(!lifecycle.is_complete());
        assert_eq!(lifecycle.state(), WindowLifecycleState::AnalyzingSignals(3));
    }

    #[test]
    fn test_complete_signals() {
        let mut lifecycle = WindowLifecycleComponent::new();
        lifecycle.start_analyzing(3);

        let all_complete = lifecycle.complete_signal();
        assert!(!all_complete);
        assert_eq!(lifecycle.state(), WindowLifecycleState::AnalyzingSignals(2));

        let all_complete = lifecycle.complete_signal();
        assert!(!all_complete);
        assert_eq!(lifecycle.state(), WindowLifecycleState::AnalyzingSignals(1));

        let all_complete = lifecycle.complete_signal();
        assert!(all_complete);
        assert!(lifecycle.is_complete());
    }
}
