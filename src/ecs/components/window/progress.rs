//! Window progress component - tracks window processing state

/// Window processing state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowProgressState {
    /// Window is waiting to be processed
    Pending,
    /// Window is currently being processed
    Processing,
    /// Window processing completed successfully
    Completed,
    /// Window processing failed
    Failed,
}

/// Component tracking window processing progress
#[derive(Debug, Clone)]
pub struct WindowProgressComponent {
    pub state: WindowProgressState,
}

impl WindowProgressComponent {
    pub fn new() -> Self {
        Self {
            state: WindowProgressState::Pending,
        }
    }

    pub fn is_pending(&self) -> bool {
        matches!(self.state, WindowProgressState::Pending)
    }

    pub fn is_processing(&self) -> bool {
        matches!(self.state, WindowProgressState::Processing)
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.state, WindowProgressState::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.state, WindowProgressState::Failed)
    }

    pub fn start_processing(&mut self) {
        self.state = WindowProgressState::Processing;
    }

    pub fn mark_completed(&mut self) {
        self.state = WindowProgressState::Completed;
    }

    pub fn mark_failed(&mut self) {
        self.state = WindowProgressState::Failed;
    }
}

impl Default for WindowProgressComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_progress_lifecycle() {
        let mut progress = WindowProgressComponent::new();
        assert!(progress.is_pending());

        progress.start_processing();
        assert!(progress.is_processing());

        progress.mark_completed();
        assert!(progress.is_completed());
    }

    #[test]
    fn test_window_progress_failure() {
        let mut progress = WindowProgressComponent::new();
        progress.start_processing();
        progress.mark_failed();
        assert!(progress.is_failed());
    }
}
