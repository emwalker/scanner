//! Pause signal for immediate cancellation of background operations

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared pause signal for immediate cancellation of background operations
#[derive(Debug, Clone)]
pub struct PauseSignal {
    paused: Arc<AtomicBool>,
}

impl PauseSignal {
    /// Create a new pause signal in unpaused state
    pub fn new() -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the pause signal (request immediate pause)
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    /// Clear the pause signal (allow resuming)
    pub fn unpause(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }
}

impl Default for PauseSignal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_signal_default_state() {
        let signal = PauseSignal::new();
        assert!(!signal.is_paused());
    }

    #[test]
    fn test_pause_signal_pause() {
        let signal = PauseSignal::new();
        signal.pause();
        assert!(signal.is_paused());
    }

    #[test]
    fn test_pause_signal_unpause() {
        let signal = PauseSignal::new();
        signal.pause();
        assert!(signal.is_paused());

        signal.unpause();
        assert!(!signal.is_paused());
    }

    #[test]
    fn test_pause_signal_clone() {
        let signal1 = PauseSignal::new();
        let signal2 = signal1.clone();

        signal1.pause();
        assert!(signal1.is_paused());
        assert!(signal2.is_paused());

        signal2.unpause();
        assert!(!signal1.is_paused());
        assert!(!signal2.is_paused());
    }
}
