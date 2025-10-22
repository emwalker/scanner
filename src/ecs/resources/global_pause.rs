//! Global pause state resource

use std::sync::{Arc, Mutex};

/// Global pause state for the entire application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalPauseState {
    /// Application is active (scanning, audio playing)
    Active,
    /// Application is globally paused
    Paused {
        /// Whether there were active scans before pausing
        had_active_scans: bool,
        /// Whether audio was playing before pausing
        had_active_audio: bool,
    },
}

/// Resource type for global pause state (thread-safe)
pub type GlobalPauseResource = Arc<Mutex<GlobalPauseState>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_pause_state_active_to_paused() {
        let state = GlobalPauseState::Active;
        assert!(matches!(state, GlobalPauseState::Active));

        let paused = GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: false,
        };
        assert!(matches!(
            paused,
            GlobalPauseState::Paused {
                had_active_scans: true,
                had_active_audio: false
            }
        ));
    }

    #[test]
    fn test_global_pause_state_paused_to_active() {
        let _state = GlobalPauseState::Paused {
            had_active_scans: true,
            had_active_audio: true,
        };
        let active = GlobalPauseState::Active;
        assert!(matches!(active, GlobalPauseState::Active));
    }

    #[test]
    fn test_global_pause_resource_creation() {
        let resource = Arc::new(Mutex::new(GlobalPauseState::Active));
        let state = resource.lock().unwrap();
        assert!(matches!(*state, GlobalPauseState::Active));
    }

    #[test]
    fn test_global_pause_resource_mutation() {
        let resource = Arc::new(Mutex::new(GlobalPauseState::Active));

        {
            let mut state = resource.lock().unwrap();
            *state = GlobalPauseState::Paused {
                had_active_scans: false,
                had_active_audio: true,
            };
        }

        let state = resource.lock().unwrap();
        assert!(matches!(
            *state,
            GlobalPauseState::Paused {
                had_active_scans: false,
                had_active_audio: true
            }
        ));
    }
}
