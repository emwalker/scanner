//! Global pause state resource

use std::sync::{Arc, Mutex};

use crate::ecs::components::{StationId, WindowId};

/// Information about a station that was playing before global pause
#[derive(Debug, Clone, PartialEq)]
pub struct PlayingStationInfo {
    pub station_id: StationId,
    pub window_id: WindowId,
    pub frequency_hz: f64,
    pub center_frequency_hz: f64,
}

/// Global pause state for the entire application
#[derive(Debug, Clone, PartialEq)]
pub enum GlobalPauseState {
    /// Application is active (scanning, audio playing)
    Active,
    /// Application is globally paused
    Paused {
        /// Whether there were active scans before pausing
        had_active_scans: bool,
        /// Stations that were playing before pausing
        playing_stations: Vec<PlayingStationInfo>,
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
            playing_stations: vec![],
        };
        assert!(matches!(paused, GlobalPauseState::Paused {
            had_active_scans: true,
            ..
        }));
    }

    #[test]
    fn test_global_pause_state_paused_to_active() {
        let _state = GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
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
                playing_stations: vec![],
            };
        }

        let state = resource.lock().unwrap();
        assert!(matches!(*state, GlobalPauseState::Paused {
            had_active_scans: false,
            ..
        }));
    }
}
