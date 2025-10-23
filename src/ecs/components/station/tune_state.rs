//! TuneState enum - type-safe state machine for station tuning

use super::{TuneAllocationComponent, TuneRequestComponent, TuneTransitionComponent};
use crate::ecs::components::window::WindowId;

/// State machine representing station tuning lifecycle
///
/// Mutually exclusive states prevent impossible combinations:
/// - Can't have both Transitioning and RequestQueued
/// - Can't have request without allocation
/// - Each state transition is explicit
#[derive(Debug, Clone)]
pub enum TuneState {
    /// Station is not being tuned
    Idle,

    /// Pause/selection requested, waiting for TuneRequestSystem
    /// to process and enqueue
    Transitioning(TuneTransitionComponent),

    /// Tuner request queued, waiting for TunerAllocationSystem
    /// to acquire tuner and spawn audio
    RequestQueued {
        request: TuneRequestComponent,
        allocation: TuneAllocationComponent,
    },

    /// Tuner acquired, audio entity active or spawning
    Active { allocation: TuneAllocationComponent },
}

impl TuneState {
    /// Create a new Idle state
    pub fn idle() -> Self {
        TuneState::Idle
    }

    /// Create a new Transitioning state
    pub fn transitioning(window_id: WindowId, center_frequency: f64) -> Self {
        TuneState::Transitioning(TuneTransitionComponent::new(window_id, center_frequency))
    }

    /// Check if state is Idle
    pub fn is_idle(&self) -> bool {
        matches!(self, TuneState::Idle)
    }

    /// Check if state is Transitioning
    pub fn is_transitioning(&self) -> bool {
        matches!(self, TuneState::Transitioning(_))
    }

    /// Check if state is RequestQueued
    pub fn is_request_queued(&self) -> bool {
        matches!(self, TuneState::RequestQueued { .. })
    }

    /// Check if state is Active
    pub fn is_active(&self) -> bool {
        matches!(self, TuneState::Active { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{TaskId, components::window::WindowId};

    #[test]
    fn test_idle_state() {
        let state = TuneState::idle();
        assert!(state.is_idle());
        assert!(!state.is_transitioning());
        assert!(!state.is_request_queued());
        assert!(!state.is_active());
    }

    #[test]
    fn test_transitioning_state() {
        let window_id = WindowId::new(TaskId::new("test"), 5);
        let state = TuneState::transitioning(window_id, 88.9e6);
        assert!(!state.is_idle());
        assert!(state.is_transitioning());
        assert!(!state.is_request_queued());
        assert!(!state.is_active());
    }

    #[test]
    fn test_request_queued_state() {
        let window_id = WindowId::new(TaskId::new("test"), 5);
        let request = TuneRequestComponent::new(window_id);
        let allocation = TuneAllocationComponent::new();
        let state = TuneState::RequestQueued {
            request,
            allocation,
        };

        assert!(!state.is_idle());
        assert!(!state.is_transitioning());
        assert!(state.is_request_queued());
        assert!(!state.is_active());
    }

    #[test]
    fn test_active_state() {
        let allocation = TuneAllocationComponent::new();
        let state = TuneState::Active { allocation };

        assert!(!state.is_idle());
        assert!(!state.is_transitioning());
        assert!(!state.is_request_queued());
        assert!(state.is_active());
    }
}
