//! Tune allocation component - tracks tuner acquisition state

use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuneAllocationState {
    Pending,
    Allocated,
    Active,
    Failed,
}

#[derive(Debug, Clone)]
pub struct TuneAllocationComponent {
    state: TuneAllocationState,
    state_changed_at: Instant,
}

impl TuneAllocationComponent {
    pub fn new() -> Self {
        TuneAllocationComponent {
            state: TuneAllocationState::Pending,
            state_changed_at: Instant::now(),
        }
    }

    pub fn state(&self) -> TuneAllocationState {
        self.state.clone()
    }

    pub fn state_changed_at(&self) -> Instant {
        self.state_changed_at
    }

    pub fn transition(&mut self, new_state: TuneAllocationState) {
        self.state = new_state;
        self.state_changed_at = Instant::now();
    }
}

impl Default for TuneAllocationComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_allocation_initial_state() {
        let component = TuneAllocationComponent::new();
        assert_eq!(component.state(), TuneAllocationState::Pending);
    }

    #[test]
    fn test_tune_allocation_state_transitions() {
        let mut component = TuneAllocationComponent::new();
        assert_eq!(component.state(), TuneAllocationState::Pending);

        component.transition(TuneAllocationState::Allocated);
        assert_eq!(component.state(), TuneAllocationState::Allocated);

        component.transition(TuneAllocationState::Active);
        assert_eq!(component.state(), TuneAllocationState::Active);
    }

    #[test]
    fn test_tune_allocation_failed_state() {
        let mut component = TuneAllocationComponent::new();
        component.transition(TuneAllocationState::Failed);
        assert_eq!(component.state(), TuneAllocationState::Failed);
    }

    #[test]
    fn test_state_changed_at_updates() {
        let before = Instant::now();
        let mut component = TuneAllocationComponent::new();
        let after = Instant::now();

        assert!(component.state_changed_at() >= before);
        assert!(component.state_changed_at() <= after);

        let transition_before = Instant::now();
        component.transition(TuneAllocationState::Allocated);
        let transition_after = Instant::now();

        assert!(component.state_changed_at() >= transition_before);
        assert!(component.state_changed_at() <= transition_after);
    }
}
