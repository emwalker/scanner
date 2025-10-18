//! Allocation component - tracks tuner allocation state

use std::time::Instant;

/// Allocation state for a tuner
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationState {
    /// Tuner is available for allocation
    Available,
    /// Tuner is currently allocated to a task
    Allocated,
}

/// Component tracking allocation status of a tuner
///
/// This component manages whether a tuner is currently in use,
/// when it was allocated, and what it's allocated to.
#[derive(Debug, Clone)]
pub struct AllocationComponent {
    /// Current allocation state
    pub state: AllocationState,

    /// When the tuner was allocated (None if Available)
    pub allocated_at: Option<Instant>,

    /// What entity this tuner is allocated to (None if Available)
    /// Could be a ScanId or AudioSessionId
    pub allocated_to: Option<String>,
}

impl AllocationComponent {
    /// Create a new allocation component in Available state
    pub fn new() -> Self {
        Self {
            state: AllocationState::Available,
            allocated_at: None,
            allocated_to: None,
        }
    }

    /// Mark tuner as allocated
    pub fn allocate(&mut self, allocated_to: String) {
        self.state = AllocationState::Allocated;
        self.allocated_at = Some(Instant::now());
        self.allocated_to = Some(allocated_to);
    }

    /// Mark tuner as available (deallocate)
    pub fn deallocate(&mut self) {
        self.state = AllocationState::Available;
        self.allocated_at = None;
        self.allocated_to = None;
    }

    /// Check if tuner is available
    pub fn is_available(&self) -> bool {
        self.state == AllocationState::Available
    }

    /// Check if tuner is allocated
    pub fn is_allocated(&self) -> bool {
        self.state == AllocationState::Allocated
    }
}

impl Default for AllocationComponent {
    fn default() -> Self {
        Self::new()
    }
}
