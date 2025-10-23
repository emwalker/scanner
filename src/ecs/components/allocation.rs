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
    /// Could be a ScanId or AudioId
    pub allocated_to: Option<String>,

    /// Which scan has reserved this tuner (scan-level, long-lived)
    pub reserved_for_scan: Option<crate::ecs::ScanId>,
}

impl AllocationComponent {
    /// Create a new allocation component in Available state
    pub fn new() -> Self {
        Self {
            state: AllocationState::Available,
            allocated_at: None,
            allocated_to: None,
            reserved_for_scan: None,
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

    /// Reserve tuner for a specific scan
    pub fn reserve_for_scan(&mut self, scan_id: crate::ecs::ScanId) {
        self.reserved_for_scan = Some(scan_id);
    }

    /// Clear scan reservation
    pub fn clear_scan_reservation(&mut self) {
        self.reserved_for_scan = None;
    }

    /// Check if tuner has a scan reservation
    pub fn is_reserved(&self) -> bool {
        self.reserved_for_scan.is_some()
    }

    /// Check if tuner is reserved for a specific scan
    pub fn is_reserved_for_scan(&self, scan_id: crate::ecs::ScanId) -> bool {
        self.reserved_for_scan == Some(scan_id)
    }
}

impl Default for AllocationComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reserve_for_scan() {
        let mut component = AllocationComponent::new();
        let scan_id = crate::ecs::ScanId::new();

        component.reserve_for_scan(scan_id);

        assert!(component.is_reserved());
        assert_eq!(component.reserved_for_scan, Some(scan_id));
        assert!(!component.is_allocated());
    }

    #[test]
    fn test_clear_scan_reservation() {
        let mut component = AllocationComponent::new();
        let scan_id = crate::ecs::ScanId::new();

        component.reserve_for_scan(scan_id);
        component.clear_scan_reservation();

        assert!(!component.is_reserved());
        assert_eq!(component.reserved_for_scan, None);
    }

    #[test]
    fn test_is_reserved_for_scan() {
        let mut component = AllocationComponent::new();
        let scan_id1 = crate::ecs::ScanId::new();
        let scan_id2 = crate::ecs::ScanId::new();

        component.reserve_for_scan(scan_id1);

        assert!(component.is_reserved_for_scan(scan_id1));
        assert!(!component.is_reserved_for_scan(scan_id2));
    }
}
