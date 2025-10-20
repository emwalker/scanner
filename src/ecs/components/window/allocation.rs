//! Window allocation component - tracks tuner allocation for window

use crate::hardware::pool::{TaskRequirements, TunerActivity, TunerId};

/// Window tuner allocation state
#[derive(Debug, Clone)]
pub enum WindowAllocationComponent {
    /// No allocation requested yet
    None,
    /// Allocation has been requested
    Requested {
        requirements: TaskRequirements,
        activity: TunerActivity,
        requester_id: String,
    },
    /// Tuner has been allocated
    Allocated { tuner_id: TunerId },
}

impl WindowAllocationComponent {
    pub fn new() -> Self {
        Self::None
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn is_requested(&self) -> bool {
        matches!(self, Self::Requested { .. })
    }

    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }

    pub fn request(
        &mut self,
        requirements: TaskRequirements,
        activity: TunerActivity,
        requester_id: String,
    ) {
        *self = Self::Requested {
            requirements,
            activity,
            requester_id,
        };
    }

    pub fn allocate(&mut self, tuner_id: TunerId) {
        *self = Self::Allocated { tuner_id };
    }

    pub fn clear(&mut self) {
        *self = Self::None;
    }

    pub fn tuner_id(&self) -> Option<&TunerId> {
        match self {
            Self::Allocated { tuner_id } => Some(tuner_id),
            _ => None,
        }
    }
}

impl Default for WindowAllocationComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_lifecycle() {
        let mut allocation = WindowAllocationComponent::new();
        assert!(allocation.is_none());

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        allocation.request(requirements, TunerActivity::Scanning, "test".to_string());
        assert!(allocation.is_requested());

        let device_id = crate::hardware::DeviceId::from_serial("mock", "dev1");
        let tuner_id = TunerId::new(device_id, 0);
        allocation.allocate(tuner_id.clone());
        assert!(allocation.is_allocated());
        assert_eq!(allocation.tuner_id(), Some(&tuner_id));

        allocation.clear();
        assert!(allocation.is_none());
    }
}
