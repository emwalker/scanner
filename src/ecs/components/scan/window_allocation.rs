use crate::hardware::pool::{TaskRequirements, TunerActivity, TunerId};

/// Component representing a window's tuner allocation request and state
///
/// This component tracks the lifecycle of tuner allocation for a window:
/// 1. Requested - WindowProcessingSystem requests allocation
/// 2. Allocated - AllocationSystem fulfills request
/// 3. None - Window processing complete, tuner returned
#[derive(Debug, Clone)]
pub enum WindowAllocationRequest {
    /// Allocation requested but not yet fulfilled
    Requested {
        window_index: usize,
        requirements: TaskRequirements,
        activity: TunerActivity,
        requester_id: String,
    },
    /// Allocation fulfilled, ready to spawn window thread
    Allocated {
        window_index: usize,
        tuner_id: TunerId,
        requester_id: String,
    },
}

impl WindowAllocationRequest {
    pub fn window_index(&self) -> usize {
        match self {
            Self::Requested { window_index, .. } => *window_index,
            Self::Allocated { window_index, .. } => *window_index,
        }
    }

    pub fn requester_id(&self) -> &str {
        match self {
            Self::Requested { requester_id, .. } => requester_id,
            Self::Allocated { requester_id, .. } => requester_id,
        }
    }

    pub fn is_requested(&self) -> bool {
        matches!(self, Self::Requested { .. })
    }

    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }
}
