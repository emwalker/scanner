//! Segment component - wraps hardware::pool::Segment for ECS

use std::sync::Arc;

use crate::hardware::pool::Segment;

/// Component that holds a hardware segment resource
///
/// Wraps the Segment type to make it part of the ECS architecture.
/// Uses Arc to allow sharing the segment between worker result and window entity.
/// When the last Arc reference is dropped, the segment is automatically cleaned up.
pub struct SegmentComponent {
    segment: Arc<Segment>,
}

impl SegmentComponent {
    pub fn new(segment: Arc<Segment>) -> Self {
        Self { segment }
    }

    pub fn segment(&self) -> &Segment {
        &self.segment
    }
}

impl std::fmt::Debug for SegmentComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentComponent")
            .field("segment", &"<Segment>")
            .finish()
    }
}
