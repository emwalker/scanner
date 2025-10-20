//! Segment component - wraps hardware::pool::Segment for ECS

use crate::hardware::pool::Segment;

/// Component that holds a hardware segment resource
///
/// Wraps the Segment type to make it part of the ECS architecture.
/// When this component is dropped (entity removed), the segment is
/// automatically cleaned up, stopping any associated SDR streams.
pub struct SegmentComponent {
    segment: Segment,
}

impl SegmentComponent {
    pub fn new(segment: Segment) -> Self {
        Self { segment }
    }

    pub fn segment(&self) -> &Segment {
        &self.segment
    }

    pub fn segment_mut(&mut self) -> &mut Segment {
        &mut self.segment
    }
}

impl std::fmt::Debug for SegmentComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentComponent")
            .field("segment", &"<Segment>")
            .finish()
    }
}
