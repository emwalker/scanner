//! Candidate ID component

use std::fmt;

/// Unique identifier for a candidate
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CandidateId {
    id: String,
}

impl CandidateId {
    /// Create a candidate ID from frequency and window
    pub fn new(frequency_hz: f64, window_id: usize) -> Self {
        Self {
            id: format!("{:.1}-{}", frequency_hz / 1e6, window_id),
        }
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for CandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}
