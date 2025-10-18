//! Scan lifecycle component

use std::time::Instant;

/// Component tracking scan lifecycle timestamps
#[derive(Debug, Clone)]
pub struct ScanLifecycleComponent {
    /// When the scan entity was created
    pub created_at: Instant,

    /// When scanning actually started (None if not yet started)
    pub started_at: Option<Instant>,

    /// When the scan was completed (None if not yet completed)
    pub completed_at: Option<Instant>,

    /// History of pause events
    pub pause_history: Vec<Instant>,
}

impl ScanLifecycleComponent {
    /// Create a new lifecycle component
    pub fn new() -> Self {
        Self {
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            pause_history: Vec::new(),
        }
    }

    /// Mark scan as started
    pub fn start(&mut self) {
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
    }

    /// Mark scan as completed
    pub fn complete(&mut self) {
        if self.completed_at.is_none() {
            self.completed_at = Some(Instant::now());
        }
    }

    /// Record a pause event
    pub fn pause(&mut self) {
        self.pause_history.push(Instant::now());
    }

    /// Get scan duration (from start to now or completion)
    pub fn duration(&self) -> Option<std::time::Duration> {
        self.started_at.map(|start| {
            if let Some(end) = self.completed_at {
                end.duration_since(start)
            } else {
                start.elapsed()
            }
        })
    }

    /// Check if scan has started
    pub fn is_started(&self) -> bool {
        self.started_at.is_some()
    }

    /// Check if scan has completed
    pub fn is_completed(&self) -> bool {
        self.completed_at.is_some()
    }

    /// Get number of times scan was paused
    pub fn pause_count(&self) -> usize {
        self.pause_history.len()
    }
}

impl Default for ScanLifecycleComponent {
    fn default() -> Self {
        Self::new()
    }
}
