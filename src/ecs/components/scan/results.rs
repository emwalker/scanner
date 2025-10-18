//! Scan results component

/// Component tracking scan results
#[derive(Debug, Clone)]
pub struct ScanResultsComponent {
    /// Number of signal candidates found
    pub candidates_found: usize,

    /// Number of candidates rejected
    pub candidates_rejected: usize,

    /// Number of stations discovered
    pub stations_discovered: usize,
}

impl ScanResultsComponent {
    /// Create a new results component
    pub fn new() -> Self {
        Self {
            candidates_found: 0,
            candidates_rejected: 0,
            stations_discovered: 0,
        }
    }

    /// Record a candidate found
    pub fn add_candidate(&mut self) {
        self.candidates_found += 1;
    }

    /// Record a candidate rejected
    pub fn reject_candidate(&mut self) {
        self.candidates_rejected += 1;
    }

    /// Record a station discovered
    pub fn add_station(&mut self) {
        self.stations_discovered += 1;
    }

    /// Get total candidates processed
    pub fn total_candidates(&self) -> usize {
        self.candidates_found + self.candidates_rejected
    }
}

impl Default for ScanResultsComponent {
    fn default() -> Self {
        Self::new()
    }
}
