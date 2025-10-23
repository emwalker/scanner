//! Scan results component

/// Component tracking scan results
#[derive(Debug, Clone)]
pub struct ScanResultsComponent {
    /// Number of signals found
    pub signals_found: usize,

    /// Number of signals rejected
    pub signals_rejected: usize,

    /// Number of stations discovered
    pub stations_discovered: usize,
}

impl ScanResultsComponent {
    /// Create a new results component
    pub fn new() -> Self {
        Self {
            signals_found: 0,
            signals_rejected: 0,
            stations_discovered: 0,
        }
    }

    /// Record a signal found
    pub fn add_signal(&mut self) {
        self.signals_found += 1;
    }

    /// Record a signal rejected
    pub fn reject_signal(&mut self) {
        self.signals_rejected += 1;
    }

    /// Record a station discovered
    pub fn add_station(&mut self) {
        self.stations_discovered += 1;
    }

    /// Get total signals processed
    pub fn total_signals(&self) -> usize {
        self.signals_found + self.signals_rejected
    }
}

impl Default for ScanResultsComponent {
    fn default() -> Self {
        Self::new()
    }
}
