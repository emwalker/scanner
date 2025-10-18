//! Station discovery component

use crate::ecs::components::scan::ScanId;
use crate::scanning::window::WindowMetadata;
use std::time::Instant;

/// Component tracking how and when a station was discovered
#[derive(Debug, Clone)]
pub struct StationDiscoveryComponent {
    /// When the station was first discovered
    pub discovered_at: Instant,

    /// ID of the scan that discovered this station
    pub scan_id: ScanId,

    /// Window index within the scan
    pub window_id: usize,

    /// Metadata about the discovery window
    pub window_metadata: WindowMetadata,
}

impl StationDiscoveryComponent {
    /// Create a new discovery component
    pub fn new(scan_id: ScanId, window_id: usize, window_metadata: WindowMetadata) -> Self {
        Self {
            discovered_at: Instant::now(),
            scan_id,
            window_id,
            window_metadata,
        }
    }

    /// Get how long ago this station was discovered
    pub fn discovered_ago(&self) -> std::time::Duration {
        self.discovered_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_discovery() {
        let scan_id = ScanId::new();
        let metadata = WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        };

        let discovery = StationDiscoveryComponent::new(scan_id, 5, metadata);

        assert_eq!(discovery.scan_id, scan_id);
        assert_eq!(discovery.window_id, 5);
        assert_eq!(discovery.window_metadata.center_frequency_hz, 88.9e6);
        assert_eq!(discovery.window_metadata.window_id, 1);
    }

    #[test]
    fn test_discovered_ago() {
        let scan_id = ScanId::new();
        let metadata = WindowMetadata {
            center_frequency_hz: 88.9e6,
            window_id: 1,
        };

        let discovery = StationDiscoveryComponent::new(scan_id, 0, metadata);
        let elapsed = discovery.discovered_ago();

        assert!(elapsed.as_secs() < 1);
    }
}
