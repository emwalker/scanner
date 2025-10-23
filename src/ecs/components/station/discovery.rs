//! Station discovery component

use std::time::Instant;

use crate::ecs::components::window::WindowId;

/// Component tracking how and when a station was discovered
#[derive(Debug, Clone)]
pub struct StationDiscoveryComponent {
    /// When the station was first discovered
    pub discovered_at: Instant,
    /// Window associated with the discovery
    pub window_id: WindowId,
}

impl StationDiscoveryComponent {
    /// Create a new discovery component
    pub fn new(window_id: WindowId) -> Self {
        Self {
            discovered_at: Instant::now(),
            window_id,
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
    use crate::ecs::TaskId;

    #[test]
    fn test_create_discovery() {
        let window_id = WindowId::new(TaskId::new("test-task"), 5);
        let discovery = StationDiscoveryComponent::new(window_id.clone());

        assert_eq!(discovery.window_id, window_id);
    }

    #[test]
    fn test_discovered_ago() {
        let window_id = WindowId::new(TaskId::new("test-task"), 0);

        let discovery = StationDiscoveryComponent::new(window_id);
        let elapsed = discovery.discovered_ago();

        assert!(elapsed.as_secs() < 1);
    }
}
