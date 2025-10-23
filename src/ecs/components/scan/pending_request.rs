//! Pending scan request component

use crate::{ecs::components::scan::ScanConfigComponent, hardware::pool::TaskRequirements};

/// Request to create a scan once compatible hardware is available
#[derive(Debug, Clone)]
pub struct PendingScanRequest {
    pub scan_config: ScanConfigComponent,
    pub scan_number: u64,
    pub requirements: TaskRequirements,
}

impl PendingScanRequest {
    pub fn new(
        scan_config: ScanConfigComponent,
        scan_number: u64,
        requirements: TaskRequirements,
    ) -> Self {
        Self {
            scan_config,
            scan_number,
            requirements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::ScanType;

    #[test]
    fn test_new_pending_request() {
        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 2.0e6, 2.0e6, 40.0, 1.0, 3);
        let requirements = TaskRequirements {
            frequency_hz: 88.0e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let request = PendingScanRequest::new(config.clone(), 1, requirements.clone());

        assert_eq!(request.scan_number, 1);
        assert_eq!(request.scan_config.freq_min, config.freq_min);
    }
}
