//! Scan tuner assignment component

use crate::hardware::pool::TunerId;

/// Component tracking which tuner is assigned to a scan
#[derive(Debug, Clone)]
pub struct ScanTunerComponent {
    pub assigned_tuner: Option<TunerId>,
}

impl ScanTunerComponent {
    pub fn new() -> Self {
        Self {
            assigned_tuner: None,
        }
    }

    pub fn assign(&mut self, tuner_id: TunerId) {
        self.assigned_tuner = Some(tuner_id);
    }

    pub fn clear(&mut self) {
        self.assigned_tuner = None;
    }

    pub fn is_assigned(&self) -> bool {
        self.assigned_tuner.is_some()
    }
}

impl Default for ScanTunerComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::DeviceId;

    #[test]
    fn test_new_component_unassigned() {
        let component = ScanTunerComponent::new();
        assert!(!component.is_assigned());
        assert_eq!(component.assigned_tuner, None);
    }

    #[test]
    fn test_assign_tuner() {
        let mut component = ScanTunerComponent::new();
        let device_id = DeviceId::from_serial("test", "dev1");
        let tuner_id = TunerId::new(device_id, 0);

        component.assign(tuner_id.clone());

        assert!(component.is_assigned());
        assert_eq!(component.assigned_tuner, Some(tuner_id));
    }

    #[test]
    fn test_clear_assignment() {
        let mut component = ScanTunerComponent::new();
        let device_id = DeviceId::from_serial("test", "dev1");
        let tuner_id = TunerId::new(device_id, 0);

        component.assign(tuner_id);
        component.clear();

        assert!(!component.is_assigned());
        assert_eq!(component.assigned_tuner, None);
    }
}
