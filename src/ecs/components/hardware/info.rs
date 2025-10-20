use crate::hardware::{Capabilities, DeviceId};

#[derive(Debug, Clone)]
pub struct HardwareInfoComponent {
    pub device_id: DeviceId,
    pub label: String,
    pub capabilities: Capabilities,
}

impl HardwareInfoComponent {
    pub fn new(device_id: DeviceId, label: String, capabilities: Capabilities) -> Self {
        Self {
            device_id,
            label,
            capabilities,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_info_creation() {
        let device_id = DeviceId::from_serial("sdrplay", "12345");
        let capabilities = Capabilities::for_device(&device_id);

        let info = HardwareInfoComponent::new(
            device_id.clone(),
            "Test Device".to_string(),
            capabilities.clone(),
        );

        assert_eq!(info.device_id, device_id);
        assert_eq!(info.label, "Test Device");
        assert_eq!(info.capabilities.channels, capabilities.channels);
    }
}
