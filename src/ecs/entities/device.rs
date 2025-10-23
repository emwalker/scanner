use std::sync::{Arc, Mutex};

use crate::{
    ecs::{DeviceConnectionComponent, DeviceInfoComponent, DeviceLifecycleComponent, Entity},
    hardware::{Capabilities, DeviceId, DeviceTrait, types::Backend},
};

/// Entity representing a physical SDR hardware device
///
/// Unlike TunerEntity which represents individual tuner channels,
/// DeviceEntity represents the device itself (which may have multiple tuners).
pub struct DeviceEntity {
    id: DeviceId,
    pub info: DeviceInfoComponent,
    pub connection: DeviceConnectionComponent,
    pub lifecycle: DeviceLifecycleComponent,
}

impl DeviceEntity {
    /// Create a new device entity for a connected device
    pub fn new(
        device_id: DeviceId,
        label: String,
        capabilities: Capabilities,
        backend: Backend,
        device: Option<Box<dyn DeviceTrait>>,
    ) -> Self {
        let num_tuners = capabilities.channels;
        let device_handle = device.map(|d| Arc::new(Mutex::new(d)));

        Self {
            id: device_id.clone(),
            info: DeviceInfoComponent::new(device_id, label, capabilities),
            connection: DeviceConnectionComponent::new_connected(device_handle),
            lifecycle: DeviceLifecycleComponent::new(backend, num_tuners),
        }
    }

    /// Create a device entity without device handle (metadata only)
    pub fn new_metadata_only(
        device_id: DeviceId,
        label: String,
        capabilities: Capabilities,
        backend: Backend,
    ) -> Self {
        let num_tuners = capabilities.channels;

        Self {
            id: device_id.clone(),
            info: DeviceInfoComponent::new(device_id, label, capabilities),
            connection: DeviceConnectionComponent::new_connected(None),
            lifecycle: DeviceLifecycleComponent::new(backend, num_tuners),
        }
    }

    /// Check if device is connected
    pub fn is_connected(&self) -> bool {
        self.connection.is_connected()
    }

    /// Get number of tuners this device has
    pub fn num_tuners(&self) -> usize {
        self.lifecycle.num_tuners
    }

    /// Get device capabilities
    pub fn capabilities(&self) -> &Capabilities {
        &self.info.capabilities
    }

    /// Get backend
    pub fn backend(&self) -> &Backend {
        &self.lifecycle.backend
    }
}

impl Entity for DeviceEntity {
    type Id = DeviceId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

impl std::fmt::Debug for DeviceEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEntity")
            .field("id", &self.id)
            .field("info", &self.info)
            .field("lifecycle", &self.lifecycle)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_entity_creation() {
        let device_id = DeviceId::from_serial("sdrplay", "12345");
        let capabilities = Capabilities::for_device(&device_id);

        let entity = DeviceEntity::new_metadata_only(
            device_id.clone(),
            "Test Device".to_string(),
            capabilities.clone(),
            Backend::Mock,
        );

        assert_eq!(*entity.id(), device_id);
        assert_eq!(entity.info.label, "Test Device");
        assert_eq!(entity.num_tuners(), capabilities.channels);
        assert_eq!(*entity.backend(), Backend::Mock);
        assert!(entity.is_connected());
    }

    #[test]
    fn test_hardware_entity_queries() {
        let device_id = DeviceId::from_serial("sdrplay", "67890");
        let capabilities = Capabilities::for_device(&device_id);

        let entity = DeviceEntity::new_metadata_only(
            device_id,
            "Another Device".to_string(),
            capabilities.clone(),
            Backend::Soapy,
        );

        assert_eq!(entity.capabilities().channels, capabilities.channels);
        assert!(!entity.capabilities().rx_sample_rate_ranges.is_empty());
    }
}
