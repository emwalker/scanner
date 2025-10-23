//! Device component - represents physical SDR hardware metadata

use crate::hardware;

/// Component containing device-level information for a tuner
///
/// This component describes the physical device that a tuner belongs to,
/// including its identity, capabilities, and connection status.
#[derive(Debug, Clone)]
pub struct DeviceComponent {
    /// The device this tuner belongs to
    pub device_id: hardware::DeviceId,

    /// Channel index within the device (0 for first tuner, 1 for second, etc.)
    pub channel_index: usize,

    /// Whether the device is currently connected
    pub connected: bool,

    /// Device capabilities (frequency range, sample rates, etc.)
    pub capabilities: hardware::Capabilities,

    /// Backend providing this device (Soapy, USB, etc.)
    pub backend: hardware::types::Backend,

    /// Antenna name (e.g., "Tuner 1 50 ohm", "Tuner 2 50 ohm")
    pub antenna: Option<String>,
}

impl DeviceComponent {
    /// Create a new device component
    pub fn new(
        device_id: hardware::DeviceId,
        channel_index: usize,
        capabilities: hardware::Capabilities,
        backend: hardware::types::Backend,
        antenna: Option<String>,
    ) -> Self {
        Self {
            device_id,
            channel_index,
            connected: true,
            capabilities,
            backend,
            antenna,
        }
    }

    /// Mark device as disconnected
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Mark device as connected
    pub fn reconnect(&mut self) {
        self.connected = true;
    }
}
