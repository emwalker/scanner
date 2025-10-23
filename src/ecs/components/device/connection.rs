use std::sync::{Arc, Mutex};

use crate::hardware::{DeviceTrait, pool::SubprocessHandle};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceConnectionState {
    Connected,
    Disconnected,
}

pub struct DeviceConnectionComponent {
    pub state: DeviceConnectionState,
    pub device: Option<Arc<Mutex<Box<dyn DeviceTrait>>>>,
    pub subprocess: Option<Arc<SubprocessHandle>>,
}

impl std::fmt::Debug for DeviceConnectionComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceConnectionComponent")
            .field("state", &self.state)
            .field("device", &self.device.as_ref().map(|_| "Some(..)"))
            .field("subprocess", &self.subprocess.as_ref().map(|_| "Some(..)"))
            .finish()
    }
}

impl DeviceConnectionComponent {
    pub fn new_connected(device: Option<Arc<Mutex<Box<dyn DeviceTrait>>>>) -> Self {
        Self {
            state: DeviceConnectionState::Connected,
            device,
            subprocess: None,
        }
    }

    pub fn new_disconnected() -> Self {
        Self {
            state: DeviceConnectionState::Disconnected,
            device: None,
            subprocess: None,
        }
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, DeviceConnectionState::Connected)
    }

    pub fn disconnect(&mut self) {
        self.state = DeviceConnectionState::Disconnected;
        self.device = None;
        self.subprocess = None;
    }

    pub fn attach_subprocess(&mut self, subprocess: Arc<SubprocessHandle>) {
        self.subprocess = Some(subprocess);
    }

    pub fn subprocess(&self) -> Option<Arc<SubprocessHandle>> {
        self.subprocess.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_lifecycle() {
        let mut conn = DeviceConnectionComponent::new_connected(None);
        assert!(conn.is_connected());
        assert!(conn.subprocess.is_none());

        conn.disconnect();
        assert!(!conn.is_connected());
        assert!(conn.device.is_none());
    }

    #[test]
    fn test_subprocess_attachment() {
        let conn = DeviceConnectionComponent::new_connected(None);
        assert!(conn.subprocess().is_none());
    }
}
