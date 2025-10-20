use crate::hardware::DeviceTrait;
use crate::hardware::pool::SubprocessHandle;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HardwareConnectionState {
    Connected,
    Disconnected,
}

pub struct HardwareConnectionComponent {
    pub state: HardwareConnectionState,
    pub device: Option<Arc<Mutex<Box<dyn DeviceTrait>>>>,
    pub subprocess: Option<Arc<SubprocessHandle>>,
}

impl std::fmt::Debug for HardwareConnectionComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HardwareConnectionComponent")
            .field("state", &self.state)
            .field("device", &self.device.as_ref().map(|_| "Some(..)"))
            .field("subprocess", &self.subprocess.as_ref().map(|_| "Some(..)"))
            .finish()
    }
}

impl HardwareConnectionComponent {
    pub fn new_connected(device: Option<Arc<Mutex<Box<dyn DeviceTrait>>>>) -> Self {
        Self {
            state: HardwareConnectionState::Connected,
            device,
            subprocess: None,
        }
    }

    pub fn new_disconnected() -> Self {
        Self {
            state: HardwareConnectionState::Disconnected,
            device: None,
            subprocess: None,
        }
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, HardwareConnectionState::Connected)
    }

    pub fn disconnect(&mut self) {
        self.state = HardwareConnectionState::Disconnected;
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
        let mut conn = HardwareConnectionComponent::new_connected(None);
        assert!(conn.is_connected());
        assert!(conn.subprocess.is_none());

        conn.disconnect();
        assert!(!conn.is_connected());
        assert!(conn.device.is_none());
    }

    #[test]
    fn test_subprocess_attachment() {
        let conn = HardwareConnectionComponent::new_connected(None);
        assert!(conn.subprocess().is_none());
    }
}
