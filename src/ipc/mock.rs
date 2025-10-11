use super::traits;
use super::{ControlMessage, IQPacket};
use crate::core::types::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Mock control channel for testing
///
/// Stores sent messages in a buffer and allows injecting received messages
pub struct MockControlChannel {
    sent: Arc<Mutex<VecDeque<ControlMessage>>>,
    to_recv: Arc<Mutex<VecDeque<ControlMessage>>>,
}

impl MockControlChannel {
    pub fn new() -> Self {
        Self {
            sent: Arc::new(Mutex::new(VecDeque::new())),
            to_recv: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn inject_message(&self, msg: ControlMessage) {
        self.to_recv.lock().unwrap().push_back(msg);
    }

    pub fn sent_messages(&self) -> Vec<ControlMessage> {
        self.sent.lock().unwrap().iter().cloned().collect()
    }
}

impl Default for MockControlChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl traits::ControlChannel for MockControlChannel {
    fn send(&mut self, msg: &ControlMessage) -> Result<()> {
        self.sent.lock().unwrap().push_back(msg.clone());
        Ok(())
    }

    fn recv(&mut self) -> Result<ControlMessage> {
        self.to_recv.lock().unwrap().pop_front().ok_or_else(|| {
            crate::core::errors::ScannerError::IpcCommunicationError(
                "No messages to receive".to_string(),
            )
        })
    }

    fn try_recv(&mut self) -> Result<Option<ControlMessage>> {
        Ok(self.to_recv.lock().unwrap().pop_front())
    }
}

/// Mock data receiver for testing
pub struct MockDataReceiver {
    to_recv: Arc<Mutex<VecDeque<IQPacket>>>,
}

impl MockDataReceiver {
    pub fn new() -> Self {
        Self {
            to_recv: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn inject_packet(&self, packet: IQPacket) {
        self.to_recv.lock().unwrap().push_back(packet);
    }
}

impl Default for MockDataReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl traits::DataReceiver for MockDataReceiver {
    fn recv(&mut self) -> Result<IQPacket> {
        self.to_recv.lock().unwrap().pop_front().ok_or_else(|| {
            crate::core::errors::ScannerError::IpcCommunicationError(
                "No packets to receive".to_string(),
            )
        })
    }
}

/// Mock data sender for testing
pub struct MockDataSender {
    sent: Arc<Mutex<VecDeque<IQPacket>>>,
}

impl MockDataSender {
    pub fn new() -> Self {
        Self {
            sent: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn sent_packets(&self) -> Vec<IQPacket> {
        self.sent.lock().unwrap().iter().cloned().collect()
    }
}

impl Default for MockDataSender {
    fn default() -> Self {
        Self::new()
    }
}

impl traits::DataSender for MockDataSender {
    fn send(&mut self, packet: &IQPacket) -> Result<()> {
        self.sent.lock().unwrap().push_back(packet.clone());
        Ok(())
    }
}
