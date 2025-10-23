use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::{ControlMessage, IQPacket, traits};
use crate::core::types::Result;

/// Mock control channel for testing
///
/// Stores sent messages in a buffer and allows injecting received messages
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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

#[cfg(test)]
mod tests {
    use num::Complex;

    use super::*;
    use crate::ipc::traits::{ControlChannel, DataReceiver, DataSender};

    #[test]
    fn test_mock_control_channel_send_recv() {
        let mut channel = MockControlChannel::new();

        let msg = ControlMessage::ConfigureAndStart {
            channel: 0,
            freq_hz: 88.9e6,
            gain_db: 24.0,
            sample_rate: 2_000_000.0,
        };

        channel.send(&msg).unwrap();

        let sent = channel.sent_messages();
        assert_eq!(sent.len(), 1);
    }

    #[test]
    fn test_mock_control_channel_inject_and_recv() {
        let channel = MockControlChannel::new();
        let mut channel_mut = channel.clone();

        let msg = ControlMessage::Ready {
            device_id: "test-device".to_string(),
            channels: 2,
        };

        channel.inject_message(msg.clone());

        let received = channel_mut.recv().unwrap();
        match received {
            ControlMessage::Ready {
                device_id,
                channels,
            } => {
                assert_eq!(device_id, "test-device");
                assert_eq!(channels, 2);
            }
            _ => panic!("Wrong message variant"),
        }
    }

    #[test]
    fn test_mock_control_channel_try_recv_empty() {
        let mut channel = MockControlChannel::new();

        let result = channel.try_recv().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_mock_control_channel_try_recv_with_message() {
        let channel = MockControlChannel::new();
        let mut channel_mut = channel.clone();

        channel.inject_message(ControlMessage::Shutdown);

        let result = channel_mut.try_recv().unwrap();
        assert!(result.is_some());
        match result.unwrap() {
            ControlMessage::Shutdown => {}
            _ => panic!("Wrong message variant"),
        }
    }

    #[test]
    fn test_mock_control_channel_recv_error_when_empty() {
        let mut channel = MockControlChannel::new();

        let result = channel.recv();
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_data_receiver_inject_and_recv() {
        let receiver = MockDataReceiver::new();
        let mut receiver_mut = receiver.clone();

        let packet = IQPacket {
            channel: 0,
            samples: vec![Complex::new(1.0, 0.5), Complex::new(-0.5, 1.0)],
            timestamp: 123456789,
            sequence: 42,
        };

        receiver.inject_packet(packet.clone());

        let received = receiver_mut.recv().unwrap();
        assert_eq!(received.channel, 0);
        assert_eq!(received.samples.len(), 2);
        assert_eq!(received.sequence, 42);
    }

    #[test]
    fn test_mock_data_receiver_error_when_empty() {
        let mut receiver = MockDataReceiver::new();

        let result = receiver.recv();
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_data_sender_send_and_inspect() {
        let sender = MockDataSender::new();
        let mut sender_mut = sender.clone();

        let packet1 = IQPacket {
            channel: 0,
            samples: vec![Complex::new(1.0, 0.0)],
            timestamp: 100,
            sequence: 1,
        };

        let packet2 = IQPacket {
            channel: 1,
            samples: vec![Complex::new(0.0, 1.0)],
            timestamp: 200,
            sequence: 2,
        };

        sender_mut.send(&packet1).unwrap();
        sender_mut.send(&packet2).unwrap();

        let sent = sender.sent_packets();
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].channel, 0);
        assert_eq!(sent[0].sequence, 1);
        assert_eq!(sent[1].channel, 1);
        assert_eq!(sent[1].sequence, 2);
    }

    #[test]
    fn test_mock_channel_cloning() {
        let channel = MockControlChannel::new();
        let channel2 = channel.clone();

        channel.inject_message(ControlMessage::Shutdown);

        let sent1 = channel.sent_messages();
        let sent2 = channel2.sent_messages();

        assert_eq!(sent1.len(), sent2.len());
    }
}
