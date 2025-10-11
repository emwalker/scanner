use super::{ControlMessage, IQPacket};
use crate::core::types::Result;

/// Trait for bidirectional control message communication
pub trait ControlChannel {
    fn send(&mut self, msg: &ControlMessage) -> Result<()>;
    fn recv(&mut self) -> Result<ControlMessage>;
    fn try_recv(&mut self) -> Result<Option<ControlMessage>>;
}

/// Trait for receiving I/Q data packets
pub trait DataReceiver {
    fn recv(&mut self) -> Result<IQPacket>;
}

/// Trait for sending I/Q data packets
pub trait DataSender {
    fn send(&mut self, packet: &IQPacket) -> Result<()>;
}
