use super::traits;
use crate::core::types::Result;
use num::Complex;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

/// Unix socket implementation of control channel
///
/// Uses RAII to ensure proper cleanup of resources
pub struct UnixControlChannel {
    stream: UnixStream,
    socket_path: Option<PathBuf>,
}

impl UnixControlChannel {
    pub fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            socket_path: None,
        }
    }

    pub fn with_cleanup(stream: UnixStream, socket_path: PathBuf) -> Self {
        Self {
            stream,
            socket_path: Some(socket_path),
        }
    }
}

impl traits::ControlChannel for UnixControlChannel {
    fn send(&mut self, msg: &ControlMessage) -> Result<()> {
        send_control_message(&mut self.stream, msg)
    }

    fn recv(&mut self) -> Result<ControlMessage> {
        recv_control_message(&mut self.stream)
    }

    fn try_recv(&mut self) -> Result<Option<ControlMessage>> {
        try_recv_control_message(&mut self.stream)
    }
}

impl Drop for UnixControlChannel {
    fn drop(&mut self) {
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Unix socket implementation for receiving I/Q data packets
///
/// Used by main process to receive samples from worker subprocess
/// Uses RAII to ensure proper cleanup of resources
pub struct UnixDataReceiver {
    stream: UnixStream,
    socket_path: Option<PathBuf>,
}

impl UnixDataReceiver {
    pub fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            socket_path: None,
        }
    }

    pub fn with_cleanup(stream: UnixStream, socket_path: PathBuf) -> Self {
        Self {
            stream,
            socket_path: Some(socket_path),
        }
    }
}

impl traits::DataReceiver for UnixDataReceiver {
    fn recv(&mut self) -> Result<IQPacket> {
        recv_iq_packet(&mut self.stream)
    }
}

impl Drop for UnixDataReceiver {
    fn drop(&mut self) {
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Unix socket implementation for sending I/Q data packets
///
/// Used by worker subprocess to send samples to main process
/// Uses RAII to ensure proper cleanup of resources
pub struct UnixDataSender {
    stream: UnixStream,
    socket_path: Option<PathBuf>,
}

impl UnixDataSender {
    pub fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            socket_path: None,
        }
    }

    pub fn with_cleanup(stream: UnixStream, socket_path: PathBuf) -> Self {
        Self {
            stream,
            socket_path: Some(socket_path),
        }
    }
}

impl traits::DataSender for UnixDataSender {
    fn send(&mut self, packet: &IQPacket) -> Result<()> {
        send_iq_packet(&mut self.stream, packet)
    }
}

impl Drop for UnixDataSender {
    fn drop(&mut self) {
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ControlMessage {
    // Main → Worker commands
    ConfigureAndStart {
        channel: usize,
        freq_hz: f64,
        gain_db: f64,
        sample_rate: f64,
    },
    StopStream {
        channel: usize,
    },
    Shutdown,

    // Worker → Main responses
    Ready {
        device_id: String,
        channels: usize,
    },
    StreamStarted {
        channel: usize,
        actual_freq: f64,
        actual_gain: f64,
        actual_sample_rate: f64,
    },
    StreamStopped {
        channel: usize,
    },
    ShutdownAck,
    Error {
        channel: Option<usize>,
        message: String,
    },
    DeviceList {
        devices: Vec<crate::hardware::DeviceInfo>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IQPacket {
    pub channel: usize,
    pub samples: Vec<Complex<f32>>,
    pub timestamp: u64,
    pub sequence: u64,
}

#[allow(dead_code)]
fn send_control_message(stream: &mut UnixStream, msg: &ControlMessage) -> Result<()> {
    let bytes = postcard::to_allocvec(msg)?;
    let len = bytes.len() as u32;

    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()?;

    Ok(())
}

#[allow(dead_code)]
fn recv_control_message(stream: &mut UnixStream) -> Result<ControlMessage> {
    // Ensure socket is in blocking mode (in case try_recv left it non-blocking)
    stream.set_nonblocking(false)?;

    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;

    let msg = postcard::from_bytes(&buf)?;
    Ok(msg)
}

#[allow(dead_code)]
fn try_recv_control_message(stream: &mut UnixStream) -> Result<Option<ControlMessage>> {
    stream.set_nonblocking(true)?;

    let mut len_bytes = [0u8; 4];
    match stream.read_exact(&mut len_bytes) {
        Ok(()) => {
            let len = u32::from_le_bytes(len_bytes) as usize;

            let mut buf = vec![0u8; len];
            stream.read_exact(&mut buf)?;

            stream.set_nonblocking(false)?;

            let msg = postcard::from_bytes(&buf)?;
            Ok(Some(msg))
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            stream.set_nonblocking(false)?;
            Ok(None)
        }
        Err(e) => {
            stream.set_nonblocking(false)?;
            Err(e.into())
        }
    }
}

#[allow(dead_code)]
fn send_iq_packet(stream: &mut UnixStream, packet: &IQPacket) -> Result<()> {
    let bytes = postcard::to_allocvec(packet)?;
    let len = bytes.len() as u32;

    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()?;

    Ok(())
}

#[allow(dead_code)]
fn recv_iq_packet(stream: &mut UnixStream) -> Result<IQPacket> {
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;

    let packet = postcard::from_bytes(&buf)?;
    Ok(packet)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_message_serialization() {
        let msg = ControlMessage::ConfigureAndStart {
            channel: 0,
            freq_hz: 88.9e6,
            gain_db: 24.0,
            sample_rate: 2_000_000.0,
        };

        let bytes = postcard::to_allocvec(&msg).unwrap();
        let deserialized: ControlMessage = postcard::from_bytes(&bytes).unwrap();

        match deserialized {
            ControlMessage::ConfigureAndStart {
                channel,
                freq_hz,
                gain_db,
                sample_rate,
            } => {
                assert_eq!(channel, 0);
                assert_eq!(freq_hz, 88.9e6);
                assert_eq!(gain_db, 24.0);
                assert_eq!(sample_rate, 2_000_000.0);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_iq_packet_serialization() {
        let packet = IQPacket {
            channel: 0,
            samples: vec![
                Complex::new(1.0, 0.5),
                Complex::new(-0.5, 1.0),
                Complex::new(0.0, -1.0),
            ],
            timestamp: 123456789,
            sequence: 42,
        };

        let bytes = postcard::to_allocvec(&packet).unwrap();
        let deserialized: IQPacket = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.channel, 0);
        assert_eq!(deserialized.samples.len(), 3);
        assert_eq!(deserialized.samples[0], Complex::new(1.0, 0.5));
        assert_eq!(deserialized.samples[1], Complex::new(-0.5, 1.0));
        assert_eq!(deserialized.samples[2], Complex::new(0.0, -1.0));
        assert_eq!(deserialized.timestamp, 123456789);
        assert_eq!(deserialized.sequence, 42);
    }

    #[test]
    fn test_all_control_message_variants() {
        let messages = vec![
            ControlMessage::ConfigureAndStart {
                channel: 0,
                freq_hz: 100.0e6,
                gain_db: 20.0,
                sample_rate: 2.4e6,
            },
            ControlMessage::StopStream { channel: 1 },
            ControlMessage::Shutdown,
            ControlMessage::Ready {
                device_id: "test-device".to_string(),
                channels: 2,
            },
            ControlMessage::StreamStarted {
                channel: 0,
                actual_freq: 100.05e6,
                actual_gain: 19.8,
                actual_sample_rate: 2.4e6,
            },
            ControlMessage::StreamStopped { channel: 1 },
            ControlMessage::ShutdownAck,
            ControlMessage::Error {
                channel: Some(0),
                message: "Test error".to_string(),
            },
        ];

        for msg in messages {
            let bytes = postcard::to_allocvec(&msg).unwrap();
            let _deserialized: ControlMessage = postcard::from_bytes(&bytes).unwrap();
        }
    }

    #[test]
    fn test_iq_packet_with_many_samples() {
        let samples: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new(i as f32 / 1024.0, (1024 - i) as f32 / 1024.0))
            .collect();

        let packet = IQPacket {
            channel: 0,
            samples: samples.clone(),
            timestamp: 987654321,
            sequence: 100,
        };

        let bytes = postcard::to_allocvec(&packet).unwrap();
        let deserialized: IQPacket = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.samples.len(), 1024);
        assert_eq!(deserialized.samples[0], samples[0]);
        assert_eq!(deserialized.samples[1023], samples[1023]);
    }

    #[test]
    fn test_error_message_with_none_channel() {
        let msg = ControlMessage::Error {
            channel: None,
            message: "Device-level error".to_string(),
        };

        let bytes = postcard::to_allocvec(&msg).unwrap();
        let deserialized: ControlMessage = postcard::from_bytes(&bytes).unwrap();

        match deserialized {
            ControlMessage::Error { channel, message } => {
                assert_eq!(channel, None);
                assert_eq!(message, "Device-level error");
            }
            _ => panic!("Wrong variant"),
        }
    }
}
