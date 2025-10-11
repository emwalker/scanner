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
