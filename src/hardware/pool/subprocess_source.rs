//! Rustradio source block for subprocess IPC data

use crate::ipc::{DataReceiver, UnixDataReceiver};
use rustradio::block::{Block, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Result};
use std::sync::{Arc, Mutex};

/// Rustradio source that reads I/Q samples from subprocess via IPC
pub struct SubprocessSource {
    dst: WriteStream<Complex>,
    data_receiver: Arc<Mutex<UnixDataReceiver>>,
    channel_index: usize,
}

impl SubprocessSource {
    /// Create new SubprocessSource
    ///
    /// Filters incoming IQPackets to only pass through samples from the specified channel.
    pub fn new(
        data_receiver: Arc<Mutex<UnixDataReceiver>>,
        channel_index: usize,
    ) -> (Self, ReadStream<Complex>) {
        let (dst, read_stream) = rustradio::stream::new_stream();
        (
            Self {
                dst,
                data_receiver,
                channel_index,
            },
            read_stream,
        )
    }
}

impl rustradio::block::BlockName for SubprocessSource {
    fn block_name(&self) -> &str {
        "SubprocessSource"
    }
}

impl rustradio::block::BlockEOF for SubprocessSource {
    fn eof(&mut self) -> bool {
        false
    }
}

impl Block for SubprocessSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let mut receiver = self
            .data_receiver
            .lock()
            .map_err(|e| rustradio::Error::msg(format!("Data receiver lock failed: {}", e)))?;

        match receiver.recv() {
            Ok(packet) => {
                if packet.channel != self.channel_index {
                    return Ok(BlockRet::Again);
                }

                let mut output = self.dst.write_buf()?;
                let n = output.len().min(packet.samples.len());

                if n == 0 {
                    return Ok(BlockRet::WaitForStream(&self.dst, packet.samples.len()));
                }

                output.slice()[..n].copy_from_slice(&packet.samples[..n]);
                output.produce(n, &[]);

                Ok(BlockRet::Again)
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("would block") || err_str.contains("timed out") {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    Ok(BlockRet::Again)
                } else {
                    Err(rustradio::Error::msg(format!("Data receiver error: {}", e)))
                }
            }
        }
    }
}
