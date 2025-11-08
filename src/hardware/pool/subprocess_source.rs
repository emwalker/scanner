//! Rustradio source block for subprocess IPC data

use std::sync::{Arc, Mutex};

use rustradio::{
    Complex, Result,
    block::{Block, BlockRet},
    stream::{ReadStream, WriteStream},
};

use crate::{
    ecs::GlobalPauseResource,
    ipc::{DataReceiver, UnixDataReceiver},
};

/// Rustradio source that reads I/Q samples from subprocess via IPC
pub struct SubprocessSource<R = UnixDataReceiver>
where
    R: DataReceiver + Send + Sync,
{
    dst: WriteStream<Complex>,
    data_receiver: Arc<Mutex<R>>,
    channel_index: usize,
    global_pause_resource: Option<GlobalPauseResource>,
}

impl<R> SubprocessSource<R>
where
    R: DataReceiver + Send + Sync,
{
    /// Create new SubprocessSource
    ///
    /// Filters incoming IQPackets to only pass through samples from the specified channel.
    pub fn new(data_receiver: Arc<Mutex<R>>, channel_index: usize) -> (Self, ReadStream<Complex>) {
        let (dst, read_stream) = rustradio::stream::new_stream();
        (
            Self {
                dst,
                data_receiver,
                channel_index,
                global_pause_resource: None,
            },
            read_stream,
        )
    }

    /// Create new SubprocessSource with pause support
    ///
    /// When global_pause_resource is Some and paused, the source will yield
    /// without reading from the socket to prevent CPU usage during pause.
    pub fn with_pause_support(
        data_receiver: Arc<Mutex<R>>,
        channel_index: usize,
        global_pause_resource: GlobalPauseResource,
    ) -> (Self, ReadStream<Complex>) {
        let (dst, read_stream) = rustradio::stream::new_stream();
        (
            Self {
                dst,
                data_receiver,
                channel_index,
                global_pause_resource: Some(global_pause_resource),
            },
            read_stream,
        )
    }
}

impl<R> rustradio::block::BlockName for SubprocessSource<R>
where
    R: DataReceiver + Send + Sync,
{
    fn block_name(&self) -> &str {
        "SubprocessSource"
    }
}

impl<R> rustradio::block::BlockEOF for SubprocessSource<R>
where
    R: DataReceiver + Send + Sync,
{
    fn eof(&mut self) -> bool {
        false
    }
}

impl<R> Block for SubprocessSource<R>
where
    R: DataReceiver + Send + Sync,
{
    fn work(&mut self) -> Result<BlockRet<'_>> {
        // Check if globally paused - yield CPU without reading socket
        if let Some(ref pause_resource) = self.global_pause_resource
            && let Ok(state) = pause_resource.try_lock()
            && matches!(*state, crate::ecs::GlobalPauseState::Paused { .. })
        {
            // Yield CPU without consuming socket data during pause
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(BlockRet::Again);
        }

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

#[cfg(test)]
mod tests {
    use rustradio::Complex;

    use super::*;
    use crate::{
        ecs::GlobalPauseState,
        ipc::{IQPacket, MockDataReceiver},
    };

    #[test]
    fn test_subprocess_source_respects_global_pause() {
        // Create pause resource in paused state
        let pause_resource = Arc::new(Mutex::new(GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));

        // Create mock data receiver with a packet ready
        let packet = IQPacket {
            channel: 0,
            samples: vec![Complex::new(1.0, 0.5), Complex::new(-0.5, 1.0)],
            timestamp: 123456,
            sequence: 1,
        };

        let mock_receiver = MockDataReceiver::new();
        mock_receiver.add_packet(packet.clone());
        let data_receiver = Arc::new(Mutex::new(mock_receiver));

        // Create source with pause support
        let (mut source, _read_stream) =
            SubprocessSource::with_pause_support(data_receiver.clone(), 0, pause_resource.clone());

        // Call work() - should yield without consuming packet when paused
        let result = source.work().unwrap();
        assert!(matches!(result, BlockRet::Again));
        drop(result); // Drop to release borrow

        // Verify packet was NOT consumed (still available)
        let receiver = data_receiver.lock().unwrap();
        assert_eq!(
            receiver.pending_packets(),
            1,
            "Packet should not be consumed when paused"
        );

        // Unpause and verify packet gets consumed
        drop(receiver); // Release lock
        {
            let mut state = pause_resource.lock().unwrap();
            *state = GlobalPauseState::Active;
        }

        let result = source.work().unwrap();
        assert!(matches!(result, BlockRet::Again));
        drop(result); // Drop to release borrow

        // Verify packet was consumed when active
        let receiver = data_receiver.lock().unwrap();
        assert_eq!(
            receiver.pending_packets(),
            0,
            "Packet should be consumed when active"
        );
    }

    #[test]
    fn test_subprocess_source_without_pause_support_always_consumes() {
        // Create mock data receiver with a packet ready
        let packet = IQPacket {
            channel: 0,
            samples: vec![Complex::new(1.0, 0.5)],
            timestamp: 123456,
            sequence: 1,
        };

        let mock_receiver = MockDataReceiver::new();
        mock_receiver.add_packet(packet);
        let data_receiver = Arc::new(Mutex::new(mock_receiver));

        // Create source WITHOUT pause support
        let (mut source, _read_stream) = SubprocessSource::new(data_receiver.clone(), 0);

        // Call work() - should consume packet even if we were "paused"
        let result = source.work().unwrap();
        assert!(matches!(result, BlockRet::Again));
        drop(result); // Drop to release borrow

        // Verify packet was consumed (backward compatibility)
        let receiver = data_receiver.lock().unwrap();
        assert_eq!(
            receiver.pending_packets(),
            0,
            "Legacy behavior should always consume"
        );
    }
}
