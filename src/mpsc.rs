use rustradio::Float;
use rustradio::stream::ReadStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use tracing::debug;

/// A batch of audio samples transmitted as a single unit
#[derive(Clone, Debug)]
pub struct AudioPacket {
    samples: Arc<Vec<f32>>,
}

impl AudioPacket {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples: Arc::new(samples),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.samples
    }
}

/// Trait for types that can send audio packets
pub trait AudioSink: Send {
    fn send(&self, packet: AudioPacket) -> std::result::Result<(), TrySendError<AudioPacket>>;
}

impl AudioSink for SyncSender<AudioPacket> {
    fn send(&self, packet: AudioPacket) -> std::result::Result<(), TrySendError<AudioPacket>> {
        self.try_send(packet)
    }
}

/// Rust Radio sink that pushes sample packets to an MPSC channel
pub struct MpscSink<A: AudioSink> {
    src: ReadStream<Float>,
    sender: A,
    channel_name: String,
    packet_size: usize,
    buffer: Vec<f32>,
}

impl<A: AudioSink> MpscSink<A> {
    pub fn new(
        src: ReadStream<Float>,
        sender: A,
        channel_name: String,
        packet_size: usize,
    ) -> Self {
        MpscSink {
            src,
            sender,
            channel_name,
            packet_size,
            buffer: Vec::with_capacity(packet_size),
        }
    }
}

impl<A: AudioSink> rustradio::block::BlockName for MpscSink<A> {
    fn block_name(&self) -> &str {
        "MpscSink"
    }
}

impl<A: AudioSink> rustradio::block::BlockEOF for MpscSink<A> {
    fn eof(&mut self) -> bool {
        self.src.eof()
    }
}

impl<A: AudioSink> rustradio::block::Block for MpscSink<A> {
    fn work(&mut self) -> rustradio::Result<rustradio::block::BlockRet<'_>> {
        let (input_buf, _) = self.src.read_buf()?;
        let samples = input_buf.slice();

        if samples.is_empty() {
            return Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1));
        }

        static BACKPRESSURE_COUNTER: AtomicUsize = AtomicUsize::new(0);
        static TOTAL_SAMPLES: AtomicU64 = AtomicU64::new(0);

        let mut consumed = 0;
        let mut packets_sent = 0;

        while consumed < samples.len() {
            let space_in_buffer = self.packet_size - self.buffer.len();
            let to_copy = space_in_buffer.min(samples.len() - consumed);

            self.buffer
                .extend_from_slice(&samples[consumed..consumed + to_copy]);
            consumed += to_copy;

            if self.buffer.len() >= self.packet_size {
                let buffer = std::mem::take(&mut self.buffer);
                self.buffer.reserve(self.packet_size);
                let packet = AudioPacket::new(buffer);

                match self.sender.send(packet) {
                    Ok(_) => packets_sent += 1,
                    Err(TrySendError::Full(_)) => {
                        let bp_count = BACKPRESSURE_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
                        debug!(
                            backpressure_event = bp_count,
                            channel_name = %self.channel_name,
                            "MPSC BACKPRESSURE: Audio output not consuming fast enough"
                        );
                        consumed = samples.len();
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        break;
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        debug!(
                            channel_name = %self.channel_name,
                            "MPSC channel disconnected"
                        );
                        consumed = samples.len();
                        break;
                    }
                }
            }
        }

        let total = TOTAL_SAMPLES.fetch_add(consumed as u64, Ordering::Relaxed) + consumed as u64;
        if packets_sent > 0 && packets_sent % 100 == 0 {
            debug!(
                packets_sent = packets_sent,
                total_samples_sent = total,
                channel_name = %self.channel_name,
                "MpscSink: sent packets"
            );
        }

        input_buf.consume(consumed);
        Ok(rustradio::block::BlockRet::Again)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustradio::block::{Block, BlockEOF};
    use rustradio::stream::WriteStream;
    use std::sync::{Arc, Mutex};

    #[derive(Default, Clone)]
    struct MockAudioSink {
        packets: Arc<Mutex<Vec<AudioPacket>>>,
    }

    impl MockAudioSink {
        fn packets(&self) -> Vec<AudioPacket> {
            self.packets.lock().unwrap().clone()
        }

        fn packet_count(&self) -> usize {
            self.packets.lock().unwrap().len()
        }

        fn total_samples(&self) -> usize {
            self.packets
                .lock()
                .unwrap()
                .iter()
                .map(|p| p.as_slice().len())
                .sum()
        }
    }

    impl AudioSink for MockAudioSink {
        fn send(&self, packet: AudioPacket) -> std::result::Result<(), TrySendError<AudioPacket>> {
            self.packets.lock().unwrap().push(packet);
            Ok(())
        }
    }

    #[test]
    fn test_mpsc_sink_batches_samples_into_packets() {
        let (input, read_stream) = WriteStream::new();
        let mock_sink = MockAudioSink::default();
        let packet_size = 1000;

        let mut sink = MpscSink::new(
            read_stream,
            mock_sink.clone(),
            "test".to_string(),
            packet_size,
        );

        let samples: Vec<f32> = (0..2500).map(|i| i as f32 * 0.001).collect();
        input.write_buf().unwrap().slice()[..2500].copy_from_slice(&samples);
        input.write_buf().unwrap().produce(2500, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 2);
        assert_eq!(mock_sink.total_samples(), 2000);

        let packets = mock_sink.packets();
        assert_eq!(packets[0].as_slice().len(), 1000);
        assert_eq!(packets[1].as_slice().len(), 1000);
        assert_eq!(packets[0].as_slice()[0], 0.0);
        assert_eq!(packets[1].as_slice()[0], 1.0);
    }

    #[test]
    fn test_mpsc_sink_buffers_partial_packet() {
        let (input, read_stream) = WriteStream::new();
        let mock_sink = MockAudioSink::default();
        let packet_size = 1000;

        let mut sink = MpscSink::new(
            read_stream,
            mock_sink.clone(),
            "test".to_string(),
            packet_size,
        );

        let samples1: Vec<f32> = (0..1500).map(|i| i as f32).collect();
        input.write_buf().unwrap().slice()[..1500].copy_from_slice(&samples1);
        input.write_buf().unwrap().produce(1500, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 1);

        let samples2: Vec<f32> = (1500..2000).map(|i| i as f32).collect();
        input.write_buf().unwrap().slice()[..500].copy_from_slice(&samples2);
        input.write_buf().unwrap().produce(500, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 2);
        let packets = mock_sink.packets();
        assert_eq!(packets[1].as_slice()[0], 1000.0);
        assert_eq!(packets[1].as_slice()[999], 1999.0);
    }

    #[test]
    fn test_mpsc_sink_handles_backpressure() {
        struct FullAudioSink;

        impl AudioSink for FullAudioSink {
            fn send(
                &self,
                packet: AudioPacket,
            ) -> std::result::Result<(), TrySendError<AudioPacket>> {
                Err(TrySendError::Full(packet))
            }
        }

        let (mut input, read_stream) = WriteStream::new();
        let full_sink = FullAudioSink;
        let packet_size = 100;

        let mut sink = MpscSink::new(read_stream, full_sink, "test".to_string(), packet_size);

        let samples: Vec<f32> = (0..200).map(|i| i as f32).collect();
        input.write_buf().unwrap().slice()[..200].copy_from_slice(&samples);
        input.write_buf().unwrap().produce(200, &[]);

        sink.work().unwrap();
    }

    #[test]
    fn test_mpsc_sink_eof_propagation() {
        let (_input, read_stream) = WriteStream::new();
        let mock_sink = MockAudioSink::default();
        let mut sink = MpscSink::new(read_stream, mock_sink, "test".to_string(), 100);

        assert!(!sink.eof());
    }
}
