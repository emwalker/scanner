use rustradio::Float;
use rustradio::stream::ReadStream;
use std::sync::Arc;
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

/// Rust Radio sink that pushes sample packets to an MPSC channel
pub struct MpscSink {
    src: ReadStream<Float>,
    sender: std::sync::mpsc::SyncSender<AudioPacket>,
    channel_name: String,
    packet_size: usize,
    buffer: Vec<f32>,
}

impl MpscSink {
    pub fn new(
        src: ReadStream<Float>,
        sender: std::sync::mpsc::SyncSender<AudioPacket>,
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

impl rustradio::block::BlockName for MpscSink {
    fn block_name(&self) -> &str {
        "MpscSink"
    }
}

impl rustradio::block::BlockEOF for MpscSink {
    fn eof(&mut self) -> bool {
        self.src.eof()
    }
}

impl rustradio::block::Block for MpscSink {
    fn work(&mut self) -> rustradio::Result<rustradio::block::BlockRet<'_>> {
        let (input_buf, _) = self.src.read_buf()?;
        let samples = input_buf.slice();

        if samples.is_empty() {
            return Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1));
        }

        static BACKPRESSURE_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        static TOTAL_SAMPLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let mut consumed = 0;
        let mut packets_sent = 0;

        while consumed < samples.len() {
            let space_in_buffer = self.packet_size - self.buffer.len();
            let to_copy = space_in_buffer.min(samples.len() - consumed);

            self.buffer
                .extend_from_slice(&samples[consumed..consumed + to_copy]);
            consumed += to_copy;

            if self.buffer.len() >= self.packet_size {
                let packet = AudioPacket::new(std::mem::replace(
                    &mut self.buffer,
                    Vec::with_capacity(self.packet_size),
                ));

                match self.sender.try_send(packet) {
                    Ok(_) => packets_sent += 1,
                    Err(std::sync::mpsc::TrySendError::Full(_)) => {
                        let bp_count = BACKPRESSURE_COUNTER
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            + 1;
                        debug!(
                            backpressure_event = bp_count,
                            channel_name = %self.channel_name,
                            "MPSC BACKPRESSURE: Audio output not consuming fast enough"
                        );
                        consumed = samples.len();
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        break;
                    }
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
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

        let total = TOTAL_SAMPLES.fetch_add(consumed as u64, std::sync::atomic::Ordering::Relaxed)
            + consumed as u64;
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
