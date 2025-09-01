use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Float, Result};
use std::sync::mpsc;
use tracing::debug;

/// Rust Radio sink that pushes samples to an MPSC channel
pub struct MpscSink {
    src: ReadStream<Float>,
    sender: std::sync::mpsc::SyncSender<f32>,
    channel_name: String,
}

impl MpscSink {
    pub fn new(
        src: ReadStream<Float>,
        sender: std::sync::mpsc::SyncSender<f32>,
        channel_name: String,
    ) -> Self {
        MpscSink {
            src,
            sender,
            channel_name,
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

        // Send samples to MPSC channel
        let mut consumed = 0;
        for &sample in samples {
            if self.sender.send(sample).is_err() {
                // Channel is full - this provides backpressure to the entire graph
                debug!(
                    "MPSC channel full for {}, backpressuring graph",
                    self.channel_name
                );
                break;
            }
            consumed += 1;
        }

        input_buf.consume(consumed);

        if consumed > 0 {
            Ok(rustradio::block::BlockRet::Again)
        } else {
            Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1))
        }
    }
}

pub struct MpscSenderSink {
    input: ReadStream<Complex>,
    sender: mpsc::SyncSender<Complex>,
}

impl MpscSenderSink {
    pub fn new(input: ReadStream<Complex>, sender: mpsc::SyncSender<Complex>) -> Self {
        MpscSenderSink { input, sender }
    }
}

impl BlockName for MpscSenderSink {
    fn block_name(&self) -> &str {
        "MpscSenderSink"
    }
}

impl BlockEOF for MpscSenderSink {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for MpscSenderSink {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let samples = input_buf.slice();

        let mut consumed = 0;
        for &sample in samples {
            if self.sender.send(sample).is_err() {
                // Receiver disconnected, so we should stop
                return Ok(BlockRet::EOF);
            }
            consumed += 1;
        }

        input_buf.consume(consumed);

        if consumed > 0 {
            Ok(BlockRet::Again)
        } else {
            Ok(BlockRet::WaitForStream(&self.input, 1))
        }
    }
}

pub struct MpscReceiverSource {
    output: WriteStream<Complex>,
    receiver: mpsc::Receiver<Complex>,
}

impl MpscReceiverSource {
    pub fn new(receiver: mpsc::Receiver<Complex>) -> (Self, ReadStream<Complex>) {
        let (output, output_read_stream) = WriteStream::new();
        (Self { output, receiver }, output_read_stream)
    }
}

impl BlockName for MpscReceiverSource {
    fn block_name(&self) -> &str {
        "MpscReceiverSource"
    }
}

impl BlockEOF for MpscReceiverSource {
    fn eof(&mut self) -> bool {
        // This block never ends on its own, it relies on the sender disconnecting
        false
    }
}

impl Block for MpscReceiverSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let mut output_buf = self.output.write_buf()?;
        let mut produced = 0;

        // Try to receive as many samples as the output buffer can hold
        for i in 0..output_buf.len() {
            match self.receiver.try_recv() {
                Ok(sample) => {
                    output_buf.slice()[i] = sample;
                    produced += 1;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // No more samples available right now
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Sender disconnected, this source should now end
                    return Ok(BlockRet::EOF);
                }
            }
        }

        output_buf.produce(produced, &[]);

        if produced == 0 {
            // If no samples were produced, wait for more
            Ok(BlockRet::Again)
        } else {
            Ok(BlockRet::Again)
        }
    }
}
