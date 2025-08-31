use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Result};
use std::sync::mpsc;

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
