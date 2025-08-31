use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::ReadStream;
use rustradio::{Complex, Result};
use std::sync::mpsc;

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
