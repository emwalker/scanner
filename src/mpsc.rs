use rustradio::Float;
use rustradio::stream::ReadStream;
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

        // Send samples to MPSC channel with try_send for better performance
        let mut consumed = 0;
        for &sample in samples {
            match self.sender.try_send(sample) {
                Ok(_) => consumed += 1,
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    // Channel is full - stop sending to provide backpressure
                    if consumed == 0 {
                        debug!(
                            "MPSC channel full for {}, backpressuring graph",
                            self.channel_name
                        );
                    }
                    break;
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    // Channel disconnected - stop processing
                    break;
                }
            }
        }

        input_buf.consume(consumed);

        if consumed > 0 {
            Ok(rustradio::block::BlockRet::Again)
        } else {
            Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1))
        }
    }
}
