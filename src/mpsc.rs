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

        static BACKPRESSURE_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static TOTAL_SAMPLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let mut consumed = 0;
        let mut backpressure_occurred = false;
        for &sample in samples {
            match self.sender.try_send(sample) {
                Ok(_) => consumed += 1,
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    backpressure_occurred = true;
                    if consumed == 0 {
                        let bp_count = BACKPRESSURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        debug!(
                            backpressure_event = bp_count,
                            channel_name = %self.channel_name,
                            "MPSC BACKPRESSURE: Audio output not consuming fast enough"
                        );
                    }
                    break;
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    debug!(
                        channel_name = %self.channel_name,
                        "MPSC channel disconnected"
                    );
                    break;
                }
            }
        }

        let total = TOTAL_SAMPLES.fetch_add(consumed as u64, std::sync::atomic::Ordering::Relaxed) + consumed as u64;
        if backpressure_occurred && total % 100000 == 0 {
            debug!(
                total_samples_sent = total,
                channel_name = %self.channel_name,
                "MpscSink sample count"
            );
        }

        input_buf.consume(consumed);

        if consumed > 0 {
            Ok(rustradio::block::BlockRet::Again)
        } else {
            Ok(rustradio::block::BlockRet::WaitForStream(&self.src, 1))
        }
    }
}
