use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Result};
use tokio::sync::broadcast;
use tracing::debug;

pub struct BroadcastSink {
    input: ReadStream<Complex>,
    sender: broadcast::Sender<Complex>,
}

impl BroadcastSink {
    pub fn new(input: ReadStream<Complex>, sender: broadcast::Sender<Complex>) -> Self {
        Self { input, sender }
    }
}

impl BlockName for BroadcastSink {
    fn block_name(&self) -> &str {
        "BroadcastSink"
    }
}

impl BlockEOF for BroadcastSink {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for BroadcastSink {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();
        if samples.is_empty() {
            return Ok(BlockRet::Again);
        }
        let mut sent = 0;
        for sample in samples {
            match self.sender.send(*sample) {
                Ok(_) => sent += 1,
                Err(_) => {
                    // No receivers or channel full - consume all samples to avoid blocking
                    sent = samples.len();
                    debug!("broadcast channel issue (no receivers or full), consuming all samples");
                    // Sleep to avoid spinning when no receivers are available
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    break;
                }
            }
        }

        if sent > 0 && sent % 1000 == 0 {
            debug!("BroadcastSink: sent {} samples", sent);
        }
        input_buf.consume(sent);
        Ok(BlockRet::Again)
    }
}

pub struct BroadcastSource {
    output: WriteStream<Complex>,
    receiver: broadcast::Receiver<Complex>,
}

impl BroadcastSource {
    pub fn new(receiver: broadcast::Receiver<Complex>) -> (Self, ReadStream<Complex>) {
        let (output, read_stream) = WriteStream::new();
        (Self { output, receiver }, read_stream)
    }
}

impl BlockName for BroadcastSource {
    fn block_name(&self) -> &str {
        "BroadcastSource"
    }
}

impl BlockEOF for BroadcastSource {
    fn eof(&mut self) -> bool {
        false
    }
}

impl Drop for BroadcastSource {
    fn drop(&mut self) {
        static DROP_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let drop_count = DROP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        debug!(
            drop_count = drop_count,
            receiver_lagged = self.receiver.len(),
            "BroadcastSource dropped - receiver being cleaned up"
        );
    }
}

impl Block for BroadcastSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        static WORK_CALL_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let count = WORK_CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count.is_multiple_of(10000) {
            debug!(count, "BroadcastSource work() called");
        }

        let mut out = self.output.write_buf()?;
        if out.is_empty() {
            debug!("BroadcastSource: output buffer empty, waiting");
            return Ok(BlockRet::WaitForStream(&self.output, 1));
        }

        let mut n = 0;

        // Aggressively drain the broadcast channel to prevent lag
        // Process entire output buffer to maximize throughput
        let batch_size = out.len();
        let mut samples_received = 0;

        // Hot loop - optimize for speed
        let out_slice = out.slice();
        while n < batch_size {
            match self.receiver.try_recv() {
                Ok(sample) => {
                    out_slice[n] = sample;
                    n += 1;
                    samples_received += 1;
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    break;
                }
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    static LAG_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    static TOTAL_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let lag_count = LAG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let total_skipped = TOTAL_SKIPPED.fetch_add(skipped as u64, std::sync::atomic::Ordering::Relaxed) + skipped as u64;
                    debug!(
                        lagged_samples = skipped,
                        lag_event_number = lag_count,
                        total_samples_skipped = total_skipped,
                        "BROADCAST LAG: Consumer falling behind sender"
                    );
                    continue;
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    debug!("BroadcastSource: channel closed");
                    return Ok(BlockRet::EOF);
                }
            }
        }

        if n > 0 {
            out.produce(n, &[]);
            if samples_received > 0 && samples_received % 1000 == 0 {
                debug!("BroadcastSource: received {} samples", samples_received);
            }
            Ok(BlockRet::Again)
        } else {
            // No samples available - yield thread to avoid busy wait
            std::thread::yield_now();
            Ok(BlockRet::Again)
        }
    }
}
