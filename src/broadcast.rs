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

impl Block for BroadcastSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let mut out = self.output.write_buf()?;
        if out.is_empty() {
            debug!("BroadcastSource: output buffer empty, waiting");
            return Ok(BlockRet::WaitForStream(&self.output, 1));
        }

        let mut n = 0;

        // Fill buffer with try_recv only (non-blocking)
        // This avoids hanging if no samples are available
        // Process more samples per call for efficiency with parallel consumers
        let batch_size = out.len().min(4096); // Process up to 4K samples per call for better throughput
        let mut samples_received = 0;
        for _ in 0..batch_size {
            match self.receiver.try_recv() {
                Ok(sample) => {
                    out.slice()[n] = sample;
                    n += 1;
                    samples_received += 1;
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    // Only log significant lag events to reduce spam
                    if skipped > 5000 {
                        debug!("BroadcastSource: lagged, skipped {} samples", skipped);
                    }
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
            debug!("BroadcastSource: no samples available, trying again");
            // No samples available right now - sleep briefly to avoid busy wait
            // Use shorter sleep for better audio responsiveness
            std::thread::sleep(std::time::Duration::from_micros(10)); // Reduced to 10 microseconds for better audio latency
            Ok(BlockRet::Again)
        }
    }
}
