use rustradio::Result;
use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use std::time::{Duration, Instant};

/// Throttle block that limits sample rate to prevent CPU busy-waiting
///
/// Modeled after GNU Radio's throttle block. This prevents the graph from
/// consuming 100% CPU by rate-limiting samples to match the expected sample rate.
///
/// Without hardware rate limiting (e.g., SDR, audio device), processing blocks
/// will consume samples as fast as possible, spinning at 100% CPU. The throttle
/// inserts sleeps to maintain the target sample rate.
pub struct Throttle<T: Copy> {
    input: ReadStream<T>,
    output: WriteStream<T>,
    sample_rate: f64,
    samples_processed: u64,
    start_time: Instant,
}

impl<T: Copy> Throttle<T> {
    /// Create a new throttle block
    ///
    /// # Arguments
    /// * `input` - Input stream
    /// * `output` - Output stream
    /// * `sample_rate` - Target sample rate in samples/second
    pub fn new(input: ReadStream<T>, output: WriteStream<T>, sample_rate: f64) -> Self {
        Self {
            input,
            output,
            sample_rate,
            samples_processed: 0,
            start_time: Instant::now(),
        }
    }
}

impl<T: Copy> BlockName for Throttle<T> {
    fn block_name(&self) -> &str {
        "Throttle"
    }
}

impl<T: Copy> BlockEOF for Throttle<T> {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl<T: Copy + Send + Sync + 'static> Block for Throttle<T> {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _tags) = self.input.read_buf()?;
        let samples = input_buf.slice();

        if samples.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.input, 1));
        }

        let mut output_buf = self.output.write_buf()?;
        let out_slice = output_buf.slice();

        if out_slice.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.output, 1));
        }

        let to_copy = samples.len().min(out_slice.len());
        out_slice[..to_copy].copy_from_slice(&samples[..to_copy]);

        input_buf.consume(to_copy);
        output_buf.produce(to_copy, &[]);

        self.samples_processed += to_copy as u64;

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let expected_samples = (elapsed * self.sample_rate) as u64;

        if self.samples_processed > expected_samples {
            let samples_ahead = self.samples_processed - expected_samples;
            let sleep_duration = Duration::from_secs_f64(samples_ahead as f64 / self.sample_rate);

            if sleep_duration > Duration::from_micros(100) {
                std::thread::sleep(sleep_duration);
                // Return Pending to let graph scheduler sleep, not Again which prevents sleep
                return Ok(BlockRet::Pending);
            }
        }

        Ok(BlockRet::Again)
    }
}
