use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Float, Result};
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
        static DROP_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
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
        static INITIAL_BUFFER_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);

        let count = WORK_CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if !INITIAL_BUFFER_LOGGED.load(std::sync::atomic::Ordering::Relaxed) {
            let initial_buffer_len = self.receiver.len();
            if initial_buffer_len > 0 {
                debug!(
                    initial_buffered_samples = initial_buffer_len,
                    "BUFFER DIAGNOSTICS: BroadcastSource starting with buffered samples"
                );
            }
            INITIAL_BUFFER_LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
        }

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
                    static LAG_COUNTER: std::sync::atomic::AtomicUsize =
                        std::sync::atomic::AtomicUsize::new(0);
                    static TOTAL_SKIPPED: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let lag_count =
                        LAG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let total_skipped = TOTAL_SKIPPED
                        .fetch_add(skipped, std::sync::atomic::Ordering::Relaxed)
                        + skipped;
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

            if count > 0 && count.is_multiple_of(5000) {
                let remaining_buffer = self.receiver.len();
                if remaining_buffer > 1000 {
                    debug!(
                        work_count = count,
                        remaining_buffered = remaining_buffer,
                        "BUFFER DIAGNOSTICS: Still draining buffer"
                    );
                }
            }

            Ok(BlockRet::Again)
        } else {
            // No samples available - yield thread to avoid busy wait
            std::thread::yield_now();
            Ok(BlockRet::Again)
        }
    }
}

pub struct AudioDiagnostic {
    input: ReadStream<Float>,
    output: WriteStream<Float>,
    quality_adjustment: f32,
}

impl AudioDiagnostic {
    pub fn new(input: ReadStream<Float>, quality_adjustment: f32) -> (Self, ReadStream<Float>) {
        let (output, read_stream) = WriteStream::new();
        (
            Self {
                input,
                output,
                quality_adjustment,
            },
            read_stream,
        )
    }
}

impl BlockName for AudioDiagnostic {
    fn block_name(&self) -> &str {
        "AudioDiagnostic"
    }
}

impl BlockEOF for AudioDiagnostic {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for AudioDiagnostic {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        static SAMPLE_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();

        if samples.is_empty() {
            return Ok(BlockRet::Again);
        }

        let count = SAMPLE_COUNT.fetch_add(samples.len(), std::sync::atomic::Ordering::Relaxed);

        if count < 50000 && (count / 5000) != ((count + samples.len()) / 5000) {
            let mut max_val = f32::NEG_INFINITY;
            let mut min_val = f32::INFINITY;
            let mut exceeds_threshold_count = 0;

            let safe_threshold = 1.0 / self.quality_adjustment;

            for &sample in samples {
                max_val = max_val.max(sample);
                min_val = min_val.min(sample);

                if sample.abs() > safe_threshold {
                    exceeds_threshold_count += 1;
                }
            }

            let peak_magnitude = max_val.abs().max(min_val.abs());
            let would_clip_after_boost = peak_magnitude * self.quality_adjustment > 1.0;

            debug!(
                sample_count = count + samples.len(),
                max_audio_sample = max_val,
                min_audio_sample = min_val,
                quality_adjustment = self.quality_adjustment,
                safe_threshold = safe_threshold,
                exceeds_threshold_count = exceeds_threshold_count,
                would_clip_after_boost = would_clip_after_boost,
                "GAIN DIAGNOSTICS: Audio levels vs quality boost"
            );
        }

        let mut out = self.output.write_buf()?;
        let out_slice = out.slice();
        let n = samples.len().min(out_slice.len());
        out_slice[..n].copy_from_slice(&samples[..n]);

        input_buf.consume(n);
        out.produce(n, &[]);

        Ok(BlockRet::Again)
    }
}
