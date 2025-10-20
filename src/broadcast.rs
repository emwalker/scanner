use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Float, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tokio::sync::broadcast;
use tracing::debug;

/// A batch of samples transmitted as a single unit through broadcast channels.
/// Wraps samples in Arc to enable cheap cloning across multiple receivers.
#[derive(Clone, Debug)]
pub struct SamplePacket {
    samples: Arc<Vec<Complex>>,
}

impl SamplePacket {
    pub fn new(samples: Vec<Complex>) -> Self {
        Self {
            samples: Arc::new(samples),
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn as_slice(&self) -> &[Complex] {
        &self.samples
    }

    pub fn iter(&self) -> impl Iterator<Item = &Complex> {
        self.samples.iter()
    }
}

impl AsRef<[Complex]> for SamplePacket {
    fn as_ref(&self) -> &[Complex] {
        &self.samples
    }
}

/// Trait for types that can send sample packets
pub trait SampleSink: Send {
    fn send(
        &self,
        packet: SamplePacket,
    ) -> std::result::Result<(), broadcast::error::SendError<SamplePacket>>;
}

impl SampleSink for broadcast::Sender<SamplePacket> {
    fn send(
        &self,
        packet: SamplePacket,
    ) -> std::result::Result<(), broadcast::error::SendError<SamplePacket>> {
        self.send(packet).map(|_| ())
    }
}

pub struct BroadcastSink<S: SampleSink> {
    input: ReadStream<Complex>,
    sender: S,
    packet_size: usize,
    buffer: Vec<Complex>,
}

impl<S: SampleSink> BroadcastSink<S> {
    pub fn new(input: ReadStream<Complex>, sender: S, packet_size: usize) -> Self {
        Self {
            input,
            sender,
            packet_size,
            buffer: Vec::with_capacity(packet_size),
        }
    }
}

impl<S: SampleSink> BlockName for BroadcastSink<S> {
    fn block_name(&self) -> &str {
        "BroadcastSink"
    }
}

impl<S: SampleSink> BlockEOF for BroadcastSink<S> {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl<S: SampleSink> Block for BroadcastSink<S> {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();
        if samples.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.input, 1));
        }

        let mut consumed = 0;
        let mut packets_sent = 0;

        while consumed < samples.len() {
            let space_in_buffer = self.packet_size - self.buffer.len();
            let to_copy = space_in_buffer.min(samples.len() - consumed);

            self.buffer
                .extend_from_slice(&samples[consumed..consumed + to_copy]);
            consumed += to_copy;

            if self.buffer.len() >= self.packet_size {
                let buffer = std::mem::take(&mut self.buffer);
                self.buffer.reserve(self.packet_size);
                let packet = SamplePacket::new(buffer);

                match self.sender.send(packet) {
                    Ok(_) => packets_sent += 1,
                    Err(_) => {
                        debug!(
                            "broadcast channel issue (no receivers or full), consuming remaining samples"
                        );
                        consumed = samples.len();
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        break;
                    }
                }
            }
        }

        if packets_sent > 0 && packets_sent % 100 == 0 {
            debug!(
                packets_sent = packets_sent,
                samples_sent = packets_sent * self.packet_size,
                "BroadcastSink: sent packets"
            );
        }

        input_buf.consume(consumed);
        Ok(BlockRet::Again)
    }
}

pub struct BroadcastSource {
    output: WriteStream<Complex>,
    receiver: broadcast::Receiver<SamplePacket>,
    leftover: Option<(SamplePacket, usize)>,
}

impl BroadcastSource {
    pub fn new(receiver: broadcast::Receiver<SamplePacket>) -> (Self, ReadStream<Complex>) {
        let (output, read_stream) = WriteStream::new();
        (
            Self {
                output,
                receiver,
                leftover: None,
            },
            read_stream,
        )
    }

    fn log_initial_buffer_state(&self) {
        static INITIAL_BUFFER_LOGGED: AtomicBool = AtomicBool::new(false);
        if !INITIAL_BUFFER_LOGGED.load(Ordering::Relaxed) {
            let initial_buffer_len = self.receiver.len();
            if initial_buffer_len > 0 {
                debug!(
                    initial_buffered_packets = initial_buffer_len,
                    "BUFFER DIAGNOSTICS: BroadcastSource starting with buffered packets"
                );
            }
            INITIAL_BUFFER_LOGGED.store(true, Ordering::Relaxed);
        }
    }

    fn log_periodic_work_call(count: usize) {
        if count.is_multiple_of(10000) {
            debug!(count, "BroadcastSource work() called");
        }
    }

    fn log_lag_event(skipped: u64) -> (usize, u64) {
        static LAG_COUNTER: AtomicUsize = AtomicUsize::new(0);
        static TOTAL_SKIPPED: AtomicU64 = AtomicU64::new(0);
        let lag_count = LAG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        let total_skipped = TOTAL_SKIPPED.fetch_add(skipped, Ordering::Relaxed) + skipped;
        debug!(
            lagged_packets = skipped,
            lag_event_number = lag_count,
            total_packets_skipped = total_skipped,
            "BROADCAST LAG: Consumer falling behind sender"
        );
        (lag_count, total_skipped)
    }

    fn log_buffer_drain_status(&self, count: usize) {
        if count > 0 && count.is_multiple_of(5000) {
            let remaining_buffer = self.receiver.len();
            if remaining_buffer > 100 {
                debug!(
                    work_count = count,
                    remaining_buffered_packets = remaining_buffer,
                    "BUFFER DIAGNOSTICS: Still draining buffer"
                );
            }
        }
    }

    fn write_leftover_and_receive_packets(&mut self, out_slice: &mut [Complex]) -> Result<usize> {
        let mut written = 0;

        if let Some((packet, offset)) = &self.leftover {
            let remaining = &packet.as_slice()[*offset..];
            let to_write = remaining.len().min(out_slice.len());
            out_slice[..to_write].copy_from_slice(&remaining[..to_write]);
            written += to_write;

            if to_write >= remaining.len() {
                self.leftover = None;
            } else {
                self.leftover = Some((packet.clone(), offset + to_write));
            }
        }

        let mut packets_received = 0;
        let max_packets_per_work = 64;
        let mut empty_count = 0;
        while written < out_slice.len() && packets_received < max_packets_per_work {
            match self.receiver.try_recv() {
                Ok(packet) => {
                    packets_received += 1;
                    let samples = packet.as_slice();
                    let to_write = samples.len().min(out_slice.len() - written);
                    out_slice[written..written + to_write].copy_from_slice(&samples[..to_write]);
                    written += to_write;

                    if to_write < samples.len() {
                        self.leftover = Some((packet, to_write));
                        break;
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    empty_count += 1;
                    if empty_count > 10 {
                        debug!(
                            empty_count,
                            written,
                            requested = out_slice.len(),
                            "BroadcastSource: Many consecutive empty receives - possible busy wait"
                        );
                    }
                    break;
                }
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    Self::log_lag_event(skipped);
                    continue;
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    debug!("BroadcastSource: channel closed");
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "Channel closed",
                    )
                    .into());
                }
            }
        }

        if packets_received > 0 && packets_received % 100 == 0 {
            debug!(
                packets_received = packets_received,
                samples_received = written,
                "BroadcastSource: received packets"
            );
        }

        Ok(written)
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
        static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let drop_count = DROP_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        debug!(
            drop_count = drop_count,
            receiver_lagged = self.receiver.len(),
            "BroadcastSource dropped - receiver being cleaned up"
        );
    }
}

impl Block for BroadcastSource {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        static WORK_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        let count = WORK_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

        self.log_initial_buffer_state();
        Self::log_periodic_work_call(count);

        let mut out = self.output.write_buf()?;
        if out.is_empty() {
            debug!("BroadcastSource: output buffer empty, waiting");
            return Ok(BlockRet::WaitForStream(&self.output, 1));
        }

        let out_slice = out.slice();
        let written = match self.write_leftover_and_receive_packets(out_slice) {
            Ok(written) => written,
            Err(_) => return Ok(BlockRet::EOF),
        };

        if written > 0 {
            out.produce(written, &[]);
            self.log_buffer_drain_status(count);
            Ok(BlockRet::Again)
        } else {
            std::thread::sleep(std::time::Duration::from_millis(10));
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

impl AudioDiagnostic {
    fn should_log_diagnostics(count: usize, samples_len: usize) -> bool {
        count < 50000 && (count / 5000) != ((count + samples_len) / 5000)
    }

    fn analyze_and_log_samples(&self, samples: &[Float], count: usize) {
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
        static SAMPLE_COUNT: AtomicUsize = AtomicUsize::new(0);

        let (input_buf, _metadata) = self.input.read_buf()?;
        let samples = input_buf.slice();

        if samples.is_empty() {
            return Ok(BlockRet::Again);
        }

        let count = SAMPLE_COUNT.fetch_add(samples.len(), Ordering::Relaxed);

        if Self::should_log_diagnostics(count, samples.len()) {
            self.analyze_and_log_samples(samples, count);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustradio::block::{Block, BlockEOF};
    use rustradio::stream::WriteStream;
    use std::sync::{Arc, Mutex};

    #[derive(Default, Clone)]
    struct MockSampleSink {
        packets: Arc<Mutex<Vec<SamplePacket>>>,
    }

    impl MockSampleSink {
        fn packets(&self) -> Vec<SamplePacket> {
            self.packets.lock().unwrap().clone()
        }

        fn packet_count(&self) -> usize {
            self.packets.lock().unwrap().len()
        }

        fn total_samples(&self) -> usize {
            self.packets.lock().unwrap().iter().map(|p| p.len()).sum()
        }
    }

    impl SampleSink for MockSampleSink {
        fn send(
            &self,
            packet: SamplePacket,
        ) -> std::result::Result<(), broadcast::error::SendError<SamplePacket>> {
            self.packets.lock().unwrap().push(packet);
            Ok(())
        }
    }

    #[test]
    fn test_broadcast_sink_batches_samples_into_packets() {
        let (input, read_stream) = WriteStream::new();
        let mock_sink = MockSampleSink::default();
        let packet_size = 100;

        let mut sink = BroadcastSink::new(read_stream, mock_sink.clone(), packet_size);

        let samples: Vec<Complex> = (0..250)
            .map(|i| Complex::new(i as f32, (i * 2) as f32))
            .collect();
        input.write_buf().unwrap().slice()[..250].copy_from_slice(&samples);
        input.write_buf().unwrap().produce(250, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 2);
        assert_eq!(mock_sink.total_samples(), 200);

        let packets = mock_sink.packets();
        assert_eq!(packets[0].len(), 100);
        assert_eq!(packets[1].len(), 100);
        assert_eq!(packets[0].as_slice()[0], Complex::new(0.0, 0.0));
        assert_eq!(packets[1].as_slice()[0], Complex::new(100.0, 200.0));
    }

    #[test]
    fn test_broadcast_sink_buffers_partial_packet() {
        let (input, read_stream) = WriteStream::new();
        let mock_sink = MockSampleSink::default();
        let packet_size = 100;

        let mut sink = BroadcastSink::new(read_stream, mock_sink.clone(), packet_size);

        let samples: Vec<Complex> = (0..150).map(|i| Complex::new(i as f32, 0.0)).collect();
        input.write_buf().unwrap().slice()[..150].copy_from_slice(&samples);
        input.write_buf().unwrap().produce(150, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 1);

        let samples2: Vec<Complex> = (150..200).map(|i| Complex::new(i as f32, 0.0)).collect();
        input.write_buf().unwrap().slice()[..50].copy_from_slice(&samples2);
        input.write_buf().unwrap().produce(50, &[]);

        sink.work().unwrap();

        assert_eq!(mock_sink.packet_count(), 2);
        let packets = mock_sink.packets();
        assert_eq!(packets[1].as_slice()[0], Complex::new(100.0, 0.0));
        assert_eq!(packets[1].as_slice()[99], Complex::new(199.0, 0.0));
    }

    #[test]
    fn test_broadcast_sink_eof_propagation() {
        let (_input, read_stream) = WriteStream::new();
        let mock_sink = MockSampleSink::default();
        let mut sink = BroadcastSink::new(read_stream, mock_sink, 100);

        assert!(!sink.eof());
    }
}
