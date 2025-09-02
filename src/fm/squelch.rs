use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Float, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tracing::debug;

#[derive(Debug, Clone, Copy)]
pub enum SquelchState {
    Learning,
    PassThrough,
    Blocked,
}

/// Audio squelch block that analyzes audio content and determines if signal contains
/// audio or just noise. If noise is detected, blocks output and triggers early exit.
pub struct SquelchBlock {
    input: ReadStream<Float>,
    output: WriteStream<Float>,
    state: SquelchState,
    noise_detected: Arc<AtomicBool>,

    // Audio analysis parameters
    _sample_rate: f32,
    _learning_duration: f32,
    samples_analyzed: usize,
    learning_samples_needed: usize,
    analysis_completed: bool,

    // Detection algorithm state
    sample_buffer: Vec<f32>,
    rms_accumulator: f32,
    zero_crossings: usize,
    prev_sample_sign: bool,

    // Thresholds for audio detection
    min_rms_threshold: f32,
    max_zero_crossing_rate: f32,
    min_zero_crossing_rate: f32,
}

impl SquelchBlock {
    pub fn new(
        input: ReadStream<Float>,
        sample_rate: f32,
        learning_duration: f32,
    ) -> (Self, ReadStream<Float>, Arc<AtomicBool>) {
        let (output, output_stream) = WriteStream::new();
        let learning_samples_needed = (sample_rate * learning_duration) as usize;
        let noise_detected = Arc::new(AtomicBool::new(false));

        let block = Self {
            input,
            output,
            state: SquelchState::Learning,
            noise_detected: noise_detected.clone(),
            _sample_rate: sample_rate,
            _learning_duration: learning_duration,
            samples_analyzed: 0,
            learning_samples_needed,
            analysis_completed: false,
            sample_buffer: Vec::with_capacity(learning_samples_needed),
            rms_accumulator: 0.0,
            zero_crossings: 0,
            prev_sample_sign: false,

            // Tuned thresholds based on real measurements
            min_rms_threshold: 0.15, // Audio threshold (88.2 noise ~0.13, real stations 0.18-0.22)
            max_zero_crossing_rate: 0.05, // Audio has lower ZC rate than noise (0.034 vs 0.066)
            min_zero_crossing_rate: 0.01, // Very low minimum
        };

        (block, output_stream, noise_detected)
    }

    fn analyze_audio_content(&self) -> bool {
        if self.sample_buffer.is_empty() {
            return false;
        }

        // Calculate RMS power
        let rms = (self.rms_accumulator / self.sample_buffer.len() as f32).sqrt();

        // Calculate zero-crossing rate
        let zero_crossing_rate = self.zero_crossings as f32 / self.sample_buffer.len() as f32;

        // Calculate signal variance (measure of structure vs noise)
        let mean = self.sample_buffer.iter().sum::<f32>() / self.sample_buffer.len() as f32;
        let variance = self
            .sample_buffer
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / self.sample_buffer.len() as f32;

        debug!(
            "Squelch analysis - RMS: {:.6}, ZC rate: {:.3}, Variance: {:.6}, Mean: {:.6}, Samples: {}",
            rms,
            zero_crossing_rate,
            variance,
            mean,
            self.sample_buffer.len()
        );

        // Audio detection based on measured values:
        // Real stations: RMS ~0.21, Variance ~0.045, ZC ~0.034
        // Noise: RMS ~0.13, Variance ~0.018, ZC ~0.066
        let has_sufficient_power = rms > self.min_rms_threshold;
        let has_structure = variance > 0.025; // Structure threshold between noise (0.018) and audio (0.045)
        let reasonable_zero_crossings = zero_crossing_rate >= self.min_zero_crossing_rate
            && zero_crossing_rate <= self.max_zero_crossing_rate;

        // All three criteria must be met for audio detection
        let is_audio = has_sufficient_power && has_structure && reasonable_zero_crossings;

        debug!(
            "Squelch decision: {} (power: {}, crossings: {}, structure: {})",
            if is_audio { "AUDIO" } else { "NOISE" },
            has_sufficient_power,
            reasonable_zero_crossings,
            has_structure
        );

        is_audio
    }

    fn process_sample_for_analysis(&mut self, sample: f32) {
        // Add to buffer for later analysis
        self.sample_buffer.push(sample);

        // Accumulate RMS calculation
        self.rms_accumulator += sample * sample;

        // Count zero crossings
        let current_sign = sample >= 0.0;
        if self.samples_analyzed > 0 && current_sign != self.prev_sample_sign {
            self.zero_crossings += 1;
        }
        self.prev_sample_sign = current_sign;

        self.samples_analyzed += 1;

        // Check if learning period is complete and analysis hasn't been done yet
        if self.samples_analyzed >= self.learning_samples_needed && !self.analysis_completed {
            let is_audio = self.analyze_audio_content();
            self.analysis_completed = true;

            if is_audio {
                debug!("Squelch: Audio detected, enabling passthrough");
                self.state = SquelchState::PassThrough;
            } else {
                debug!("Squelch: Noise detected, blocking output and signaling early exit");
                self.state = SquelchState::Blocked;

                // Signal that noise was detected
                self.noise_detected.store(true, Ordering::Relaxed);
            }
        }
    }
}

impl BlockName for SquelchBlock {
    fn block_name(&self) -> &str {
        "SquelchBlock"
    }
}

impl BlockEOF for SquelchBlock {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for SquelchBlock {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let input_samples = input_buf.slice();

        if input_samples.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.input, 1));
        }

        match self.state {
            SquelchState::Learning => {
                // During learning, analyze samples but don't output yet
                let sample_count = input_samples.len();
                for &sample in input_samples {
                    self.process_sample_for_analysis(sample);
                }

                // Consume input but don't produce output during learning
                input_buf.consume(sample_count);
                Ok(BlockRet::Again)
            }

            SquelchState::PassThrough => {
                // Audio detected, pass through all samples
                let mut output_buf = self.output.write_buf()?;
                let to_copy = input_samples.len().min(output_buf.len());

                output_buf.slice()[..to_copy].copy_from_slice(&input_samples[..to_copy]);

                input_buf.consume(to_copy);
                output_buf.produce(to_copy, &[]);

                Ok(BlockRet::Again)
            }

            SquelchState::Blocked => {
                // Noise detected, consume input but don't produce output
                let sample_count = input_samples.len();
                input_buf.consume(sample_count);
                Ok(BlockRet::Again)
            }
        }
    }
}
