use crate::testing::AudioFileMetadata;
use crate::types::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::debug;

/// Metadata for I/Q files
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IqFileMetadata {
    pub sample_rate: f64,
    pub center_frequency: f64,
    pub capture_duration: f64,
    pub total_samples: usize,
    pub format: String,                // e.g., "f32_le_complex"
    pub expected_candidates: Vec<f64>, // Expected station frequencies in Hz

    // Peak detection parameters used during scanning
    pub fft_size: usize,
    pub peak_detection_threshold: f32,
    pub peak_scan_duration: Option<f64>,
    pub driver: String, // SDR driver used (e.g., "driver=sdrplay")
}

impl IqFileMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        capture_duration: f64,
        total_samples: usize,
        fft_size: usize,
        peak_detection_threshold: f32,
        peak_scan_duration: Option<f64>,
        driver: String,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            capture_duration,
            total_samples,
            format: "f32_le_complex".to_string(),
            expected_candidates: Vec::new(),
            fft_size,
            peak_detection_threshold,
            peak_scan_duration,
            driver,
        }
    }

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

/// Capturing wrapper for audio samples - saves demodulated audio to file for squelch testing
pub struct AudioCaptureSink {
    samples_captured: usize,
    max_samples: usize,
    sample_rate: f32,
    output_file: String,
    capture_duration: f64,
    squelch_learning_duration: f32,
    frequency_hz: f64,
    center_freq: f64,
    driver: String,
    writer: Option<BufWriter<File>>,
}

impl AudioCaptureSink {
    pub fn new(
        output_file: &str,
        sample_rate: f32,
        capture_duration: f64,
        squelch_learning_duration: f32,
        frequency_hz: f64,
        center_freq: f64,
        driver: String,
    ) -> crate::types::Result<Self> {
        let max_samples = (sample_rate * capture_duration as f32) as usize;

        let file = File::create(output_file)?;
        let writer = BufWriter::new(file);

        debug!(
            message = "Starting audio capture",
            output_file = output_file,
            capture_duration = capture_duration,
            max_samples = max_samples,
            sample_rate = sample_rate
        );

        Ok(Self {
            samples_captured: 0,
            max_samples,
            sample_rate,
            output_file: output_file.to_string(),
            capture_duration,
            squelch_learning_duration,
            frequency_hz,
            center_freq,
            driver,
            writer: Some(writer),
        })
    }

    /// Capture audio samples to file
    pub fn capture_samples(&mut self, samples: &[f32]) -> crate::types::Result<()> {
        if let Some(ref mut writer) = self.writer
            && self.samples_captured < self.max_samples
        {
            let samples_to_capture = (self.max_samples - self.samples_captured).min(samples.len());

            for sample in samples.iter().take(samples_to_capture) {
                writer.write_all(&sample.to_le_bytes())?;
            }

            self.samples_captured += samples_to_capture;
        }

        Ok(())
    }

    /// Save metadata for the captured audio file
    fn save_metadata(&self) -> crate::types::Result<()> {
        let metadata = AudioFileMetadata::new(
            self.sample_rate,
            self.squelch_learning_duration, // Use squelch learning duration, not capture duration
            self.samples_captured,
            "unknown".to_string(), // Will need to be set by caller based on squelch decision
            format!(
                "Captured demodulated audio at {} Hz for {} seconds",
                self.sample_rate, self.capture_duration
            ),
            self.frequency_hz,
            self.center_freq,
            self.driver.clone(),
        );

        let metadata_path = self.output_file.replace(".audio", ".json");
        metadata.to_file(&metadata_path)?;

        debug!(
            message = "Audio metadata saved",
            metadata_file = metadata_path,
            total_samples = self.samples_captured
        );

        Ok(())
    }
}

impl Drop for AudioCaptureSink {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            debug!("Saving audio file and metadata");
            if let Err(e) = writer.into_inner() {
                tracing::error!("Failed to flush audio capture file on drop: {}", e);
            }
            if let Err(e) = self.save_metadata() {
                tracing::error!("Failed to save audio metadata on drop: {}", e);
            }
        }
    }
}

/// Audio capture block that saves audio samples to file while passing them through unchanged
pub struct AudioCaptureBlock {
    input: rustradio::stream::ReadStream<rustradio::Float>,
    output: rustradio::stream::WriteStream<rustradio::Float>,
    audio_capturer: Option<AudioCaptureSink>,
}

impl AudioCaptureBlock {
    pub fn new(
        input: rustradio::stream::ReadStream<rustradio::Float>,
        audio_capturer: Option<AudioCaptureSink>,
    ) -> (Self, rustradio::stream::ReadStream<rustradio::Float>) {
        let (output, output_stream) = rustradio::stream::WriteStream::new();

        let block = Self {
            input,
            output,
            audio_capturer,
        };

        (block, output_stream)
    }
}

impl rustradio::block::BlockName for AudioCaptureBlock {
    fn block_name(&self) -> &str {
        "AudioCaptureBlock"
    }
}

impl rustradio::block::BlockEOF for AudioCaptureBlock {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl rustradio::block::Block for AudioCaptureBlock {
    fn work(&mut self) -> rustradio::Result<rustradio::block::BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let input_samples = input_buf.slice();

        if input_samples.is_empty() {
            return Ok(rustradio::block::BlockRet::WaitForStream(&self.input, 1));
        }

        // Get output buffer
        let mut output_buf = self.output.write_buf()?;
        let to_copy = input_samples.len().min(output_buf.len());

        // Pass through all samples unchanged
        output_buf.slice()[..to_copy].copy_from_slice(&input_samples[..to_copy]);

        // Capture samples if requested
        if let Some(ref mut capturer) = self.audio_capturer
            && let Err(e) = capturer.capture_samples(&input_samples[..to_copy])
        {
            tracing::debug!("Audio capture error: {}", e);
        }

        input_buf.consume(to_copy);
        output_buf.produce(to_copy, &[]);

        Ok(rustradio::block::BlockRet::Again)
    }
}
